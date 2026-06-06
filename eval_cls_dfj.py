import argparse
import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, CLIPImageProcessor, AutoConfig

# LISA 相关导入
from model.LISA import LISAForCausalLM
from model.llava.model.resnet_expert import ResNetExpert
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args():
    parser = argparse.ArgumentParser(description="LISA Classification Evaluation")
    parser.add_argument("--test_json", type=str, default="/home/yz/myLISA_old/datasets/DeepfakeJudge/dfj-detect/data.jsonl")
    parser.add_argument("--image_root", type=str, default="/home/yz/myLISA_old/datasets/DeepfakeJudge/dfj-detect")
    parser.add_argument("--version", default="./checkpoints_stage2/full_epoch3_50hint/merged_final")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--npr_ckpt", type=str, default="./checkpoints/npr_stage1_augmented_best.pth")
    parser.add_argument("--disable_expert_hint", action="store_true", default=False, help="是否禁用专家提示词注入")
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    return parser.parse_args()


class NPRClassifierHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


class UnifiedNPRExpert(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.resnet = ResNetExpert(use_low_level="npr", pretrained=False)
        self.classifier = NPRClassifierHead(input_dim=512, num_classes=2)
        
        if ckpt_path is not None:
            print(f"✅ Loading pretrained NPR expert from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            msg1=self.resnet.load_state_dict(checkpoint['resnet'], strict=True)
            msg2=self.classifier.load_state_dict(checkpoint['classifier'], strict=True)
            print(f"✅ renset加载结果: {msg1} | classifier加载结果: {msg2}")
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        spatial_features = self.resnet(images)
        logits = self.classifier(spatial_features)
        expert_preds = logits.argmax(dim=1)
        return spatial_features, expert_preds


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def extract_predicted_class(text):
    """精准提取：匹配完整的预设句子"""
    text_lower = text.lower().strip()
    if "this is a fake image" in text_lower:
        return 1
    elif "this is a real image" in text_lower:
        return 0
        
    return 0 


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"📂 读取测试集文件: {args.test_json}")
    test_data = []
    with open(args.test_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    print(f"📊 数据加载完毕: 测试集={len(test_data)}")

    npr_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    if args.disable_expert_hint:
        print("⏭️ 已跳过提示词注入")
    else:
        print("🧠 正在初始化独立 NPR 专家网络进行提示词生成...")
        expert_model = UnifiedNPRExpert(ckpt_path=args.npr_ckpt).to(device)
        expert_model.eval()
        batch_size = 32
        for i in tqdm(range(0, len(test_data), batch_size), desc="Injecting Expert Hints"):
            batch_items = test_data[i : i + batch_size]
            img_tensors = []
            valid_indices = []
            
            for idx, item in enumerate(batch_items):
                raw_path = item["images"][0]
                clean_path = raw_path[2:] if raw_path.startswith("./") else raw_path
                img_path = os.path.join(args.image_root, clean_path)
                
                try:
                    image_np = cv2.imread(img_path)
                    if image_np is None: raise ValueError("None")
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(image_np)
                    img_tensors.append(npr_transform(pil_img))
                    valid_indices.append(idx)
                except Exception:
                    print("未读取到图片！！！")
                    batch_items[idx]["expert_hint"] = "[System Forensic Expert Hint: Failed to analyze low-level pixels.]"
            
            if img_tensors:
                batch_tensor = torch.stack(img_tensors).to(device)
                with torch.no_grad():
                    _, expert_preds = expert_model(batch_tensor)
                    preds = expert_preds.cpu().numpy()
                
                for valid_idx, pred in zip(valid_indices, preds):
                    if pred == 1:
                        hint = "[System Forensic Expert Hint: Low-level pixel analysis indicates the presence of AIGC forgery traces in this image.]"
                    else:
                        hint = "[System Forensic Expert Hint: Low-level pixel analysis indicates this is a natural image with no forged traces detected.]"
                    batch_items[valid_idx]["expert_hint"] = hint

        del expert_model
        torch.cuda.empty_cache()
        print("✅ 测试集专家提示词注入完成，显存已释放。")

    y_true_cls, y_pred_cls = [], []

    print("🚀 初始化 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.version, cache_dir=None, model_max_length=args.model_max_length, padding_side="right", use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    if "[SEG]" in tokenizer.get_vocab():
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        args.seg_token_idx = -1 

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.half if args.precision == "fp16" else torch.float32

    print(f"⚙️ 加载 Config: {args.version}")
    cfg = AutoConfig.from_pretrained(args.version)
    cfg.npr_pretrained_path = None 

    print("🚀 加载模型本体...")
    model = LISAForCausalLM.from_pretrained(
        args.version, config=cfg, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, torch_dtype=torch_dtype
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=device)

    model = model.to(device)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()
    print("✅ 模型加载完毕，准备纯分类评估！")


    pbar = tqdm(test_data, desc="Evaluating Classification")
    for idx, item in enumerate(pbar):
        # --- 获取 GT 标签：直接从 answer 字段读取 ---
        answer = str(item.get("answer", "")).strip().lower()
        gt_class = 1 if answer == "fake" else 0
        y_true_cls.append(gt_class)

        # --- 准备图像 ---
        raw_path = item["images"][0]
        clean_path = raw_path[2:] if raw_path.startswith("./") else raw_path
        img_path = os.path.join(args.image_root, clean_path)
        
        if not os.path.exists(img_path):
            print("原图找不到，为了不打断评估，默认记一次错误预测")
            y_pred_cls.append(1 - gt_class) 
            continue

        image_np = cv2.imread(img_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(device, dtype=torch_dtype)
        pil_img = Image.fromarray(image_np)
        image_npr = npr_transform(pil_img).unsqueeze(0).cuda().to(device, dtype=torch_dtype)
        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device, dtype=torch_dtype)
        
        base_prompt = ("Please determine whether this image is fake or real, and provide the reasons for your judgment. If it is a fake image, please also segment the forged area.")
        expert_hint = item.get("expert_hint", "")
        prompt = base_prompt + "\n" + expert_hint
 
        if DEFAULT_IMAGE_TOKEN not in prompt:
             prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        else:
             prompt = prompt.replace(prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip(), prompt.strip())
             
        if args.use_mm_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
        model.get_model().current_images_npr = image_npr
        # --- 模型推理 ---
        with torch.no_grad():
            output_ids, _ = model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=7, 
                tokenizer=tokenizer,
            )
            
        # --- 解析文本分类 ---
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        if idx < 5:
            gt_label_str = "Fake" if gt_class == 1 else "Real"
            # 使用 tqdm.write 可以在不打断进度条显示的情况下打印信息
            tqdm.write(f"\n[Sample {idx+1}] GT: {gt_label_str} | Output: {text_output.strip()}")
        pred_class = extract_predicted_class(text_output)
        y_pred_cls.append(pred_class)

        # 实时更新总 ACC 进度
        pbar.set_postfix({"Overall ACC": f"{accuracy_score(y_true_cls, y_pred_cls):.3f}"})


    # =============================================================
    # 打印超级详细的分类指标
    # =============================================================
    # 1. 计算总指标
    overall_acc = accuracy_score(y_true_cls, y_pred_cls)
    overall_f1 = f1_score(y_true_cls, y_pred_cls, average='macro')

    # 2. 计算各个类别的详细指标 (使用混淆矩阵和分类报告)
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    # cm 结构:
    # [ [True Negative (Real->Real), False Positive (Real->Fake)],
    #   [False Negative (Fake->Real), True Positive (Fake->Fake)] ]
    
    real_acc = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    fake_acc = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

    report = classification_report(y_true_cls, y_pred_cls, target_names=['Real (0)', 'Fake (1)'], output_dict=True)
    real_f1 = report['Real (0)']['f1-score']
    fake_f1 = report['Fake (1)']['f1-score']

    print("\n" + "="*50)
    print(" 🏆 测试集评估最终结果 (分类专用) 🏆")
    print("="*50)
    print(f"📌 测试集总数据量: {len(y_true_cls)} 张")
    print(f"   - 真实图像 (Real) 数量: {cm[0, 0] + cm[0, 1]}")
    print(f"   - 伪造图像 (Fake) 数量: {cm[1, 0] + cm[1, 1]}\n")

    print("📊 总体指标 (Overall Metrics):")
    print(f"   ✅ Total Accuracy (ACC) : {overall_acc:.4f}")
    print(f"   ✅ Total F1-Score (Macro): {overall_f1:.4f}\n")

    print("🟢 真实图像指标 (Real Class Metrics):")
    print(f"   ✅ Real Accuracy (Recall): {real_acc:.4f}  <- (准确认出真图的比例)")
    print(f"   ✅ Real F1-Score         : {real_f1:.4f}\n")

    print("🔴 伪造图像指标 (Fake Class Metrics):")
    print(f"   ✅ Fake Accuracy (Recall): {fake_acc:.4f}  <- (准确抓出假图的比例)")
    print(f"   ✅ Fake F1-Score         : {fake_f1:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()