import argparse
import os
import sys
import json
import random
import cv2
import io
import math
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

# =================================================================
# 🛡️ 鲁棒性测试：图像扰动函数
# =================================================================
def apply_jpeg_compression(image_np, quality):
    """JPEG 压缩"""
    img_pil = Image.fromarray(image_np)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def apply_resizing(image_np, scale):
    """缩放 (先缩小，再用三次插值放大回原尺寸，保留失真但对齐坐标)"""
    h, w = image_np.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    img_down = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_restored = cv2.resize(img_down, (w, h), interpolation=cv2.INTER_CUBIC)
    return img_restored

def apply_gaussian_noise(image_np, variance):
    """高斯噪声"""
    sigma = math.sqrt(variance)
    noise = np.random.normal(0, sigma, image_np.shape)
    noisy_image = image_np.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# 定义所有测试条件
CONDITIONS = {
    "JPEG_70": lambda x: apply_jpeg_compression(x, 70),
    "JPEG_80": lambda x: apply_jpeg_compression(x, 80),
    "Resize_0.50": lambda x: apply_resizing(x, 0.5),
    "Resize_0.75": lambda x: apply_resizing(x, 0.75),
    "Noise_Var5": lambda x: apply_gaussian_noise(x, 5),
    "Noise_Var10": lambda x: apply_gaussian_noise(x, 10),
}
# =================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LISA Classification Robustness Evaluation")
    parser.add_argument("--test_json", type=str, default="./datasets/AIGI-Holmes-Dataset/dataset/test_mini_400.jsonl")
    parser.add_argument("--image_root", type=str, default="./datasets/AIGI-Holmes-Dataset")
    parser.add_argument("--version", default="./checkpoints_stage2/full_half_hint_epoch2/merged_final")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--npr_ckpt", type=str, default="./checkpoints/npr_stage1_augmented_best.pth")
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
    text_lower = text.lower().strip()
    if "this is a fake image" in text_lower: return 1
    elif "this is a real image" in text_lower: return 0
    if "fake" in text_lower: return 1
    if "real" in text_lower: return 0
    return 0 

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    print(f"📂 读取测试集文件: {args.test_json}")
    test_data = []
    with open(args.test_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): 
                item = json.loads(line)
                item["expert_hints"] = {} # 初始化多条件提示词字典
                test_data.append(item)
    # 固定随机种子，保证每次跑 eval 抽到的 40% 都是同一批图片，方便对比不同模型的性能
    # random.seed(42) 
    # sample_size = int(len(test_data) * 0.4)
    # test_data = random.sample(test_data, sample_size)

    # print(f"📉 已随机截取 40% 的测试集，当前用于评估的数据量为: {len(test_data)} 条")
    # =================================================================
    # Stage 1: 初始化专家模型并为所有扰动条件生成 Hints
    # =================================================================
    print("🧠 正在初始化独立 NPR 专家网络进行提示词生成...")
    expert_model = UnifiedNPRExpert(ckpt_path=args.npr_ckpt).to(device)
    expert_model.eval()
    npr_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    batch_size = 64
    for cond_name, cond_func in CONDITIONS.items():
        print(f"\n🌀 [专家阶段] 注入扰动条件: {cond_name}")
        for i in tqdm(range(0, len(test_data), batch_size), desc=f"Injecting Hints ({cond_name})"):
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
                    
                    # 💡 应用扰动
                    image_np = cond_func(image_np)
        
                    pil_img = Image.fromarray(image_np)
                    img_tensors.append(npr_transform(pil_img))
                    valid_indices.append(idx)
                except Exception:
                    batch_items[idx]["expert_hints"][cond_name] = "[System Forensic Expert Hint: Failed to analyze low-level pixels.]"
            
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
                    batch_items[valid_idx]["expert_hints"][cond_name] = hint

    del expert_model
    torch.cuda.empty_cache()
    print("✅ 测试集所有条件专家提示词注入完成，显存已释放。")

    # =================================================================
    # Stage 2: LISA 主模型评估阶段
    # =================================================================
    print("🚀 初始化 Tokenizer 和 LISA 模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.version, cache_dir=None, model_max_length=args.model_max_length, padding_side="right", use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    if "[SEG]" in tokenizer.get_vocab():
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        args.seg_token_idx = -1 

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.half if args.precision == "fp16" else torch.float32

    cfg = AutoConfig.from_pretrained(args.version)
    cfg.npr_pretrained_path = None 

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

    # 存储所有条件的结果
    all_results = {}

    for cond_name, cond_func in CONDITIONS.items():
        print(f"\n=======================================================")
        print(f" 🧪 开始评估条件: {cond_name} ")
        print(f"=======================================================")
        
        y_true_cls, y_pred_cls = [], []
        pbar = tqdm(test_data, desc=f"Eval {cond_name}")
        
        for idx, item in enumerate(pbar):
            mask_path = item.get("mask", "")
            gt_class = 1 if (mask_path and len(mask_path.strip()) > 0) else 0
            y_true_cls.append(gt_class)

            raw_path = item["images"][0]
            clean_path = raw_path[2:] if raw_path.startswith("./") else raw_path
            img_path = os.path.join(args.image_root, clean_path)
            
            if not os.path.exists(img_path):
                y_pred_cls.append(1 - gt_class) 
                continue

            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # 💡 应用扰动
            image_np = cond_func(image_np)
            
            original_size_list = [image_np.shape[:2]]

            image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(device, dtype=torch_dtype)
            
            pil_img = Image.fromarray(image_np)
            image_npr = npr_transform(pil_img).unsqueeze(0).cuda().to(device, dtype=torch_dtype)
            
            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]
            image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device, dtype=torch_dtype)

            base_prompt = ("Please determine whether this image is fake or real, and provide the reasons for your judgment. If it is a fake image, please also segment the forged area.")
            expert_hint = item["expert_hints"][cond_name] # 提取当前扰动对应的 hint
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
            
            with torch.no_grad():
                output_ids, _ = model.evaluate(
                    image_clip, image, input_ids, resize_list, original_size_list,
                    max_new_tokens=7, tokenizer=tokenizer,
                )
                
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            
            # 仅在跑 Original 时打印前5个结果，防止刷屏
            if idx < 5 and cond_name == "Original":
                gt_label_str = "Fake" if gt_class == 1 else "Real"
                tqdm.write(f"\n[Sample {idx+1}] GT: {gt_label_str} | Output: {text_output.strip()}")
                
            pred_class = extract_predicted_class(text_output)
            y_pred_cls.append(pred_class)
            pbar.set_postfix({"ACC": f"{accuracy_score(y_true_cls, y_pred_cls):.3f}"})

        # 记录并保存指标
        acc = accuracy_score(y_true_cls, y_pred_cls)
        f1 = f1_score(y_true_cls, y_pred_cls, average='macro')
        report = classification_report(y_true_cls, y_pred_cls, target_names=['Real (0)', 'Fake (1)'], output_dict=True)
        
        all_results[cond_name] = {
            "ACC": acc,
            "Macro_F1": f1,
            "Real_F1": report['Real (0)']['f1-score'],
            "Fake_F1": report['Fake (1)']['f1-score']
        }

    # =================================================================
    # 打印全局鲁棒性评估总结表
    # =================================================================
    print("\n\n" + "="*80)
    print(" 🛡️ 鲁棒性测试 (Robustness Evaluation) 最终报告 🛡️")
    print("="*80)
    print(f"| {'Condition':<15} | {'Accuracy':<10} | {'Macro F1':<10} | {'Real F1':<10} | {'Fake F1':<10} |")
    print("-" * 80)
    for cond_name, metrics in all_results.items():
        print(f"| {cond_name:<15} | {metrics['ACC']:.4f}     | {metrics['Macro_F1']:.4f}     | {metrics['Real_F1']:.4f}     | {metrics['Fake_F1']:.4f}     |")
    print("="*80)

if __name__ == "__main__":
    main()