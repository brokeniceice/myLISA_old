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

# NLP 评估指标库
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# LISA 相关导入
from transformers import AutoTokenizer, CLIPImageProcessor, AutoConfig
from model.LISA import LISAForCausalLM
from model.llava.model.resnet_expert import ResNetExpert
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

# 确保 nltk 词典已下载 (用于分词)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# ==========================================
# 专家网络定义 (NPR Expert)
# ==========================================
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
            msg1 = self.resnet.load_state_dict(checkpoint['resnet'], strict=True)
            msg2 = self.classifier.load_state_dict(checkpoint['classifier'], strict=True)
            print(f"✅ resnet加载结果: {msg1} | classifier加载结果: {msg2}")
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

def parse_args():
    parser = argparse.ArgumentParser(description="LISA Text Explanation Evaluation")
    # 🔥 默认测试集改为了你的 test_text.jsonl
    parser.add_argument("--test_json", type=str, default="./datasets/AIGI-Holmes-Dataset/dataset/test_mini_400.jsonl")
    parser.add_argument("--image_root", type=str, default="./datasets/AIGI-Holmes-Dataset")
    parser.add_argument("--version", default="./checkpoints_stage2/my_best_model/merged_final")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--npr_ckpt", type=str, default="./checkpoints/npr_stage1_augmented_best.pth")
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 1. 读取数据
    # ==========================================
    print(f"📂 读取测试集文件: {args.test_json}")
    test_data = []
    with open(args.test_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): 
                item = json.loads(line)
                if "response" in item: # 确保有真实文本用于对比
                    test_data.append(item)
    print(f"✅ 找到 {len(test_data)} 条包含 Ground Truth 文本的样本。")

    # ==========================================
    # 2. 注入专家提示词 (Expert Hint)
    # ==========================================
    print("\n🧠 正在初始化独立 NPR 专家网络进行提示词生成...")
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
                print(f"未读取到图片！！！路径: {img_path}")
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

    # 释放专家模型
    del expert_model
    torch.cuda.empty_cache()
    print("✅ 测试集专家提示词注入完成，显存已释放。\n")

    # ==========================================
    # 3. 加载 LISA (MLLM) 模型
    # ==========================================
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

    print("🚀 加载 LISA 模型本体...")
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
    print("✅ LISA 模型加载完毕，准备纯文本解释评估！\n")

    # ==========================================
    # 4. MLLM 推理生成长文本
    # ==========================================
    gt_texts = []
    pred_texts = []

    pbar = tqdm(test_data, desc="Generating Text Explanations")
    for idx, item in enumerate(pbar):
        # --- 准备文本 GT ---
        gt_text = item["response"]

        # --- 准备图像 ---
        raw_path = item["images"][0]
        clean_path = raw_path[2:] if raw_path.startswith("./") else raw_path
        img_path = os.path.join(args.image_root, clean_path)
        
        if not os.path.exists(img_path):
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

        # --- 构建包含 Expert Hint 的 Prompt ---
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
        
        # --- 模型推理 (🔥 赋予极大的 Token 额度) ---
        with torch.no_grad():
            output_ids, _ = model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=1500,  # 🔥 调大至 2048，足够输出千字长文
                tokenizer=tokenizer,
            )
            
        # --- 解析生成的文本 ---
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        # skip_special_tokens=True 会去掉 [CLS], [SEG] 等，留下纯人类语言
        raw_pred_text = tokenizer.decode(output_ids, skip_special_tokens=False).strip()

        # 🔥 核心清洗逻辑：切掉冗长的 Prompt，只保留 Assistant 生成的纯净回答
        # 动态获取当前 conversation 模板中 Assistant 的角色名 (通常是 'ASSISTANT:')
        assistant_marker = conv.roles[1] + ":" 
        
        if assistant_marker in raw_pred_text:
            pred_text = raw_pred_text.split(assistant_marker)[-1].strip()
        elif "ASSISTANT:" in raw_pred_text.upper(): # 增加鲁棒性防大小写问题
            import re
            pred_text = re.split(r'(?i)ASSISTANT:', raw_pred_text)[-1].strip()
        else:
            pred_text = raw_pred_text  # 兜底

        # 收集用于计算指标
        gt_texts.append(gt_text)
        pred_texts.append(pred_text)

        # 随机打印前几个作为检查
        if idx < 3:
            tqdm.write(f"\n[Sample {idx+1} Preview]")
            tqdm.write(f"  GT  : {gt_text[:100]}...")
            tqdm.write(f"  Pred: {pred_text[:100]}...")


    # ==========================================
    # 5. NLP 三大金刚指标计算
    # ==========================================
    if len(pred_texts) == 0:
        print("没有成功生成的文本数据！")
        return

    print("\n" + "="*50)
    print(" 🧠 开始计算自然语言理解 (NLU) 指标...")
    
    # [A] BLEU-4
    chencherry = SmoothingFunction().method1
    bleu_scores = []
    for ref, pred in zip(gt_texts, pred_texts):
        ref_tokens = nltk.word_tokenize(ref)
        pred_tokens = nltk.word_tokenize(pred)
        b_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=chencherry)
        bleu_scores.append(b_score)
    mean_bleu = np.mean(bleu_scores)
    print(f"✅ BLEU-4 评分计算完成: {mean_bleu:.4f}")

    # [B] ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    for ref, pred in zip(gt_texts, pred_texts):
        r_score = scorer.score(ref, pred)['rougeL'].fmeasure
        rouge_scores.append(r_score)
    mean_rouge = np.mean(rouge_scores)
    print(f"✅ ROUGE-L 评分计算完成: {mean_rouge:.4f}")

    # [C] BERTScore (余弦相似度)
    print("⏳ 正在计算 BERTScore (Cosine Similarity)...")
    P, R, F1 = bert_score_fn(pred_texts, gt_texts, lang="en", verbose=False)
    mean_bertscore = F1.mean().item()
    print(f"✅ BERTScore 评分计算完成: {mean_bertscore:.4f}")

    # ==========================================
    # 最终报告
    # ==========================================
    print("\n" + "="*50)
    print(" 🎯 文本解释能力评估最终结果 (Text Explanations) 🎯")
    print("="*50)
    print(f"📌 参与测试总样本数: {len(pred_texts)} 张")
    print(f"   🔹 Mean BLEU-4   (传统字面匹配): {mean_bleu:.4f}")
    print(f"   🔹 Mean ROUGE-L  (序列结构匹配): {mean_rouge:.4f}")
    print(f"   🔥 BERTScore-F1 (余弦语义相似度): {mean_bertscore:.4f}  <-- 最重要指标！")
    print("="*50)

if __name__ == "__main__":
    main()