import argparse
import os
import sys
import json
import cv2
import io
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, CLIPImageProcessor, AutoConfig

# LISA 相关导入
from model.LISA import LISAForCausalLM
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
    """缩放 (先缩小，再用三次插值放大回原尺寸，完美对齐 GT Mask)"""
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

def calculate_iou_f1(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    iou = intersection / union if union > 0 else 0.0
    f1 = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    return iou, f1

def parse_args():
    parser = argparse.ArgumentParser(description="LISA Mask Robustness Evaluation")
    parser.add_argument("--test_json", type=str, default="./datasets/AIGI-Holmes-Dataset/dataset/test_mini_400.jsonl")
    parser.add_argument("--image_root", type=str, default="./datasets/AIGI-Holmes-Dataset")
    parser.add_argument("--version", default="./checkpoints_stage2/full_half_hint_epoch2/merged_final")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    return parser.parse_args()

def preprocess(
    x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    # 1. 过滤测试集，只保留有 GT Mask 的假图
    print(f"📂 读取并过滤测试集: {args.test_json}")
    fake_data = []
    with open(args.test_json, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            mask_path = item.get("mask", "")
            if mask_path and len(mask_path.strip()) > 0:
                fake_data.append(item)
    print(f"✅ 找到 {len(fake_data)} 张带有真实掩码的伪造图像用于评估。")

    npr_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # 2. 加载 LISA 大模型
    print("🚀 加载 LISA 模型本体...")
    tokenizer = AutoTokenizer.from_pretrained(args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0] if "[SEG]" in tokenizer.get_vocab() else -1

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.half if args.precision == "fp16" else torch.float32
    cfg = AutoConfig.from_pretrained(args.version)
    cfg.npr_pretrained_path = None 

    model = LISAForCausalLM.from_pretrained(
        args.version, config=cfg, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, torch_dtype=torch_dtype
    ).to(device)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=torch_dtype, device=device)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    # 字典用于存储所有条件的结果
    all_results = {}

    # 3. 多条件评估循环
    for cond_name, cond_func in CONDITIONS.items():
        print(f"\n=======================================================")
        print(f" 🧪 开始评估掩码条件: {cond_name} ")
        print(f"=======================================================")
        
        total_iou = 0.0
        total_f1 = 0.0
        total_samples = 0
        valid_mask_samples = 0
        zero_mask_count = 0

        pbar = tqdm(fake_data, desc=f"Eval {cond_name}")
        for item in pbar:
            img_path = os.path.join(args.image_root, item["images"][0].lstrip("./"))
            mask_path = os.path.join(args.image_root, item["mask"].lstrip("./"))
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue
                
            total_samples += 1

            # 读取真实图像
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # 💡 应用扰动 (对输入图像施加攻击，破坏其频域/像素连续性)
            image_np = cond_func(image_np)
            
            original_size_list = [image_np.shape[:2]]

            # 读取 GT 掩码 (不受扰动影响，作为绝对的 Ground Truth)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask.shape[:2] != image_np.shape[:2]:
                gt_mask = cv2.resize(gt_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_mask = gt_mask > 127

            image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(device, dtype=torch_dtype)
            image_npr = npr_transform(Image.fromarray(image_np)).unsqueeze(0).to(device, dtype=torch_dtype)
            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]
            image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device, dtype=torch_dtype)

            # 💡 硬编码提示词，强制模型产生 [SEG] 并进行分割
            base_prompt = "Please determine whether this image is fake or real, and provide the reasons for your judgment. If it is a fake image, please also segment the forged area."
            oracle_hint = "[System Forensic Expert Hint: Low-level pixel analysis indicates the presence of AIGC forgery traces in this image.]"
            prompt = base_prompt + "\n" + oracle_hint
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            if args.use_mm_start_end:
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)

            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt").unsqueeze(0).to(device)
            model.get_model().current_images_npr = image_npr

            # 推理
            with torch.no_grad():
                output_ids, pred_masks = model.evaluate(
                    image_clip, image, input_ids, resize_list, original_size_list, max_new_tokens=8, tokenizer=tokenizer,
                )

            # 提取掩码并计算指标 (依然执行漏报剔除)
            if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
                pred_mask = pred_masks[0].detach().cpu().numpy()[0] > 0
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                
                iou, f1 = calculate_iou_f1(pred_mask, gt_mask)
                total_iou += iou
                total_f1 += f1
                valid_mask_samples += 1
            else:
                zero_mask_count += 1

            current_mean_iou = total_iou / valid_mask_samples if valid_mask_samples > 0 else 0.0
            current_mean_f1 = total_f1 / valid_mask_samples if valid_mask_samples > 0 else 0.0
            pbar.set_postfix({"IoU": f"{current_mean_iou:.3f}", "F1": f"{current_mean_f1:.3f}"})

        # 记录并保存当前条件的指标
        mean_iou = total_iou / valid_mask_samples if valid_mask_samples > 0 else 0.0
        mean_f1 = total_f1 / valid_mask_samples if valid_mask_samples > 0 else 0.0
        
        all_results[cond_name] = {
            "Mean_IoU": mean_iou,
            "Mean_F1": mean_f1,
            "Valid_Samples": valid_mask_samples,
            "Zero_Masks": zero_mask_count
        }

    # =================================================================
    # 打印全局鲁棒性评估总结表
    # =================================================================
    print("\n\n" + "="*85)
    print(" 🛡️ 伪造掩码分割鲁棒性测试 (Mask Robustness Evaluation) 最终报告 🛡️")
    print("="*85)
    print(f"| {'Condition':<15} | {'Mean IoU':<10} | {'Mean F1 (Dice)':<15} | {'Valid Masks':<12} | {'Missed (Zero)':<13} |")
    print("-" * 85)
    for cond_name, metrics in all_results.items():
        print(f"| {cond_name:<15} | {metrics['Mean_IoU']:.4f}     | {metrics['Mean_F1']:.4f}          | {metrics['Valid_Samples']:<12} | {metrics['Zero_Masks']:<13} |")
    print("="*85)
    print("📝 说明: \n - 'Valid Masks' 表示模型成功输出 [SEG] 并生成掩码的样本数。")
    print(" - 'Missed (Zero)' 表示漏报次数，这些样本已被剔除，不计入平均 IoU 和 F1。")

if __name__ == "__main__":
    main()