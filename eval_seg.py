import argparse
import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


# =========================================================
# 掩码计算核心函数
# =========================================================
def calculate_iou_f1(pred_mask, gt_mask):
    """
    计算二值掩码的 IoU 和 F1 Score (Dice)
    要求 pred_mask 和 gt_mask 都是 numpy boolean 数组，且 shape 必须一致
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    iou = intersection / union if union > 0 else 0.0
    f1 = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    
    return iou, f1

def parse_args():
    parser = argparse.ArgumentParser(description="LISA Mask Evaluation")
    parser.add_argument("--test_json", type=str, default="./datasets/AIGI-Holmes-Dataset/dataset/test_mini_400.jsonl")
    parser.add_argument("--image_root", type=str, default="./datasets/AIGI-Holmes-Dataset")
    parser.add_argument("--version", default="./checkpoints_stage2/full_half_hint_epoch2/merged_final")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=1, type=int)
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


    # 3. 加载 LISA 大模型
    print("🚀 加载 LISA 模型本体...")
    from transformers import AutoTokenizer, CLIPImageProcessor, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0] if "[SEG]" in tokenizer.get_vocab() else -1

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.half if args.precision == "fp16" else torch.float32
    cfg = AutoConfig.from_pretrained(args.version)
    cfg.npr_pretrained_path = None 

    model = LISAForCausalLM.from_pretrained(args.version, config=cfg, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, torch_dtype=torch_dtype).to(device)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=torch_dtype, device=device)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    # 4. 评估循环
    total_iou = 0.0
    total_f1 = 0.0
    total_samples = 0       # 记录读取成功的总测试样本数
    valid_mask_samples = 0  # 记录成功生成掩码的样本数 (用于计算均值)
    zero_mask_count = 0     # 记录模型没有预测出掩码的次数 (LLM 漏掉 <SEG>)

    pbar = tqdm(fake_data, desc="Evaluating Masks")
    for item in pbar:
        img_path = os.path.join(args.image_root, item["images"][0].lstrip("./"))
        mask_path = os.path.join(args.image_root, item["mask"].lstrip("./"))
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
            
        total_samples += 1

        # 读取真实图像
        image_np = cv2.imread(img_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        # 读取 GT 掩码 (灰度图)，并转为 boolean
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 如果 GT mask 尺寸与原图不一致，强行 resize 对齐
        if gt_mask.shape[:2] != image_np.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask = gt_mask > 127

        image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(device, dtype=torch_dtype)
        image_npr = npr_transform(Image.fromarray(image_np)).unsqueeze(0).to(device, dtype=torch_dtype)
        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device, dtype=torch_dtype)

        # 构建 Prompt
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

        # 提取模型预测的掩码并计算指标
        if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
            # 拿到第一个预测掩码 (通常只有 1 个)
            pred_mask = pred_masks[0].detach().cpu().numpy()[0] > 0
            # 确保 shape 对齐 (evaluate 默认返回的就是 original_size_list 大小的 mask)
            if pred_mask.shape != gt_mask.shape:
                # 万一出现极小概率的尺寸对不齐，强制对齐防止报错
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            
            # ✅ 仅在成功生成掩码时，才计算并累加 IoU 和 F1
            iou, f1 = calculate_iou_f1(pred_mask, gt_mask)
            total_iou += iou
            total_f1 += f1
            valid_mask_samples += 1
        else:
            # ❌ LLM 没有输出 <SEG> 或者 SAM 生成失败 (漏报)，跳过指标计算
            zero_mask_count += 1
            # output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            # text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            # tqdm.write(f"\n⚠️ [漏报样本分析 - 序号 {total_samples}]")
            # tqdm.write(f"图片路径: {img_path}")
            # tqdm.write(f"🤖 模型输出: {text_output.strip()}")
            # tqdm.write("-" * 60)

        # 动态打印当前进度 (加入除 0 保护)
        current_mean_iou = total_iou / valid_mask_samples if valid_mask_samples > 0 else 0.0
        current_mean_f1 = total_f1 / valid_mask_samples if valid_mask_samples > 0 else 0.0
        pbar.set_postfix({"Mean IoU": f"{current_mean_iou:.4f}", "Mean F1": f"{current_mean_f1:.4f}"})

    # =========================================================
    # 打印最终全局指标
    # =========================================================
    if valid_mask_samples > 0:
        mean_iou = total_iou / valid_mask_samples
        mean_f1 = total_f1 / valid_mask_samples
        print("\n" + "="*50)
        print(" 🎯 伪造掩码分割评估最终结果 (Segmentation Metrics) 🎯")
        print("="*50)
        print(f"📌 参与测试总样本数: {total_samples} 张")
        print(f"✅ 成功生成掩码数  : {valid_mask_samples} 张")
        print(f"⚠️ 漏报次数 (未生成): {zero_mask_count} 次\n")
        print(f"   ✅ Mean IoU (交并比, 剔除漏报) : {mean_iou:.4f}")
        print(f"   ✅ Mean F1-Score (Dice, 剔除漏报): {mean_f1:.4f}")
        print("="*50)
    elif total_samples > 0:
        print("\n⚠️ 所有样本均漏报，未能成功生成任何掩码，无法计算 IoU 和 F1。")

if __name__ == "__main__":
    main()