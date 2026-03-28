import os
import json
import cv2
import argparse
import random
import swanlab
from datetime import datetime
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F   
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence 
from functools import partial
import transformers
from transformers import get_cosine_schedule_with_warmup
transformers.utils.import_utils._torch_flash_attention_2_available = True
from transformers import CLIPImageProcessor
from peft import LoraConfig, get_peft_model

# 确保路径正确导入
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib 
from model.llava.mm_utils import tokenizer_image_token
from model.llava.constants import IGNORE_INDEX
from model.segment_anything.utils.transforms import ResizeLongestSide
# =========================
# 参数配置
# =========================
def parse_args():
    parser = argparse.ArgumentParser("LISA Stage-2 Training (AIGI Format)")

    parser.add_argument("--image_root", type=str, default="./datasets/AIGI-Holmes-Dataset")
    parser.add_argument("--train_json", type=str, default="./datasets/AIGI-Holmes-Dataset/dataset/train.jsonl")
    parser.add_argument("--val_json", type=str, default="./datasets/AIGI-Holmes-Dataset/dataset/val.jsonl")
    parser.add_argument("--llm_version", type=str, default="/home/yz/myLISA/checkpoints/LISA-7B-v1")
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--npr_ckpt", type=str, default="./checkpoints/npr_stage1_augmented_best.pth")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage2/wo_hint_epoch2")


    parser.add_argument("--batch_size", type=int, default=8, help="显存不够必须设为1")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="梯度累积步数，模拟大Batch")
    
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


# =========================
# Dataset (适配 AIGI 格式)
# =========================
class AIGIDataset(Dataset):
    def __init__(self, data_list, tokenizer, image_processor, image_root="."):
        self.data = data_list
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.sam_transform = ResizeLongestSide(1024) 
        self.npr_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        self.image_root = image_root
        self.temp_image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 图片处理
        raw_path = item["images"][0]
        clean_path = raw_path[2:] if raw_path.startswith("./") else raw_path
        img_path = os.path.join(self.image_root, clean_path)
        if os.path.exists(img_path):
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            image_np = np.zeros((224, 224, 3), dtype=np.uint8)
            print("严重警告，未读取到图片")
        image_clip = self.image_processor(image_np, return_tensors='pt')['pixel_values'][0]

        pil_img = Image.fromarray(image_np)
        image_npr = self.npr_transform(pil_img)

        image_sam_resized = self.sam_transform.apply_image(image_np)
        resize_shape = image_sam_resized.shape[:2] 
        image_sam = preprocess(torch.from_numpy(image_sam_resized).permute(2, 0, 1).contiguous())
        
        # 2. Mask 处理
        mask_path = item.get("mask", "")
        target_mask = torch.zeros((0, resize_shape[0], resize_shape[1]), dtype=torch.int64)
        
        if mask_path and isinstance(mask_path, str) and len(mask_path.strip()) > 0:
            clean_mask_path = mask_path[2:] if mask_path.startswith("./") else mask_path
            abs_mask_path = os.path.join(self.image_root, clean_mask_path)
            
            if os.path.exists(abs_mask_path):
                try:
                    mask_img = Image.open(abs_mask_path).convert("L")
                    mask_img = mask_img.resize((resize_shape[1], resize_shape[0]), Image.NEAREST)
                    mask_np = np.array(mask_img)
                    mask_np = (mask_np > 0).astype(np.int64) 
                    
                    if mask_np.sum() > 0:
                        target_mask = torch.from_numpy(mask_np).unsqueeze(0) 
                except Exception as e:
                    pass

            else:
                print("严重警告，未读取到掩码")
        # 3. 清理数据并构建对话模板
        query = item["query"]
        response = item["response"]


        if "<image>" not in query:
            query = "<image>\n" + query

        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], response)
        prompt = conv.get_prompt() 
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
       
        # 4. 构建 Labels
        targets = input_ids.clone()
        conv_eval = conversation_lib.conv_templates["llava_v1"].copy()
        conv_eval.append_message(conv_eval.roles[0], query)
        conv_eval.append_message(conv_eval.roles[1], None) 
        prompt_eval = conv_eval.get_prompt()
        
        instruction_len = len(tokenizer_image_token(prompt_eval, self.tokenizer))
        targets[:instruction_len] = IGNORE_INDEX 
        
        MAX_LEN = 1792
        if input_ids.shape[0] >= MAX_LEN:
            input_ids = input_ids[:MAX_LEN - 1]
            targets = targets[:MAX_LEN - 1]
            input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id])])
            targets = torch.cat([targets, torch.tensor([self.tokenizer.eos_token_id])])
        
        return {
            "input_ids": input_ids,
            "images_clip": image_clip,
            "images_sam": image_sam,
            "images_npr": image_npr,
            "labels": targets,
            "masks": target_mask,
            "resize_shape": resize_shape
        }

def collate_fn(batch, pad_token_id=0):
    input_ids = [x["input_ids"] for x in batch]
    labels = [x["labels"] for x in batch]
    images_clip = [x["images_clip"] for x in batch]
    images_clip_stacked = torch.stack(images_clip)
    images_npr = [x["images_npr"] for x in batch]
    images_npr_stacked = torch.stack(images_npr)
    images_sam = [x["images_sam"] for x in batch]
    images_sam_stacked = torch.stack(images_sam)
    
    masks_list = []
    resize_list = []
    label_list = [] 
    
    for x in batch:
        masks_list.append(x["masks"]) 
        label_list.append(x["masks"]) 
        resize_list.append(x["resize_shape"])

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = input_ids_padded.ne(pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "images_clip": images_clip_stacked,
        "images_sam": images_sam_stacked, 
        "images_npr": images_npr_stacked,
        "labels": labels_padded,
        "masks_list": masks_list,    
        "label_list": label_list,        
        "resize_list": resize_list   
    }


# =========================
# 主函数
# =========================
def main():
    args = parse_args()
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{time_str}"
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    swanlab.init(
        project="myLISA", 
        name=f"Stage2-Finetuning_wo_hint_{time_str}",
        config=vars(args)
    )

    print(f"📂 读取训练集文件: {args.train_json}")
    train_data = []
    with open(args.train_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))

    print(f"📊 数据加载完毕: 训练集={len(train_data)}")

    print(f"📂 读取验证集文件: {args.val_json}")
    val_data = []
    with open(args.val_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                val_data.append(json.loads(line))
    print(f"📊 验证集加载完毕: {len(val_data)} 条数据")
    

    print("🚀 初始化 Tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.llm_version,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    tokens_to_add = ["<image>", "[SEG]"]
    tokens_to_add = [t for t in tokens_to_add if t not in tokenizer.get_vocab()]
    tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")

    print("🚀 初始化 LISA 模型...")
    config = transformers.AutoConfig.from_pretrained(args.llm_version)
    config.train_mask_decoder = True 
    config.npr_pretrained_path = args.npr_ckpt
    config.attention_bias = False
    for attr, default in [("attention_dropout", 0.0), ("rope_theta", 10000.0), ("intermediate_dropout", 0.0)]:
        if not hasattr(config, attr):
            setattr(config, attr, default)

    model_args = {
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": seg_token_idx,
    }

    model = LISAForCausalLM.from_pretrained(
        args.llm_version,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"":args.device},
        **model_args
    )
    
    if model.seg_token_idx != tokenizer.convert_tokens_to_ids('[SEG]'):
        model.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=device, dtype=torch.bfloat16)
    model.to(dtype=torch.bfloat16, device=device)
   
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(dtype=torch.bfloat16, device=device)
    trainable_keys = ["npr_projector","npr_cross_attn", "lora", "embed_tokens", "lm_head", "mask_decoder","text_hidden_fcs"]

    from collections import defaultdict
    key_param_count = defaultdict(int)
    for name, param in model.named_parameters():
        for key in trainable_keys:
            if key in name:
                param.requires_grad = True
                key_param_count[key] += param.numel()
                break  # 防止一个参数被多个 key 重复统计

    # 打印每个模块参数量
    print("\n📊 各模块可训练参数量：")
    total = 0
    for key in trainable_keys:
        count = key_param_count[key]
        total += count
        print(f"{key:20s}: {count / 1e6:8.2f} M")

    print(f"\n📊 总可训练参数量: {total / 1e6:.2f} M")
    model.resize_token_embeddings(len(tokenizer))

    image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    train_dataset = AIGIDataset(train_data, tokenizer, image_processor, args.image_root)
    val_dataset = AIGIDataset(val_data, tokenizer, image_processor, args.image_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
            val_dataset,
            batch_size=1, 
            shuffle=False, # 验证集不需要打乱
            num_workers=4,
            collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * 0.05) 
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # =========================
    # Training Loop
    # =========================
    print(f"🚀 开始训练! 总步数预估: {total_steps}")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train() 
        total_loss = 0.0
        total_ce_loss=0.0
        total_mask_loss=0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        optimizer.zero_grad() 
        
        for step, batch in enumerate(pbar):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            if "masks_list" in batch:
                # 无论是假图的 [1, H, W] 还是真图的保底 [0, H, W]，直接推到显存里！
                masks_list = [m.to(device) for m in batch["masks_list"]]
            else:
                masks_list = None
                
            input_ids=batch["input_ids"]
            images_sam = batch["images_sam"].to(device, dtype=torch.bfloat16)
            images_clip = batch["images_clip"].to(device, dtype=torch.bfloat16)
            current_bs = images_sam.shape[0]
            offset = torch.arange(current_bs + 1, dtype=torch.long, device=device)
      
            label_list = [l.to(device) if l is not None else None for l in batch["label_list"]]

            # 🔥 1. 取出并转换 NPR 图片的精度
            images_npr = batch["images_npr"].to(device, dtype=torch.bfloat16)
            
            # 🔥 2. 极其优雅的 Hack：把专属图片临时挂载到底层的 LlavaMetaModel 上！
            model.get_model().current_images_npr = images_npr
            outputs = model(
                input_ids=input_ids,
                attention_masks=batch["attention_mask"],
                images_clip=images_clip,
                images=images_sam,
                labels=batch["labels"],
                offset=offset,
                masks_list=masks_list,
                label_list=label_list,
                resize_list=batch["resize_list"],
                inference=False
            )

            if isinstance(outputs, dict):
                loss = outputs["loss"]
                ce_loss_t = outputs.get("ce_loss", loss)
                mask_loss_t = outputs.get("mask_loss", torch.tensor(0.0).to(loss.device))
            else:
                loss = outputs.loss
                ce_loss_t = getattr(outputs, "ce_loss", loss)
                mask_loss_t = getattr(outputs, "mask_loss", torch.tensor(0.0).to(loss.device))
            
            loss = loss / args.grad_accum_steps
            loss.backward()
            ce_loss_val = ce_loss_t.item() if isinstance(ce_loss_t, torch.Tensor) else ce_loss_t
            mask_loss_val = mask_loss_t.item() if isinstance(mask_loss_t, torch.Tensor) else mask_loss_t
            
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                global_step += 1

                swanlab.log({
                    "Train/Total_Loss": loss.item() * args.grad_accum_steps,
                    "Train/Text_Loss": ce_loss_val,
                    "Train/Mask_Loss": mask_loss_val,
                    "Train/Learning_Rate": scheduler.get_last_lr()[0]
                }, step=global_step)

            total_loss += loss.item() * args.grad_accum_steps
            total_ce_loss += ce_loss_val
            total_mask_loss += mask_loss_val
            
            pbar.set_postfix({
                "Loss": f"{loss.item() * args.grad_accum_steps:.4f}",
                "Text":  f"{ce_loss_val:.4f}",  
                "Mask":  f"{mask_loss_val:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_loss = total_loss / len(train_loader)
        avg_text_loss=total_ce_loss / len(train_loader)
        avg_mask_loss=total_mask_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} 训练完成 | Train Loss: {avg_loss:.4f} | Text Loss: {avg_text_loss:.4f} | Mask Loss: {avg_mask_loss:.4f}")

        model.eval() # 切换到验证模式，关闭 Dropout
        val_total_loss = 0.0
        val_total_ce_loss = 0.0
        val_total_mask_loss = 0.0
        
        pbar_val = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{args.epochs}")
        
        with torch.no_grad(): # 关闭梯度计算，节省显存并加速
            for batch in pbar_val:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
                if "masks_list" in batch:
                    masks_list = [m.to(device) for m in batch["masks_list"]]
                else:
                    masks_list = None
                    
                input_ids = batch["input_ids"]
                images_sam = batch["images_sam"].to(device, dtype=torch.bfloat16)
                images_clip = batch["images_clip"].to(device, dtype=torch.bfloat16)
                current_bs = images_sam.shape[0]
                offset = torch.arange(current_bs + 1, dtype=torch.long, device=device)
                label_list = [l.to(device) if l is not None else None for l in batch["label_list"]]
                images_npr = batch["images_npr"].to(device, dtype=torch.bfloat16)
                model.get_model().current_images_npr = images_npr
                
                outputs = model(
                    input_ids=input_ids,
                    attention_masks=batch["attention_mask"],
                    images_clip=images_clip,
                    images=images_sam,
                    labels=batch["labels"],
                    offset=offset,
                    masks_list=masks_list,
                    label_list=label_list,
                    resize_list=batch["resize_list"],
                    inference=False 
                )

                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                    ce_loss_t = outputs.get("ce_loss", loss)
                    mask_loss_t = outputs.get("mask_loss", torch.tensor(0.0).to(loss.device))
                else:
                    loss = outputs.loss
                    ce_loss_t = getattr(outputs, "ce_loss", loss)
                    mask_loss_t = getattr(outputs, "mask_loss", torch.tensor(0.0).to(loss.device))
                
                ce_loss_val = ce_loss_t.item() if isinstance(ce_loss_t, torch.Tensor) else ce_loss_t
                mask_loss_val = mask_loss_t.item() if isinstance(mask_loss_t, torch.Tensor) else mask_loss_t
                swanlab.log({
                "Val/Total_Loss": loss.item(),
                "Val/Text_Loss": ce_loss_val,
                "Val/Mask_Loss": mask_loss_val
            }) 
                
                # 验证集不涉及梯度累积，直接按正常 loss 累加
                val_total_loss += loss.item()
                val_total_ce_loss += ce_loss_val
                val_total_mask_loss += mask_loss_val
                
                pbar_val.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Text": f"{ce_loss_val:.4f}",  
                    "Mask": f"{mask_loss_val:.4f}"
                })
                
        # 计算平均验证集 Loss
        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_ce_loss = val_total_ce_loss / len(val_loader)
        avg_val_mask_loss = val_total_mask_loss / len(val_loader)
        
        print(f"🔍 Epoch {epoch+1} 验证完成 | Val Loss: {avg_val_loss:.4f} | Val Text: {avg_val_ce_loss:.4f} | Val Mask: {avg_val_mask_loss:.4f}")
        
   
        # Checkpoint 保存逻辑...
        save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # 合并导出逻辑...
    final_save_path = os.path.join(args.output_dir, "merged_final")
    os.makedirs(final_save_path, exist_ok=True)
    merged_model = model.merge_and_unload()
    full_state_dict = merged_model.state_dict()
    keys_to_save = {k: v.cpu() for k, v in full_state_dict.items() if "vision_tower" not in k}
    merged_model.save_pretrained(final_save_path, state_dict=keys_to_save)
    tokenizer.save_pretrained(final_save_path)
    print("🎉 Stage-2 训练与验证流程圆满结束！")

if __name__ == "__main__":
    main()