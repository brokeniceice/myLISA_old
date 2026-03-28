import os
import json
import cv2
import argparse
import swanlab
import random
import torch.nn as nn
from datetime import datetime
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F   
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence 
from functools import partial
import transformers
import torchvision.transforms.functional as TF
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import get_cosine_schedule_with_warmup
transformers.utils.import_utils._torch_flash_attention_2_available = True
from transformers import CLIPImageProcessor
from peft import LoraConfig, get_peft_model


from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib 
from model.llava.mm_utils import tokenizer_image_token
from model.llava.constants import IGNORE_INDEX
from model.segment_anything.utils.transforms import ResizeLongestSide
# =========================
# 参数配置
# =========================
def parse_args():
    parser = argparse.ArgumentParser("LISA Stage-2 Training")

    parser.add_argument("--image_root", type=str, default="./checkpoints_LISA/dataset")
    parser.add_argument("--train_json", type=str, default="./checkpoints_LISA/dataset/MMTD_Set/formatted_splits/train.json")
    parser.add_argument("--val_json", type=str, default="./checkpoints_LISA/dataset/MMTD_Set/formatted_splits/val.json")
    parser.add_argument("--llm_version", type=str, default="./checkpoints_LISA/checkpoints/LISA-7B-v1")
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_LISA")


    parser.add_argument("--batch_size", type=int, default=24, help="显存不够必须设为1")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="梯度累积步数，模拟大Batch")
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)


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
# Dataset 
# =========================
class MyDataset(Dataset):
    def __init__(self, data_list, tokenizer, image_processor, image_root="."):
        self.data = data_list
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.sam_transform = ResizeLongestSide(1024) 


        self.image_root = image_root
        self.temp_image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 图片处理
        raw_path = item["image"] 
        clean_path = raw_path[2:] if raw_path.startswith("./") else raw_path
        img_path = os.path.join(self.image_root, clean_path)
        if os.path.exists(img_path):
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            image_np = np.zeros((224, 224, 3), dtype=np.uint8)
            print("严重警告，未读取到图片")

        # ==========================================
        # 4. 为 SAM 分支生成特征和掩码
        # ==========================================
        image_clip = self.image_processor(image_np, return_tensors='pt')['pixel_values'][0]

        image_sam_resized = self.sam_transform.apply_image(image_np)
        resize_shape = image_sam_resized.shape[:2] 
        image_sam = preprocess(torch.from_numpy(image_sam_resized).permute(2, 0, 1).contiguous())
        
        # 构建 SAM 的目标掩码 Tensor
        mask_path = item.get("mask", "")
        if mask_path and isinstance(mask_path, str) and mask_path.strip():
            # 获取掩码绝对路径
            clean_mask_path = mask_path[2:] if mask_path.startswith("./") else mask_path
            abs_mask_path = os.path.join(self.image_root, clean_mask_path)
            
            if os.path.exists(abs_mask_path):
                # 读取全局掩码 (灰度图)
                mask_raw = cv2.imread(abs_mask_path, cv2.IMREAD_GRAYSCALE)
                # 使用最近邻插值缩放到 SAM 对应的 resize_shape (注意 cv2.resize 是 W, H 顺序)
                mask_resized = cv2.resize(mask_raw, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # 二值化并转换为 Tensor，形状扩展为 (1, H, W) 表示 1 个篡改目标
                mask_tensor = (torch.from_numpy(mask_resized) > 0).long()
                target_mask = mask_tensor.unsqueeze(0)
            else:
                print("没有找到掩码！！！")
                # 找不到掩码文件，当作真实图像处理，0 个目标
                target_mask = torch.zeros((0, resize_shape[0], resize_shape[1]), dtype=torch.int64)
        else:
            # 真实图像（无掩码）
            target_mask = torch.zeros((1, resize_shape[0], resize_shape[1]), dtype=torch.int64)
        

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
            "labels": targets,
            "masks": target_mask,
            "resize_shape": resize_shape
        }

def collate_fn(batch, pad_token_id=0):
    input_ids = [x["input_ids"] for x in batch]
    labels = [x["labels"] for x in batch]
    images_clip = [x["images_clip"] for x in batch]
    images_clip_stacked = torch.stack(images_clip)
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
    args.output_dir = f"{args.output_dir}/{time_str}"
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16" ,# A6000 完美支持 bf16，极大避免大模型 Loss 溢出！
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        swanlab.init(
            project="LISA", 
            name=f"Stage2-Finetuning_MultiGPU_{time_str}",
            config=vars(args)
        )

    print(f"📂 读取训练集文件: {args.train_json}")
    with open(args.train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f) 
    print(f"📊 数据加载完毕: 训练集={len(train_data)}")

    print(f"📂 读取验证集文件: {args.val_json}")
    with open(args.val_json, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
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
        device_map={"": accelerator.process_index}, 
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
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False 
    # 🔥 开启【非重入式】梯度检查点！
    # 既能省下海量显存，又绝对不会触发 DDP 的 marked as ready twice 报错！
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ], 
        bias="none", 
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(torch.bfloat16)

    trainable_keys = ["lora", "embed_tokens", "lm_head", "mask_decoder","text_hidden_fcs"]

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
    train_dataset = MyDataset(train_data, tokenizer, image_processor, args.image_root)
    val_dataset = MyDataset(val_data, tokenizer, image_processor, args.image_root)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=8,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )
    
    # 应用分层学习率 
    lora_params = []
    embed_head_params = []
    pretrained_decoder_params = []
    new_projector_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # 1. LoRA 权重：外挂的新脑容量，可以激进学习 (2e-4)
        if "lora" in name:
            lora_params.append(p)
            
        # 2. 词嵌入与输出头：LLM 的核心语言中枢，极其脆弱，极低学习率保护 (2e-5)
        elif "embed_tokens" in name or "lm_head" in name:
            embed_head_params.append(p)
            
        # 3. LISA 预训练老兵：已经懂分割了，只需要微调适应取证任务，降维保护！(2e-5)
        elif "mask_decoder" in name or "text_hidden_fcs" in name:
            pretrained_decoder_params.append(p) 
        else:
            new_projector_params.append(p) # 兜底

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr}, 
        {"params": embed_head_params, "lr": args.lr * 0.1}, 
        {"params": pretrained_decoder_params, "lr": args.lr * 0.1}, # 🔥 压低 10 倍！保护 LISA 的 SAM 先验知识
        {"params": new_projector_params, "lr": args.lr}             # 🔥 保持基础高学习率！专攻 NPR 特征翻译
    ], weight_decay=0.01)

    print("\n⚙️ 优化器已启用分层学习率 (Layer-wise LR)！")

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * 0.05) 
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 让 Accelerator 接管所有组件！
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        print(f"🚀 开始多卡分布式训练! 每张卡 Batch={args.batch_size}, 梯度累积={args.grad_accum_steps}")

    # =========================
    # Training Loop
    # =========================
    global_step = 0
    best_mask_f1 = 0.0  # 🔥 初始化最佳指标追踪
    
    for epoch in range(args.epochs):
        model.train() 
        total_loss, total_ce_loss, total_mask_loss = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_local_main_process)
        
        optimizer.zero_grad() 
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                masks_list = [m.to(device, dtype=torch.bfloat16) for m in batch["masks_list"]] if "masks_list" in batch else None
                images_sam = batch["images_sam"].to(device)
                images_clip = batch["images_clip"].to(device)
                # 注意：LISA 内部通常需要 offset 来处理变长序列
                offset = torch.arange(images_sam.shape[0] + 1, dtype=torch.long, device=device)
                
                
                outputs = model(
                    input_ids=batch["input_ids"], attention_masks=batch["attention_mask"],
                    images_clip=images_clip, images=images_sam, labels=batch["labels"],
                    offset=offset, masks_list=masks_list, label_list=batch["label_list"],
                    resize_list=batch["resize_list"], inference=False
                )

                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()

                # 统计数据记录
                ce_val = outputs.get("ce_loss", loss).item()
                mask_val = outputs.get("mask_loss", torch.tensor(0.0)).item()
                total_loss += loss.item()
                total_ce_loss += ce_val
                total_mask_loss += mask_val

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    swanlab.log({
                        "Train/Total_Loss": loss.item() * args.grad_accum_steps,
                        "Train/Text_Loss": ce_val,
                        "Train/Mask_Loss": mask_val,
                        "Train/Learning_Rate": scheduler.get_last_lr()[0]
                    }, step=global_step)
                    pbar.set_postfix({"Loss": f"{loss.item()*args.grad_accum_steps:.4f}", "Mask": f"{mask_val:.4f}"})

        # --- 验证环节 ---
        model.eval() 
        val_metrics = torch.zeros(7, device=device) # [loss, ce, mask, inter, union, pred, gt]
        pbar_val = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", disable=not accelerator.is_main_process)
        
        with torch.no_grad():
            for batch in pbar_val:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(
                    input_ids=batch["input_ids"], attention_masks=batch["attention_mask"],
                    images_clip=batch["images_clip"], images=batch["images_sam"], labels=batch["labels"],
                    offset=torch.arange(batch["images_sam"].shape[0] + 1, device=device),
                    masks_list=[m.to(device, dtype=torch.bfloat16) for m in batch["masks_list"]],
                    label_list=batch["label_list"], resize_list=batch["resize_list"], inference=False 
                )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                ce_loss_t = outputs.get("ce_loss", loss) if isinstance(outputs, dict) else getattr(outputs, "ce_loss", loss)
                mask_loss_t = outputs.get("mask_loss", torch.tensor(0.0).to(device)) if isinstance(outputs, dict) else getattr(outputs, "mask_loss", torch.tensor(0.0).to(device))

                pred_masks = outputs.get("pred_masks", None) if isinstance(outputs, dict) else getattr(outputs, "pred_masks", None)
                # 累加指标数据
                val_metrics[0] += loss.item()
                val_metrics[1] += (ce_loss_t.item() if isinstance(ce_loss_t, torch.Tensor) else ce_loss_t)
                val_metrics[2] += (mask_loss_t.item() if isinstance(mask_loss_t, torch.Tensor) else mask_loss_t)
                

                if pred_masks is not None and batch["masks_list"] is not None:
                    for p_mask, gt_mask in zip(pred_masks, batch["masks_list"]):
                        p_bin, gt_bin = (p_mask > 0.0).float(), (gt_mask.to(device) > 0.5).float()
                        if gt_bin.sum() > 0:
                            inter = (p_bin * gt_bin).sum()
                            val_metrics[3] += inter
                            val_metrics[4] += (p_bin.sum() + gt_bin.sum() - inter)
                            val_metrics[5] += p_bin.sum()
                            val_metrics[6] += gt_bin.sum()

        # 🔥 多卡同步验证指标
        global_val_metrics = accelerator.reduce(val_metrics, reduction="sum")
        if accelerator.is_main_process:
            v_steps = len(val_loader) * accelerator.num_processes
            avg_val_loss = global_val_metrics[0].item() / v_steps
            epoch_mIoU = (global_val_metrics[3] / (global_val_metrics[4] + 1e-6)).item()
            epoch_mask_F1 = (2.0 * global_val_metrics[3] / (global_val_metrics[5] + global_val_metrics[6] + 1e-6)).item()
            
            print(f"🔍 Epoch {epoch+1} | Val F1: {epoch_mask_F1:.4f} | mIoU: {epoch_mIoU:.4f}")
            swanlab.log({"Val/Mask_F1": epoch_mask_F1, "Val/mIoU": epoch_mIoU}, step=epoch+1)

            # 🔥 保存逻辑：判断是否为最佳模型
            is_best = epoch_mask_F1 > best_mask_f1
            if is_best:
                best_mask_f1 = epoch_mask_F1
                save_path = os.path.join(args.output_dir, "best_model")
                print(f"🌟 发现最佳模型，正在保存至: {save_path}")
            else:
                save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            
            os.makedirs(save_path, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            
            # 1. 保存 PEFT
            unwrapped_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # 2. 保存全量权重（带 CPU 转换，防止显存爆炸）
            full_sd = {k: v.cpu() for k, v in unwrapped_model.state_dict().items() if "vision_tower" not in k}
            torch.save(full_sd, os.path.join(save_path, "full_state_dict.pth"))

    # --- 训练结束后的最终合并 ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("🎉 训练完成，正在合并最终 LoRA 权重...")
        final_save_path = os.path.join(args.output_dir, "merged_final")
        os.makedirs(final_save_path, exist_ok=True)
        
        # 重新加载 best 以确保最终导出的是最好的
        # (或者直接使用当前的 model，取决于你的需求)
        unwrapped_model = accelerator.unwrap_model(model)
        merged_model = unwrapped_model.merge_and_unload()
        
        # 导出合并后的全量 state_dict (转 CPU)
        final_sd = {k: v.cpu() for k, v in merged_model.state_dict().items() if "vision_tower" not in k}
        merged_model.save_pretrained(final_save_path, state_dict=final_sd)
        tokenizer.save_pretrained(final_save_path)
        print(f"✅ 最终模型已保存至: {final_save_path}")

if __name__ == "__main__":
    main()