#lora微调大模型，冻结视觉塔和投影层，第一阶段npr专家网络训练完全根据官方github仓库的代码实现。
import os
import json
import cv2
import argparse
import swanlab
import random
from datetime import datetime
from accelerate import Accelerator, DistributedDataParallelKwargs
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
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPImageProcessor
from peft import LoraConfig, get_peft_model
from collections import defaultdict

# 确保路径正确导入
from model.LISA import LISAForCausalLM
from model.llava.model.resnet import resnet50
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
    parser.add_argument("--npr_ckpt", type=str, default="/home/yz/NPR-DeepfakeDetection/checkpoints/experiment_name2026_06_16_11_12_15/model_epoch_best.pth")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage2/office_npr")


    parser.add_argument("--batch_size", type=int, default=12, help="显存不够必须设为1")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="梯度累积步数，模拟大Batch")
    
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--adapter_lr", type=float, default=2e-4)
    parser.add_argument("--lm_head_lr", type=float, default=2e-5)
    parser.add_argument("--vision_lr", type=float, default=2e-5, help="视觉塔解冻高层的学习率")
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
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


def sync_vocab_size_in_config(config, vocab_size):
    config.vocab_size = vocab_size
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return
    if isinstance(text_config, dict):
        text_config["vocab_size"] = vocab_size
    else:
        setattr(text_config, "vocab_size", vocab_size)


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
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
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
            # ✅ 第一步：使用 PIL 读取，确保 NPR 对齐
            try:
                pil_img = Image.open(img_path).convert("RGB")
                image_np = np.array(pil_img) # 转换为 numpy 供 SAM 使用
            except Exception as e:
                print(f"读取图片异常 {img_path}: {e}")
                image_np = np.zeros((224, 224, 3), dtype=np.uint8)
                pil_img = Image.fromarray(image_np)
        else:
            image_np = np.zeros((224, 224, 3), dtype=np.uint8)
            pil_img = Image.fromarray(image_np)
            print("严重警告，未读取到图片")

        image_clip = self.image_processor(image_np, return_tensors='pt')['pixel_values'][0]
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

def reserve_vram(device, reserve_gb=44):
    """
    预占指定大小的显存，防止被其他进程抢占。
    参数:
        device: 当前所在的 GPU device
        reserve_gb: 需要预占的显存大小（GB）
    """
    try:
        print(f"🔒 正在预占 {reserve_gb} GB 显存，建立 PyTorch 缓存池...")
        # 1 GB = 1024^3 bytes。我们分配一个巨大的 int8 (1 byte) 的全零 Tensor
        dummy_tensor = torch.empty(int(reserve_gb * (1024 ** 3)), dtype=torch.int8, device=device)
        
        # 核心操作：删除这个 Tensor
        # 此时这 44GB 显存会回到 PyTorch 的 Caching Allocator 中
        # nvidia-smi 依然会显示这部分显存被占用，其他人的进程进不来
        del dummy_tensor
        print(f"✅ 成功霸占 {reserve_gb} GB 显存！")
    except RuntimeError as e:
        print(f"❌ 预占显存失败，请检查当前 GPU 是否还有 {reserve_gb} GB 空闲显存。")

# =========================
# 主函数
# =========================
def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16" ,
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{time_str}"

    if accelerator.is_main_process:
        print("🛡️ 启动预占显存机制...")
    reserve_vram(device, reserve_gb=46)

    if accelerator.is_main_process: 
        os.makedirs(args.output_dir, exist_ok=True)
        swanlab.init(
            project="myLISA", 
            name=f"Stage2-Finetuning_{time_str}",
            config=vars(args)
        )

    if accelerator.is_main_process: print(f"📂 读取训练集文件: {args.train_json}")
    train_data = []
    with open(args.train_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))
    if accelerator.is_main_process: print(f"📊 数据加载完毕: 训练集={len(train_data)}")

    if accelerator.is_main_process: print(f"📂 读取验证集文件: {args.val_json}")
    val_data = []
    with open(args.val_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                val_data.append(json.loads(line))
    if accelerator.is_main_process: print(f"📊 验证集加载完毕: {len(val_data)} 条数据")

    if accelerator.is_main_process: print("加载外部专家网络权重进行数据预处理...")
    expert_model = resnet50() 
    state_dict = torch.load(args.npr_ckpt, map_location='cpu')
    expert_model.load_state_dict(state_dict, strict=True)
    expert_model.to(device)
    expert_model.eval()

    # 与 NPR 严格一致的测试阶段图像预处理
    npr_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    def inject_expert_hints_batched(data_list, desc, batch_size=32):
        expert_y_true = []
        expert_y_pred = []
        # 使用 range 按 batch_size 步长进行切片
        for i in tqdm(range(0, len(data_list), batch_size), desc=desc):
            batch_items = data_list[i : i + batch_size]
            img_tensors = []
            valid_indices = [] 
            batch_gts = [] # 记录当前 Batch 的真实标签
            
            # 读取图片
            for idx, item in enumerate(batch_items):
                #获取当前图片的真实标签
                mask_path = item.get("mask", "")
                if mask_path and len(mask_path.strip()) > 0:
                    gt_class = 1 # Fake
                else:
                    gt_class = 0 # Real

                raw_path = item["images"][0]
                clean_path = raw_path[2:] if raw_path.startswith("./") else raw_path
                img_path = os.path.join(args.image_root, clean_path)
                
                try:
                    # ✅ 完美对齐官方 Stage-1 的读取引擎
                    pil_img = Image.open(img_path).convert('RGB')
                    img_tensor = npr_transform(pil_img)
                    img_tensors.append(img_tensor)
                    valid_indices.append(idx)
                    batch_gts.append(gt_class)
                except Exception as e:
                    print("error，未读取到图片")
            
            if not img_tensors:
                continue
                
            batch_tensor = torch.stack(img_tensors).to(device)
            # 推理
            with torch.no_grad():
                logits, _ = expert_model(batch_tensor)
                probs = torch.sigmoid(logits)
                expert_preds = (probs >= 0.5).int().flatten()
                preds = expert_preds.cpu().numpy() 
                
            expert_y_pred.extend(preds)
            expert_y_true.extend(batch_gts)

            for valid_idx, pred in zip(valid_indices, preds):
                item = batch_items[valid_idx]
                if pred == 1:
                    hint = "[System Forensic Expert Hint: Low-level pixel analysis indicates the presence of AIGC forgery traces in this image.]"
                else:
                    hint = "[System Forensic Expert Hint: Low-level pixel analysis indicates this is a natural image with no forged traces detected.]"
                #随机抹除 50% 的专家提示
                if random.random() < 0.5:
                    item["query"] = item["query"].rstrip() + "\n" + hint
        
        print("\n" + "="*50)
        print(" 🎯 提示词注入阶段: NPR 专家网络性能自检 🎯")
        print("="*50)
        expert_acc = accuracy_score(expert_y_true, expert_y_pred)
        expert_f1 = f1_score(expert_y_true, expert_y_pred, average='macro')
        print(f"✅ NPR 实际注入准确率 (Accuracy): {expert_acc * 100:.2f}%")
        print(f"✅ NPR 实际注入 F1 分数 (Macro): {expert_f1 * 100:.2f}%")
        print("="*50 + "\n")
        return data_list

    # 执行注入
    train_data = inject_expert_hints_batched(train_data, "注入训练集提示词", batch_size=64)
    val_data = inject_expert_hints_batched(val_data, "注入验证集提示词", batch_size=64)
    accelerator.wait_for_everyone()
    # 立刻把临时专家模型删掉，清空显存
    del expert_model
    torch.cuda.empty_cache()
    if accelerator.is_main_process: print("✅ 专家提示词注入完成，显存已释放。")
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process: print("🚀 初始化 Tokenizer...")
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

    if accelerator.is_main_process: print("🚀 初始化 LISA 模型...")
    config = transformers.AutoConfig.from_pretrained(args.llm_version)
    config.train_mask_decoder = True 
    config.npr_pretrained_path = args.npr_ckpt
    config.attention_bias = False
    config.vision_tower = args.vision_tower
    config.mm_vision_tower = args.vision_tower
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
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    exclude_list = [
        "vision_tower", "npr_projector", "npr_cross_attn", 
        "embed_tokens", "lm_head", "mask_decoder", "text_hidden_fcs"
    ]
    
    target_modules = []
    lora_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for name, module in model.named_modules():
        # 1. 排除名单
        if any(ex_key in name for ex_key in exclude_list):
            continue
            
        # 2. 大模型lora目标层
        if any(name.endswith(kw) for kw in lora_keywords):
            target_modules.append(name)

    if accelerator.is_main_process:
        print(f"🎯 过滤完成！共找到 {len(target_modules)} 个模块准备注入 LoRA。")

    # 注入 LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(dtype=torch.bfloat16, device=device)

    if accelerator.is_main_process:
        # 实时抓取刚刚生成的全部 lora 权重
        injected_lora_params = sum(param.numel() for name, param in model.named_parameters() if "lora" in name)
        
        print(f"🎯 检查：共找到 {len(target_modules)} 个 LoRA 模块。")
        print(f"💉 产生 LoRA 参数量: {injected_lora_params / 1e6:.2f} M")

    exclude_list = [
        "npr_projector", "npr_cross_attn", 
        "embed_tokens", "lm_head", "mask_decoder", "text_hidden_fcs"
    ]
    # 依次参数解冻其他所有模块
    for name, param in model.named_parameters():
        if any(key in name for key in exclude_list):
            param.requires_grad = True
    
    key_param_count = defaultdict(int)
    print_keys = ["llm_lora", "vision_lora"] + exclude_list

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # 没解冻的直接不统计
        
        if "lora" in name:
            if "vision_tower" in name:
                key_param_count["vision_lora"] += param.numel()
            else:
                key_param_count["llm_lora"] += param.numel()

        else:
            # 统计其他解冻的适配器
            for key in exclude_list:
                if key in name:
                    key_param_count[key] += param.numel()
                    break

    # 打印最终表格
    if accelerator.is_main_process:
        print("\n📊 各模块可训练参数量：")
        total = 0
        for key in print_keys:
            count = key_param_count[key]
            if count > 0:
                total += count
                print(f"{key:20s}: {count / 1e6:8.2f} M")
        print("-" * 35)
        print(f"📊 总可训练参数量  : {total / 1e6:8.2f} M\n")

    llm_lora_params = []
    vision_lora_params = []
    adapter_params = []
    lm_head_params = []
    fallback_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" in name:
            if "vision_tower" in name:
                vision_lora_params.append(param)
            else:
                llm_lora_params.append(param)
        
        elif any(keyword in name for keyword in ["npr_projector", "npr_cross_attn", "mask_decoder", "text_hidden_fcs"]):
            adapter_params.append(param)
        
        elif any(keyword in name for keyword in ["embed_tokens", "lm_head"]):
            lm_head_params.append(param)
        else:
            fallback_params.append(param)

    model.resize_token_embeddings(len(tokenizer))
    sync_vocab_size_in_config(model.config, len(tokenizer))
    sync_vocab_size_in_config(model.get_model().config, len(tokenizer))

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
    
    optimizer_param_groups = []
    # 大模型 LoRA 用 2e-4
    if llm_lora_params:
        optimizer_param_groups.append({"params": llm_lora_params, "lr": args.lora_lr})
    # 视觉塔 LoRA 用 2e-5 
    if vision_lora_params:
        optimizer_param_groups.append({"params": vision_lora_params, "lr": args.vision_lr})
    if adapter_params:
        optimizer_param_groups.append({"params": adapter_params, "lr": args.adapter_lr})
    if lm_head_params:
        optimizer_param_groups.append({"params": lm_head_params, "lr": args.lm_head_lr})
    if fallback_params:
        optimizer_param_groups.append({"params": fallback_params, "lr": args.lr})

    optimizer = torch.optim.AdamW(optimizer_param_groups, weight_decay=0.01)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * 0.05) 
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler) # 🔥 新增：包装学习率调度器
    # =========================
    # Training Loop
    # =========================
    if accelerator.is_main_process: print(f"🚀 开始训练! 总步数预估: {total_steps}")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train() 
        total_loss = 0.0
        total_ce_loss=0.0
        total_mask_loss=0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)
        
        optimizer.zero_grad() 
        
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                # ================= 准备数据 =================
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
                
                if "masks_list" in batch:
                    masks_list = [m.to(device, dtype=torch.bfloat16) for m in batch["masks_list"]]
                else:
                    masks_list = None
                    
                input_ids = batch["input_ids"]
                images_sam = batch["images_sam"].to(device, dtype=torch.bfloat16)
                images_clip = batch["images_clip"].to(device, dtype=torch.bfloat16)
                current_bs = images_sam.shape[0]
                offset = torch.arange(current_bs + 1, dtype=torch.long, device=device)
                label_list = [l.to(device) if l is not None else None for l in batch["label_list"]]
                images_npr = batch["images_npr"].to(device, dtype=torch.bfloat16)
                
                accelerator.unwrap_model(model).get_model().current_images_npr = images_npr
                
                # ================= 前向传播 =================
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
                    mask_loss_t = outputs.get("mask_loss", torch.tensor(0.0).to(device))
                else:
                    loss = outputs.loss
                    ce_loss_t = getattr(outputs, "ce_loss", loss)
                    mask_loss_t = getattr(outputs, "mask_loss", torch.tensor(0.0).to(device))
                
                # ================= 反向传播与更新 =================
                # 🔥 2. 删除了手动除以 grad_accum_steps 的代码，直接传原始 loss
                accelerator.backward(loss)
                
                # 🔥 3. 只有在跨卡同步完成的那一步，才进行梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 🔥 4. 这三个 step 函数直接写在这里，accelerator 会在底层自动决定要不要跳过
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                
                # ================= 提取数值用于记录 =================
                ce_loss_val = ce_loss_t.item() if isinstance(ce_loss_t, torch.Tensor) else ce_loss_t
                mask_loss_val = mask_loss_t.item() if isinstance(mask_loss_t, torch.Tensor) else mask_loss_t

            # ================= 日志与进度条记录 =================
            # 🔥 5. 只有在真正发生参数更新的那一步，才增加全局步数并记录到 Swanlab
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process: 
                    swanlab.log({
                        "Train/Total_Loss": loss.item(), # 🔥 6. 删除了乘以 grad_accum_steps，直接记真实的 loss
                        "Train/Text_Loss": ce_loss_val,
                        "Train/Mask_Loss": mask_loss_val,
                        "Train/Learning_Rate": scheduler.get_last_lr()[0]
                    }, step=global_step)

            # 更新累计指标
            total_loss += loss.item()
            total_ce_loss += ce_loss_val
            total_mask_loss += mask_loss_val
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Text":  f"{ce_loss_val:.4f}",  
                "Mask":  f"{mask_loss_val:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # ================= 循环结束后的 Epoch 结算 =================
        avg_loss = total_loss / len(train_loader)
        avg_text_loss = total_ce_loss / len(train_loader)
        avg_mask_loss = total_mask_loss / len(train_loader)
        
        if accelerator.is_main_process:
            print(f"✅ Epoch {epoch+1} 训练完成 | Train Loss: {avg_loss:.4f} | Text Loss: {avg_text_loss:.4f} | Mask Loss: {avg_mask_loss:.4f}")

        # ================= 验证环节 =================
        model.eval() 
        # 🔥 1. 创建一个 Tensor 来存储要累加的 Loss，方便跨卡通信
        val_metrics = torch.zeros(3, device=device) # [total_loss, total_ce_loss, total_mask_loss]
        
        pbar_val = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)
        
        with torch.no_grad(): # 关闭梯度计算，节省显存并加速
            for batch in pbar_val:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
                if "masks_list" in batch:
                    masks_list = [m.to(device, dtype=torch.bfloat16) for m in batch["masks_list"]]
                else:
                    masks_list = None
                    
                input_ids = batch["input_ids"]
                images_sam = batch["images_sam"].to(device, dtype=torch.bfloat16)
                images_clip = batch["images_clip"].to(device, dtype=torch.bfloat16)
                current_bs = images_sam.shape[0]
                offset = torch.arange(current_bs + 1, dtype=torch.long, device=device)
                label_list = [l.to(device) if l is not None else None for l in batch["label_list"]]
                images_npr = batch["images_npr"].to(device, dtype=torch.bfloat16)
                
                accelerator.unwrap_model(model).get_model().current_images_npr = images_npr
                
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
                    mask_loss_t = outputs.get("mask_loss", torch.tensor(0.0).to(device))
                else:
                    loss = outputs.loss
                    ce_loss_t = getattr(outputs, "ce_loss", loss)
                    mask_loss_t = getattr(outputs, "mask_loss", torch.tensor(0.0).to(device))
                
                ce_loss_val = ce_loss_t.item() if isinstance(ce_loss_t, torch.Tensor) else ce_loss_t
                mask_loss_val = mask_loss_t.item() if isinstance(mask_loss_t, torch.Tensor) else mask_loss_t
                
                # 🔥 2. 将当前 batch 的 loss 累加到本卡的 Tensor 中
                val_metrics[0] += loss.item()
                val_metrics[1] += ce_loss_val
                val_metrics[2] += mask_loss_val
                
                pbar_val.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Text": f"{ce_loss_val:.4f}",  
                    "Mask": f"{mask_loss_val:.4f}"
                })
                
        # 🔥 3. 【核心同步】把所有 GPU 上的 val_metrics 加在一起！
        global_val_metrics = accelerator.reduce(val_metrics, reduction="sum")
        
        if accelerator.is_main_process: 
            # 🔥 4. 计算全局真实的平均 Loss
            # len(val_loader) 是单卡上的批次数，总批次数需要乘以 GPU 数量
            total_val_batches = len(val_loader) * accelerator.num_processes
            
            avg_val_loss = global_val_metrics[0].item() / total_val_batches
            avg_val_ce_loss = global_val_metrics[1].item() / total_val_batches
            avg_val_mask_loss = global_val_metrics[2].item() / total_val_batches
            
            # 🔥 5. 统一在这里打点日志，保证曲线平滑且不会冲突
            swanlab.log({
                "Val/Total_Loss": avg_val_loss,
                "Val/Text_Loss": avg_val_ce_loss,
                "Val/Mask_Loss": avg_val_mask_loss,
                "Epoch": epoch + 1
            }) 
            
            print(f"🔍 Epoch {epoch+1} 验证完成 | Val Loss: {avg_val_loss:.4f} | Val Text: {avg_val_ce_loss:.4f} | Val Mask: {avg_val_mask_loss:.4f}")

            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            # 🔥 修改：解包后再保存
            sync_vocab_size_in_config(accelerator.unwrap_model(model).config, len(tokenizer))
            sync_vocab_size_in_config(accelerator.unwrap_model(model).get_model().config, len(tokenizer))
            accelerator.unwrap_model(model).save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    if accelerator.is_main_process: # 🔥 新增：仅主进程执行最终合并导出
        final_save_path = os.path.join(args.output_dir, "merged_final")
        os.makedirs(final_save_path, exist_ok=True)
        # 🔥 修改：解包后再 merge
        merged_model = accelerator.unwrap_model(model).merge_and_unload()
        sync_vocab_size_in_config(merged_model.config, len(tokenizer))
        sync_vocab_size_in_config(merged_model.get_model().config, len(tokenizer))

        full_state_dict = merged_model.state_dict()
        keys_to_save = {k: v.cpu() for k, v in full_state_dict.items() if "vision_tower" not in k}
        merged_model.save_pretrained(final_save_path, state_dict=keys_to_save)
        tokenizer.save_pretrained(final_save_path)

        # updated_vision_tower_path = os.path.join(final_save_path, "updated_vision_tower")
        # raw_vision_tower = merged_model.get_model().get_vision_tower()
        # if hasattr(raw_vision_tower, "vision_tower"):
        #     raw_vision_tower.vision_tower.save_pretrained(updated_vision_tower_path)
        #     image_processor.save_pretrained(updated_vision_tower_path)
        #     print(f"💾 微调后的全新视觉塔权重已成功保存至: {updated_vision_tower_path}")
        print("🎉 Stage-2 训练与验证流程圆满结束！")

if __name__ == "__main__":
    main()