import argparse
import os
import sys
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor, AutoConfig
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.model.resnet_expert import ResNetExpert
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    # 🔥 修改 1: 默认路径指向你的合并模型
    parser.add_argument("--version", default="./checkpoints_stage2/my_best_model/merged_final")
    parser.add_argument("--npr_ckpt", type=str, default="./checkpoints/npr_stage1_augmented_best.pth")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int) # 建议改为 2048 或和训练保持一致
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)

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
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # 检查 [SEG] token 是否存在，不存在则使用默认处理 (防报错)
    if "[SEG]" in tokenizer.get_vocab():
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        print("⚠️ Warning: [SEG] token not found in tokenizer. Using generic fallback.")
        args.seg_token_idx = -1 


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # =============================================================
    #  强制修改 Config
    # =============================================================
    print(f"⚙️ Loading Config from {args.version}...")
    cfg = AutoConfig.from_pretrained(args.version)
    
    
    cfg.npr_pretrained_path = None 
    # print("✅ Forced config.npr_pretrained_path = None (Using Merged Weights)")

    model = LISAForCausalLM.from_pretrained(
        args.version, 
        config=cfg, # 传入修改后的 config
        low_cpu_mem_usage=True, 
        vision_tower=args.vision_tower, 
        seg_token_idx=args.seg_token_idx, 
        **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 初始化 Vision Modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda().to(device)
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda().to(device)
    elif args.precision == "fp32":
        model = model.float().cuda().to(device)

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    print("🧠 Loading NPR Expert Model...")
    expert_model = UnifiedNPRExpert(ckpt_path=args.npr_ckpt).cuda().to(device)
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
    print("🚀 Model loaded successfully! Ready to chat. (Type 'exit' or 'quit' to stop)")
    
    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        # prompt =input("Please input your prompt: ")
        # if prompt.strip().lower() in ['exit', 'quit']:
        #     print("👋 Exiting chat. Goodbye!")
        #     break
        image_path = input("Please input the image path: ")
        if image_path.strip().lower() in ['exit', 'quit']:
            print("👋 Exiting chat. Goodbye!")
            break
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        # npr提示词注入
        pil_img = Image.fromarray(image_np)
        image_npr = npr_transform(pil_img).unsqueeze(0).cuda().to(device)
        with torch.no_grad():
            _, expert_preds = expert_model(image_npr)
            pred = expert_preds.item()
        if pred == 1:
            hint = "[System Forensic Expert Hint: Low-level pixel analysis indicates the presence of AIGC forgery traces in this image.]"
        else:
            hint = "[System Forensic Expert Hint: Low-level pixel analysis indicates this is a natural image with no forged traces detected.]"
        if args.precision == "bf16":
            image_npr = image_npr.bfloat16()
        elif args.precision == "fp16":
            image_npr = image_npr.half()
        else:
            image_npr = image_npr.float()
            
        prompt ="Please determine whether this image is fake or real, and provide the reasons for your judgment. If it is a fake image, please also segment the forged area."
        prompt = prompt.rstrip() + "\n" + hint 
        prompt = prompt.rstrip() 
        # 兼容处理：确保 <image> 标签存在
        if DEFAULT_IMAGE_TOKEN not in prompt:
             prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
             
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()


        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda().to(device)
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda().to(device)
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda().to(device)
        model.get_model().current_images_npr = image_npr

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=2048,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        print("text_output: ", text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 255)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])