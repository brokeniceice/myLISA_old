import sys
import os
import io
import base64
import hashlib
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import cv2
import numpy as np
import torch
import json
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
import uvicorn

# ==========================================
# 🌟 核心越狱代码：打通上一级目录的模块导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer, CLIPImageProcessor, AutoConfig
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.model.resnet_expert import ResNetExpert
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

# ==========================================
# 百度翻译配置：把你的 AppID 和密钥填到这里
# ==========================================
BAIDU_TRANSLATE_APP_ID = "20260622002635886"
BAIDU_TRANSLATE_SECRET_KEY = "WJjjSC4oAKrdEdp5yUal"
BAIDU_TRANSLATE_URL = "https://fanyi-api.baidu.com/api/trans/vip/translate"
BAIDU_TRANSLATE_FROM = "en"
BAIDU_TRANSLATE_TO = "zh"
BAIDU_TRANSLATE_TIMEOUT = 8
BAIDU_TRANSLATE_MAX_CHARS = 4500
BAIDU_TRANSLATE_RETRY_DELAYS = (1.0, 2.0, 4.0)
BAIDU_PARAGRAPH_BREAK_TOKEN = "x9x9lisabrk7q7q"
BAIDU_TRANSLATE_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ==========================================
# 1. 静态参数配置
# ==========================================
class Args:
    version = "../checkpoints_stage2/my_best_model/merged_final"
    npr_ckpt = "../checkpoints/npr_stage1_augmented_best.pth"
    precision = "bf16"
    image_size = 1024
    model_max_length = 2048
    local_rank = 0
    load_in_8bit = False
    load_in_4bit = False
    use_mm_start_end = True
    conv_type = "llava_v1"
    seg_token_idx = -1

args = Args()

# ==========================================
# 2. 网络结构定义 (从 chat.py 迁移)
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
        if ckpt_path is not None and os.path.exists(ckpt_path):
            print(f"✅ Loading pretrained NPR expert from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.resnet.load_state_dict(checkpoint['resnet'], strict=True)
            self.classifier.load_state_dict(checkpoint['classifier'], strict=True)
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        spatial_features = self.resnet(images)
        logits = self.classifier(spatial_features)
        expert_preds = logits.argmax(dim=1)
        return spatial_features, expert_preds

def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024):
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def baidu_translate_en_to_zh(text):
    text = text.strip()
    if not text or not BAIDU_TRANSLATE_APP_ID or not BAIDU_TRANSLATE_SECRET_KEY:
        return None

    for attempt in range(len(BAIDU_TRANSLATE_RETRY_DELAYS) + 1):
        salt = str(random.randint(32768, 65536))
        sign_source = BAIDU_TRANSLATE_APP_ID + text + salt + BAIDU_TRANSLATE_SECRET_KEY
        sign = hashlib.md5(sign_source.encode("utf-8")).hexdigest()
        payload = urllib.parse.urlencode({
            "q": text,
            "from": BAIDU_TRANSLATE_FROM,
            "to": BAIDU_TRANSLATE_TO,
            "appid": BAIDU_TRANSLATE_APP_ID,
            "salt": salt,
            "sign": sign,
        }).encode("utf-8")

        request = urllib.request.Request(
            BAIDU_TRANSLATE_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with BAIDU_TRANSLATE_OPENER.open(request, timeout=BAIDU_TRANSLATE_TIMEOUT) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"[Baidu Translate] request failed: {type(exc).__name__}: {exc}")
            return None

        error_code = str(result.get("error_code", ""))
        if error_code == "54003" and attempt < len(BAIDU_TRANSLATE_RETRY_DELAYS):
            delay = BAIDU_TRANSLATE_RETRY_DELAYS[attempt]
            print(f"[Baidu Translate] rate limited, retrying in {delay:.0f}s")
            time.sleep(delay)
            continue

        if "error_code" in result:
            print(f"[Baidu Translate] API error {result.get('error_code')}: {result.get('error_msg', '')}")
            return None

        break

    translated_items = result.get("trans_result")
    if not translated_items:
        print("[Baidu Translate] missing trans_result in response")
        return None

    translated_text = "\n".join(item.get("dst", "") for item in translated_items).strip()
    return translated_text or None

def protect_translation_paragraph_breaks(text):
    protected_lines = []
    blank_pending = False

    for line in text.strip().splitlines():
        if line.strip():
            if blank_pending and protected_lines:
                protected_lines.append(BAIDU_PARAGRAPH_BREAK_TOKEN)
            protected_lines.append(line)
            blank_pending = False
        else:
            blank_pending = True

    return "\n".join(protected_lines)

def restore_translation_paragraph_breaks(text):
    restored_lines = []
    paragraph_token = BAIDU_PARAGRAPH_BREAK_TOKEN.lower()

    for line in text.splitlines():
        if paragraph_token in line.lower():
            if restored_lines and restored_lines[-1] != "":
                restored_lines.append("")
            continue
        restored_lines.append(line)

    return "\n".join(restored_lines).strip()

def split_text_for_translation(text, max_chars=BAIDU_TRANSLATE_MAX_CHARS):
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > max_chars and current.strip():
            chunks.append(current)
            current = ""

        while len(line) > max_chars:
            chunks.append(line[:max_chars])
            line = line[max_chars:]

        current += line

    if current.strip():
        chunks.append(current)

    return chunks

def translate_complete_analysis(text):
    source = text.strip()
    if not source:
        return None

    protected_source = protect_translation_paragraph_breaks(source)
    translated_chunks = []
    for chunk in split_text_for_translation(protected_source):
        translated = baidu_translate_en_to_zh(chunk)
        if not translated:
            return None
        translated_chunks.append(translated)

    translated_text = "\n".join(translated_chunks)
    return restore_translation_paragraph_breaks(translated_text)

# ==========================================
# 3. 全局模型挂载 (只在启动时执行一次)
# ==========================================
print(">>> 正在挂载模型到GPU...")
device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.version, cache_dir=None, model_max_length=args.model_max_length, padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token
if "[SEG]" in tokenizer.get_vocab():
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
kwargs = {"torch_dtype": torch_dtype}

cfg = AutoConfig.from_pretrained(args.version)
cfg.npr_pretrained_path = None 

model = LISAForCausalLM.from_pretrained(args.version, config=cfg, low_cpu_mem_usage=True, seg_token_idx=args.seg_token_idx, **kwargs)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype)

model = model.bfloat16().cuda().to(device) if args.precision == "bf16" else model.float().cuda().to(device)
vision_tower.to(device=args.local_rank)

clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
transform = ResizeLongestSide(args.image_size)
model.eval()

expert_model = UnifiedNPRExpert(ckpt_path=args.npr_ckpt).cuda().to(device)
expert_model.eval()
npr_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])
print(">>> 引擎挂载完毕！服务已就绪。")


# ==========================================
# 4. FastAPI 服务搭建
# ==========================================
app = FastAPI()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/")
async def get_index():
    with open(os.path.join(current_dir, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # --- 1. 读取图像与预处理 (保持不变) ---
    contents = await file.read()
    image_np_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image_np_bgr is None:
        raise HTTPException(status_code=400, detail="无法读取上传的图像文件")

    image_np = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    pil_img = Image.fromarray(image_np)

    image_npr = npr_transform(pil_img).unsqueeze(0).cuda().to(device)
    with torch.no_grad():
        spatial_features = expert_model.resnet(image_npr)
        logits = expert_model.classifier(spatial_features)
        probabilities = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probabilities, dim=1).item())
        real_probability = float(probabilities[0, 0].detach().cpu().item())
        aigc_probability = float(probabilities[0, 1].detach().cpu().item())
        confidence = aigc_probability if pred == 1 else real_probability

    is_aigc = pred == 1
    verdict_label = "AIGC图像" if is_aigc else "真实图像"
    safe_filename = os.path.basename(file.filename or "uploaded_image")
    detected_at = datetime.now().astimezone().isoformat()
    verdict_payload = {
        "type": "verdict",
        "label": verdict_label,
        "is_aigc": is_aigc,
        "confidence": confidence,
        "probabilities": {
            "real": real_probability,
            "aigc": aigc_probability,
        },
        "filename": safe_filename,
        "detected_at": detected_at,
    }
        
    hint = "[System Forensic Expert Hint: Low-level pixel analysis indicates the presence of AIGC forgery traces in this image.]" if pred == 1 else "[System Forensic Expert Hint: Low-level pixel analysis indicates this is a natural image with no forged traces detected.]"
    image_npr = image_npr.bfloat16() if args.precision == "bf16" else image_npr.float()

    prompt = "Please determine whether this image is fake or real, and provide the reasons for your judgment. If it is a fake image, please also segment the forged area.\n" + hint 
    if DEFAULT_IMAGE_TOKEN not in prompt:
         prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)

    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda().to(device)
    image_clip = image_clip.bfloat16() if args.precision == "bf16" else image_clip.float()
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().to(device)
    image = image.bfloat16() if args.precision == "bf16" else image.float()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda().to(device)
    model.get_model().current_images_npr = image_npr

    # --- 2. 准备流式生成器 (Streamer) ---
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    result_container = {} # 用于在线程间传递最终的 Mask

    # 定义后台推理任务
    def generation_task():
        try:
            # 💡 注意：这里传入了 streamer 参数！
            output_ids, pred_masks = model.evaluate_stream(
                image_clip, image, input_ids, resize_list, original_size_list, 
                max_new_tokens=2048, tokenizer=tokenizer, streamer=streamer
            )
            result_container['masks'] = pred_masks
        except Exception as e:
            result_container['error'] = str(e)

    # 启动后台线程执行推理
    thread = Thread(target=generation_task)
    thread.start()

    def event_generator():
        full_analysis_text = []

        yield f"data: {json.dumps(verdict_payload)}\n\n"
        yield f"data: {json.dumps({'type': 'hint', 'content': hint})}\n\n"

        for new_text in streamer:
            if new_text:
                # 过滤掉前缀和结尾符
                clean_text = new_text.replace("ASSISTANT:", "").replace("ASSISTANT", "").replace("</s>", "")
                if clean_text:
                    full_analysis_text.append(clean_text)
                    yield f"data: {json.dumps({'type': 'text', 'content': clean_text})}\n\n"

        thread.join()

        # D. 如果有 Mask，处理并发送 Base64
        if 'error' in result_container:
            yield f"data: {json.dumps({'type': 'error', 'content': result_container['error']})}\n\n"
        elif is_aigc and 'masks' in result_container:
            pred_masks = result_container['masks']
            if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
                pred_mask = pred_masks[0].detach().cpu().numpy()[0] > 0
                save_img = image_np.copy()
                save_img[pred_mask] = (image_np * 0.5 + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask]
                save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', save_img_bgr)
                overlay_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                
                # 最后一步：推送 Mask 给前端
                yield f"data: {json.dumps({'type': 'mask', 'content': overlay_base64})}\n\n"

        yield f"data: {json.dumps({'type': 'analysis_done'})}\n\n"

        translated_analysis = translate_complete_analysis("".join(full_analysis_text))
        if translated_analysis:
            yield f"data: {json.dumps({'type': 'translation_text', 'content': translated_analysis})}\n\n"

    # 使用 StreamingResponse 返回 SSE 流
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
