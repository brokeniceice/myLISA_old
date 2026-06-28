#第二版，数据集加载改为直接读取JSONL文件。
import os
import random
import torch
import json
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from model.llava.model.resnet_expert import ResNetExpert  
import swanlab  
from tqdm import tqdm  
from PIL import Image
import io
import cv2
import math

# ----------------------------
# ⚙️ 1. 配置参数 & 初始化
# ----------------------------
DATA_DIR = "datasets"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 5e-4
IMG_SIZE = 224
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_WORKERS = 16

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

swanlab.init(
    project="myLISA",
    name="Stage1-NPR-7x-Robustness",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "lr": LR,
        "img_size": IMG_SIZE,
        "seed": SEED,
        "expansion": "7x"
    }
)

# ----------------------------
# 🛡️ 2. 底层扰动函数 (Numpy级别，严格对齐测试脚本)
# ----------------------------
def apply_jpeg_compression(image_np, quality):
    """JPEG 压缩"""
    img_pil = Image.fromarray(image_np)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def apply_resizing(image_np, scale):
    """缩放 (先缩小，再用三次插值放大回原尺寸，保留失真)"""
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

# ----------------------------
# 🖼️ 3. 基础空间与色彩变换 (PIL/Tensor级别)
# ----------------------------
base_train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

base_val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

# ----------------------------
# 🔄 4. 统一的鲁棒性变换管道 (原图np -> 扰动np -> 转PIL -> base_transform)
# ----------------------------
class RobustnessTransform:
    def __init__(self, base_transform, attack_type=None, attack_params=None):
        self.base_transform = base_transform
        self.attack_type = attack_type
        self.attack_params = attack_params if attack_params else {}
    
    def __call__(self, img_np):
        # 1. 注入扰动 (在Numpy数组上进行，完全对齐测试期行为)
        if self.attack_type == "jpeg":
            quality = self.attack_params.get('quality', 70)
            img_np = apply_jpeg_compression(img_np, quality)
        elif self.attack_type == "resize":
            scale = self.attack_params.get('scale', 0.5)
            img_np = apply_resizing(img_np, scale)
        elif self.attack_type == "noise":
            variance = self.attack_params.get('variance', 5)
            img_np = apply_gaussian_noise(img_np, variance)
        
        # 2. 转为 PIL Image 交给 Torchvision 处理
        img_pil = Image.fromarray(img_np)
        return self.base_transform(img_pil)

# 创建7种不同的训练变换
train_transforms_7x = [
    lambda img: RobustnessTransform(base_train_transform, None)(img),
    lambda img: RobustnessTransform(base_train_transform, "jpeg", {'quality': 70})(img),
    lambda img: RobustnessTransform(base_train_transform, "jpeg", {'quality': 80})(img),
    lambda img: RobustnessTransform(base_train_transform, "resize", {'scale': 0.5})(img),
    lambda img: RobustnessTransform(base_train_transform, "resize", {'scale': 0.75})(img),
    lambda img: RobustnessTransform(base_train_transform, "noise", {'variance': 5})(img),
    lambda img: RobustnessTransform(base_train_transform, "noise", {'variance': 10})(img),
]

val_transforms_7x = [
    lambda img: RobustnessTransform(base_val_transform, None)(img),
    lambda img: RobustnessTransform(base_val_transform, "jpeg", {'quality': 70})(img),
    lambda img: RobustnessTransform(base_val_transform, "jpeg", {'quality': 80})(img),
    lambda img: RobustnessTransform(base_val_transform, "resize", {'scale': 0.5})(img),
    lambda img: RobustnessTransform(base_val_transform, "resize", {'scale': 0.75})(img),
    lambda img: RobustnessTransform(base_val_transform, "noise", {'variance': 5})(img),
    lambda img: RobustnessTransform(base_val_transform, "noise", {'variance': 10})(img),
]

# 测试集盲测变换字典
test_transforms_dict = {
    "Clean 原图": lambda img: RobustnessTransform(base_val_transform, None)(img),
    "JPEG 70": lambda img: RobustnessTransform(base_val_transform, "jpeg", {'quality': 70})(img),
    "JPEG 80": lambda img: RobustnessTransform(base_val_transform, "jpeg", {'quality': 80})(img),
    "Resize 0.5": lambda img: RobustnessTransform(base_val_transform, "resize", {'scale': 0.5})(img),
    "Resize 0.75": lambda img: RobustnessTransform(base_val_transform, "resize", {'scale': 0.75})(img),
    "Noise Var=5": lambda img: RobustnessTransform(base_val_transform, "noise", {'variance': 5})(img),
    "Noise Var=10": lambda img: RobustnessTransform(base_val_transform, "noise", {'variance': 10})(img),
}

# ----------------------------
# 📦 5. 数据集加载与切分 (严格按照 JSONL 加载)
# ----------------------------
class BaseJSONLDataset(Dataset):
    """基础数据底座：只负责读取 JSONL 并返回原始的 RGB Numpy 数组"""
    def __init__(self, jsonl_path, image_root):
        self.samples = []
        self.image_root = image_root
        
        mode_name = os.path.basename(jsonl_path).split('.')[0].upper()
        print(f"📂 正在加载 {mode_name} 集: {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                img_rel = item["images"][0].lstrip("./")
                mask_rel = item.get("mask", "")
                
                # 有 mask 认为是假图(1)，没有则是真图(0)
                if mask_rel and len(mask_rel.strip()) > 0:
                    self.samples.append((img_rel, 1)) 
                else:
                    self.samples.append((img_rel, 0))    
        print(f"✅ {mode_name} 集成功加载 {len(self.samples)} 条数据！")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel, label = self.samples[idx]
        img_path = os.path.join(self.image_root, img_rel)
        
        # 严格使用 cv2 读取并转换为 RGB Numpy 数组
        image_np = cv2.imread(img_path)
        if image_np is None:
            # 容错机制：遇到损坏图片返回全黑
            image_np = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
        return image_np, label

class Expanded7xDataset(Dataset):
    def __init__(self, subset, transforms_list):
        self.subset = subset
        self.transforms_list = transforms_list

    def __getitem__(self, index):
        real_idx = index // 7
        mode_idx = index % 7
        x, y = self.subset[real_idx] # x 这里拿到的是 BaseJSONLDataset 吐出的 Numpy 数组
        x_tensor = self.transforms_list[mode_idx](x)
        return x_tensor, y

    def __len__(self):
        return len(self.subset) * 7

class MapDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform if transform else (lambda img: RobustnessTransform(base_val_transform, None)(img))

    def __getitem__(self, index):
        x, y = self.subset[index]
        x_tensor = self.transform(x)
        return x_tensor, y

    def __len__(self):
        return len(self.subset)

# --- 替换为你的真实路径 ---
train_jsonl = "/home/yz/myLISA_old/datasets/AIGI-Holmes-Dataset/dataset/train.jsonl"
val_jsonl = "/home/yz/myLISA_old/datasets/AIGI-Holmes-Dataset/dataset/val.jsonl"
test_jsonl = "/home/yz/myLISA_old/datasets/AIGI-Holmes-Dataset/dataset/test.jsonl"
image_root = "/home/yz/myLISA_old/datasets/AIGI-Holmes-Dataset"

# 1. 实例化基础数据底座
train_subset = BaseJSONLDataset(train_jsonl, image_root)
val_subset = BaseJSONLDataset(val_jsonl, image_root)
test_subset = BaseJSONLDataset(test_jsonl, image_root)

train_size = len(train_subset)
val_size = len(val_subset)
test_size = len(test_subset)

# 2. 挂载相应的 Transform 外壳
train_dataset = Expanded7xDataset(train_subset, transforms_list=train_transforms_7x)
val_dataset = Expanded7xDataset(val_subset, transforms_list=val_transforms_7x)

# 3. 生成 DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

# ----------------------------
# 🧠 6. 模型与优化器初始化
# ----------------------------
model = ResNetExpert(use_low_level="npr", pretrained=True).to(DEVICE) 

class TemporaryClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.head = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = self.avgpool(x)       
        x = x.view(x.size(0), -1) 
        return self.head(x)       

classifier = TemporaryClassifier(input_dim=512, num_classes=2).to(DEVICE)

optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

total_steps = len(train_loader) * NUM_EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

# ----------------------------
# 🏃 7. 训练与验证逻辑
# ----------------------------
global_step = 0 

def train_one_epoch(model, classifier, loader, optimizer, criterion, scheduler, epoch):
    global global_step
    model.train()
    classifier.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}/{NUM_EPOCHS}", leave=False)
    for step, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        features = model(imgs)               
        logits = classifier(features)        
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()  
        global_step += 1
        
        batch_loss = loss.item()
        preds = logits.argmax(dim=1)
        batch_acc = (preds == labels).sum().item() / imgs.size(0)
        
        swanlab.log({
            "Train/Loss": batch_loss,
            "Train/Accuracy": batch_acc,
            "Train/Learning_Rate": scheduler.get_last_lr()[0]
        }, step=global_step)
        
        running_loss += batch_loss * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, classifier, loader, criterion, epoch):
    model.eval()
    classifier.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc=f"  Val Epoch {epoch}/{NUM_EPOCHS}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        features = model(imgs)
        logits = classifier(features)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        
    return running_loss / total, correct / total

# ----------------------------
# 🎬 8. 主流程执行
# ----------------------------
print("=" * 60)
print(f"   - 基础训练集 (Train)   : {train_size} 张 -> (扩容后: {len(train_dataset)} 张)")
print(f"   - 验证集 (Val)         : {val_size} 张 -> (扩容后: {len(val_dataset)} 张)")
print(f"   - 测试盲测集 (Test)    : {test_size} 张")
print(f"🚀 开始 Stage-1 NPR 预训练，共计 {total_steps} 步...")
print("=" * 60)

best_val_acc = 0.0
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "npr_stage1_augmented.pth")

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, classifier, train_loader, optimizer, criterion, scheduler, epoch)
    val_loss, val_acc = validate(model, classifier, val_loader, criterion, epoch)

    swanlab.log({
        "Val/Loss": val_loss,
        "Val/Accuracy": val_acc,
        "Epoch": epoch
    }, step=global_step)
    
    print(f"[Epoch {epoch:02d}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        print(f"⭐ 发现巅峰模型！验证集准确率提升至 {val_acc:.4f}！已保存。")
        best_val_acc = val_acc
        torch.save({
            'resnet': model.state_dict(),
            'classifier': classifier.state_dict(),
        }, SAVE_PATH)

print("=" * 60)
print(f"🎉 训练阶段结束！最佳验证准确率: {best_val_acc:.4f}")

# ----------------------------
# 🛡️ 9. 终极鲁棒性盲测阶段
# ----------------------------
print("\n" + "=" * 60)
print("🛡️ 开始在测试集上进行全方位鲁棒性评估 (Robustness Test)...")
print("=" * 60)

checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['resnet'])
classifier.load_state_dict(checkpoint['classifier'])

model.eval()
classifier.eval()

for name, transform_func in test_transforms_dict.items():
    current_test_dataset = MapDataset(test_subset, transform=transform_func)
    current_test_loader = DataLoader(current_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(current_test_loader, desc=f"Testing [{name}]", leave=False):
            imgs = imgs.to(DEVICE)
            features = model(imgs)
            logits = classifier(features)
            
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f" 📊 [{name:<15}] -> ACC: {acc:.4f} | Macro F1: {f1:.4f}")
    
    swanlab.log({
        f"Test_ACC/{name}": acc,
        f"Test_F1/{name}": f1
    })

print("=" * 60)
print("🏆 所有流程圆满收官！")