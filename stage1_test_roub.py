import os
import io
import math
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 修复 numpy 在部分环境下的弃用报错
np.int = int
np.float = float
np.bool = bool

# 导入你自己的模型
from model.llava.model.resnet_expert import ResNetExpert  

# ----------------------------
# ⚙️ 1. 配置参数
# ----------------------------
TEST_JSONL = "/home/yz/myLISA_old/datasets/AIGI-Holmes-Dataset/dataset/test.jsonl"
IMAGE_ROOT = "/home/yz/myLISA_old/datasets/AIGI-Holmes-Dataset"
CHECKPOINT_PATH = "./checkpoints/npr_stage1_augmented_10.pth"  # 指向你训练好的最佳权重

BATCH_SIZE = 32
IMG_SIZE = 224
NUM_WORKERS = 8
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ----------------------------
# 🛡️ 2. 底层扰动函数 (Numpy级别)
# ----------------------------
def apply_jpeg_compression(image_np, quality):
    img_pil = Image.fromarray(image_np)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def apply_resizing(image_np, scale):
    h, w = image_np.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    img_down = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_restored = cv2.resize(img_down, (w, h), interpolation=cv2.INTER_CUBIC)
    return img_restored

def apply_gaussian_noise(image_np, variance):
    sigma = math.sqrt(variance)
    noise = np.random.normal(0, sigma, image_np.shape)
    noisy_image = image_np.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# ----------------------------
# 🖼️ 3. 验证集专用的基础变换
# ----------------------------
base_val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

# 统一的鲁棒性变换管道
class RobustnessTransform:
    def __init__(self, base_transform, attack_type=None, attack_params=None):
        self.base_transform = base_transform
        self.attack_type = attack_type
        self.attack_params = attack_params if attack_params else {}
    
    def __call__(self, img_np):
        if self.attack_type == "jpeg":
            quality = self.attack_params.get('quality', 70)
            img_np = apply_jpeg_compression(img_np, quality)
        elif self.attack_type == "resize":
            scale = self.attack_params.get('scale', 0.5)
            img_np = apply_resizing(img_np, scale)
        elif self.attack_type == "noise":
            variance = self.attack_params.get('variance', 5)
            img_np = apply_gaussian_noise(img_np, variance)
        
        img_pil = Image.fromarray(img_np)
        return self.base_transform(img_pil)

# 7 种测试用例映射表
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
# 📦 4. JSONL 解耦数据底座
# ----------------------------
class BaseJSONLDataset(Dataset):
    """只负责读取 JSONL 并返回原始的 RGB Numpy 数组"""
    def __init__(self, jsonl_path, image_root):
        self.samples = []
        self.image_root = image_root
        print(f"📂 正在加载测试集: {jsonl_path}")
        
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
        print(f"✅ 测试集成功加载 {len(self.samples)} 条数据！")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel, label = self.samples[idx]
        img_path = os.path.join(self.image_root, img_rel)
        
        image_np = cv2.imread(img_path)
        if image_np is None:
            image_np = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
        return image_np, label

class MapDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        x_tensor = self.transform(x)
        return x_tensor, y

    def __len__(self):
        return len(self.subset)

# ----------------------------
# 🧠 5. 模型定义与加载
# ----------------------------
class TemporaryClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.head = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)       
        x = x.view(x.size(0), -1) 
        return self.head(x)       

def main():
    print("\n" + "=" * 60)
    print("🚀 初始化网络并加载权重...")
    model = ResNetExpert(use_low_level="npr", pretrained=False).to(DEVICE) 
    classifier = TemporaryClassifier(input_dim=512, num_classes=2).to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 找不到权重文件: {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['resnet'])
    classifier.load_state_dict(checkpoint['classifier'])
    print(f"✅ 成功加载权重: {CHECKPOINT_PATH}")

    model.eval()
    classifier.eval()

    # 准备基础数据池
    test_subset = BaseJSONLDataset(TEST_JSONL, IMAGE_ROOT)

    print("\n" + "=" * 60)
    print("🛡️ 开始全方位鲁棒性盲测 (Robustness Test)...")
    print("=" * 60)

    # ----------------------------
    # 🎯 6. 盲测执行循环
    # ----------------------------
    for name, transform_func in test_transforms_dict.items():
        # 套上对应的降级变换外壳
        current_test_dataset = MapDataset(test_subset, transform=transform_func)
        current_test_loader = DataLoader(
            current_test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(current_test_loader, desc=f"Testing [{name}]", leave=False):
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE).long() # 安全强转
                
                features = model(imgs)
                logits = classifier(features)
                
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f" 📊 [{name:<15}] -> ACC: {acc:.4f} | Macro F1: {f1:.4f}")

    print("=" * 60)
    print("🏆 测试圆满结束！")

if __name__ == "__main__":
    main()