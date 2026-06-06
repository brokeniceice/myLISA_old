import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50,ResNet50_Weights

class ResNetExpert(nn.Module):
    def __init__(self, num_classes=1024, use_low_level="npr", pretrained=True):
        super().__init__()
        self.use_low_level = use_low_level
        
        # 1. 准备权重配置
        # 如果 pretrained=True，自动下载并加载 ImageNet 权重
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        
        # 2. 加载完整的 ResNet50 (包含权重)
        full_resnet = resnet50(weights=weights)
        
        if pretrained:
            print(f"✅ [ResNetExpert] Loaded ImageNet pretrained weights into backbone.")

        # 只保留到 layer2，丢弃 layer3 和 layer4
        self.backbone = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((16, 16))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)
        

    def interpolate(self, img, factor):
        # 记录原始精度 (如 bf16)
        original_dtype = img.dtype
        # 临时转为 float32 以避免 "not implemented for BFloat16" 错误
        img = img.float()
        
        # 执行插值
        out = F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest',
            recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
        
        # 转回原始精度
        return out.to(dtype=original_dtype)

    def forward(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # === AIGI 核心逻辑: NPR 噪声提取 ===
        if self.use_low_level == 'npr':
            n, c, w, h = x.shape
            # 处理边缘，保证是偶数
            if w % 2 == 1: x = x[:, :, :-1, :]
            if h % 2 == 1: x = x[:, :, :, :-1]
            
            # 核心公式: 原始图 - (下采样再上采样) = 高频残差
            # *2/3 是 AIGI 代码中的经验系数
            NPR = (x - self.interpolate(x, 0.5)) * 2 / 3
            x = NPR
        # ==================================

        # 正常的 ResNet 前向传播
        features = self.backbone(x)
        features = self.spatial_pool(features) # 变成 [B, 512, 16, 16]
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        
        return features

def build_resnet_expert(config,pretrained=True):
    # 默认输出 1024 维，与 CLIP Large 对齐
    hidden_size = getattr(config, 'mm_hidden_size', 1024) 
    return ResNetExpert(num_classes=hidden_size, use_low_level="npr",pretrained=pretrained)