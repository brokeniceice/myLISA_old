import os

from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if os.path.isdir(vision_tower) or os.path.isfile(vision_tower):
        print(f"Loading vision tower from local path: {vision_tower}")
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if (
        vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "clip" in vision_tower
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
