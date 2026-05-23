"""
Multi-Crop 图像裁剪工具。
为 CLOVER 框架的 B' 组件提供全局视图 + 局部视图的图像预处理。
"""

from pathlib import Path
from PIL import Image


def generate_local_crop(
    img_path: Path,
    output_dir: Path,
    crop_ratio: float = 0.6,
    crop_size: int = 256,
) -> Path:
    """
    从原始图像生成中心裁剪的局部放大视图。

    Args:
        img_path: 原始图像路径
        output_dir: 裁剪图像输出目录
        crop_ratio: 中心裁剪区域占原图的比例 (默认 0.6)
        crop_size: 裁剪后 resize 的目标尺寸 (默认 256)

    Returns:
        生成的裁剪图像路径
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # 计算中心裁剪区域
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    cropped = img.crop((left, top, right, bottom))
    cropped = cropped.resize((crop_size, crop_size), Image.LANCZOS)

    # 保存
    stem = img_path.stem
    crop_name = f"{stem}_crop.png"
    crop_path = output_dir / crop_name
    cropped.save(crop_path, "PNG")

    return crop_path


def get_crop_rel_path(img_rel_path: str) -> str:
    """
    根据原始图像的相对路径生成裁剪图像的相对路径。

    Args:
        img_rel_path: 原始图像相对于 PROCESSED_DIR 的路径，如 "images/batch1_张三_1.png"

    Returns:
        裁剪图像相对于 PROCESSED_DIR 的路径，如 "images/batch1_张三_1_crop.png"
    """
    p = Path(img_rel_path)
    stem = p.stem
    return str(p.parent / f"{stem}_crop.png")
