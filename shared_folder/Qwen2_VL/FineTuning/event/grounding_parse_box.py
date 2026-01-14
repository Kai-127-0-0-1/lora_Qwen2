import os
import re
import json
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

BOX_RE = re.compile(
    r"<\|box_start\|\>\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)<\|box_end\|\>"
)


def parse_box(text: str) -> Optional[List[int]]:
    m = BOX_RE.search(text)
    if not m:
        return None
    return list(map(int, m.groups()))  # [x1,y1,x2,y2]
    
    
def has_valid_xyxy(box: Optional[List[int]]) -> bool:
    """僅判斷是否有 4 個座標（不做 clamp），可用於 debug/log。"""
    return isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, int) for v in box)
    
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def scale_box_xyxy(box: List[int], src_w: int, src_h: int, dst_w: int, dst_h: int) -> List[int]:
    """Scale xyxy from src size -> dst size, clamp, ensure valid, avoid zero-area."""
    x1, y1, x2, y2 = box
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    x1 = int(round(x1 * sx))
    x2 = int(round(x2 * sx))
    y1 = int(round(y1 * sy))
    y2 = int(round(y2 * sy))

    x1 = int(round(clamp(x1, 0, dst_w - 1)))
    x2 = int(round(clamp(x2, 0, dst_w - 1)))
    y1 = int(round(clamp(y1, 0, dst_h - 1)))
    y2 = int(round(clamp(y2, 0, dst_h - 1)))

    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    if x2 == x1: x2 = min(dst_w - 1, x1 + 1)
    if y2 == y1: y2 = min(dst_h - 1, y1 + 1)
    return [x1, y1, x2, y2]
    
    
def box_resize_to_original(box_resized: List[int], orig_w: int, orig_h: int, resize_w: int, resize_h: int,) -> List[int]:
    """Map box from resized image coords -> original image coords."""
    x1,y1,x2,y2 = box_resized
    sx = orig_w / float(resize_w)
    sy = orig_h / float(resize_h)
    ox1 = int(round(x1 * sx))
    ox2 = int(round(x2 * sx))
    oy1 = int(round(y1 * sy))
    oy2 = int(round(y2 * sy))

    ox1 = clamp(ox1, 0, orig_w - 1)
    ox2 = clamp(ox2, 0, orig_w - 1)
    oy1 = clamp(oy1, 0, orig_h - 1)
    oy2 = clamp(oy2, 0, orig_h - 1)

    if ox2 < ox1: ox1, ox2 = ox2, ox1
    if oy2 < oy1: oy1, oy2 = oy2, oy1
    if ox2 == ox1: ox2 = min(orig_w - 1, ox1 + 1)
    if oy2 == oy1: oy2 = min(orig_h - 1, oy1 + 1)
    return [ox1, oy1, ox2, oy2]


def draw_box(img: Image.Image, box: List[int], label: str = "", color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """Draw rectangle + label on image and return a copy."""
    out = img.copy()
    d = ImageDraw.Draw(out)

    x1,y1,x2,y2 = box
    # 線寬：依圖大小自動調整
    w, h = out.size
    thickness = max(2, int(round(min(w, h) * 0.006)))

    # 畫框（紅色）
    for t in range(thickness):
        d.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)

    # 標籤
    if label:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        text = label
        tw, th = d.textbbox((0,0), text, font=font)[2:]
        pad = 2
        tx, ty = x1, max(0, y1 - (th + 2*pad))
        d.rectangle([tx, ty, tx + tw + 2*pad, ty + th + 2*pad], fill=color)
        d.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)

    return out
    
    