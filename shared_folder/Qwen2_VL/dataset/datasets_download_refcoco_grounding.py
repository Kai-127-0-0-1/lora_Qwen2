import os
import json
import argparse
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import random

BOX_FMT = "<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
QRY_FMT = "Find <|object_ref_start|>{cap}<|object_ref_end|>"


def clamp(v: float, lo: float, hi: float) -> float:
    # 限制在 [lo, hi] 範圍內，避免 bbox 超出圖片邊界
    return max(lo, min(hi, v))


def xywh_to_xyxy(x, y, bw, bh):
    return [x, y, x + bw, y + bh]


def to_int_xyxy(bbox_xyxy: List[float], w: int, h: int) -> List[int]:
    # bbox is [x1,y1,x2,y2] (RefCOCO in the HF cookbook uses xyxy)
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(round(clamp(x1, 0, w - 1)))
    y1 = int(round(clamp(y1, 0, h - 1)))
    x2 = int(round(clamp(x2, 0, w - 1)))
    y2 = int(round(clamp(y2, 0, h - 1)))
    # 保證 x2>=x1, y2>=y1
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    
    # 確保至少 1 像素寬高
    if x2 == x1:
        x2 = min(w - 1, x1 + 1)
    if y2 == y1:
        y2 = min(h - 1, y1 + 1)
    
    return [x1, y1, x2, y2]



def parse_flickr_url(raw_image_info: Any) -> Optional[str]:
    """
    raw_image_info 可能是 str(JSON) 或 dict、flickr_url
    """
    try:
        if raw_image_info is None:
            return None
        if isinstance(raw_image_info, str):
            info = json.loads(raw_image_info)
        elif isinstance(raw_image_info, dict):
            info = raw_image_info
        else:
            return None
        return info.get("flickr_url")
    except Exception:
        return None


def download_image(url: str, timeout: int = 20) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img
        
    except requests.exceptions.HTTPError as errh:
        print("HTTP 錯誤：", errh)
        return None
        
    except requests.exceptions.ConnectionError as errc:
        print("連線錯誤：", errc)
        return None
        
    except requests.exceptions.Timeout as errt:
        print("逾時錯誤：", errt)
        return None
        
    except requests.exceptions.RequestException as err:
        print("其他錯誤：", err)
        return None
        
    except Exception:
        return None
        

def safe_name(x: Any) -> str:
    s = str(x)
    return s.replace("/", "_").replace("\\", "_")


# 過濾減少句子太簡短問題
def keep_caption_by_words(cap: str) -> bool:
    n = len(cap.strip().split())
    if n <= 2:
        return random.random() < 0.25
    if n <= 4:
        return random.random() < 0.75
    return True



def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./refcoco_grounding")
    ap.add_argument("--max_images", type=int, default=200, help="最多下載幾張不同圖片(不是樣本數)")
    ap.add_argument("--max_rows", type=int, default=0, help="最多輸出幾行 JSONL(樣本數)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache_dir", default="./refcoco_grounding")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, f"{args.split}.jsonl")

    '''
    # 查看資料集 train、val、test
    builder = load_dataset("jxu124/refcoco", cache_dir=args.cache_dir)
    print(builder)           # 會列出 DatasetDict keys
    print(builder.keys())    # 例如 dict_keys(['val','test'])
    '''
    
    print("Loading dataset:", "jxu124/refcoco", "split:", args.split)
    ds = load_dataset("jxu124/refcoco", split=args.split, cache_dir=args.cache_dir)

    total = min(len(ds), args.max_images) if args.max_images else len(ds)


    # 下載過的圖片避免重複抓
    downloaded = {}  # image_key -> abs_path
    num_rows = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, total=total, desc=f"Building {args.split}.jsonl"):
            if args.max_images and num_rows >= args.max_images:
                break

            # 1. Get URL
            url = parse_flickr_url(ex.get("raw_image_info"))
            if not url: continue
            
            # 2. 決定圖片 key/檔名(同圖多 caption 會重複出現)
            image_key = ex.get("image_id") or ex.get("global_image_id") or ex.get("file_name") or url
            image_key = safe_name(image_key)
            
            # 3. 下載/讀取圖片
            if image_key in downloaded:
                img_path = downloaded[image_key]
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    # 本地壞掉就重新抓
                    img = download_image(url, timeout=args.timeout)
                    if img is None:
                        continue
                    img.save(img_path, quality=95)
            else:
                if args.max_images and len(downloaded) >= args.max_images:
                    # 圖片數上限到了，但樣本數還沒到：就不再新增新圖，跳過
                    continue
                img = download_image(url, timeout=20)
                if img is None:
                    continue
            
                img_path = os.path.abspath(os.path.join(img_dir, f"{image_key}.jpg"))
                img.save(img_path, quality=95)
                downloaded[image_key] = img_path
            
            w, h = img.size

            # 4. bbox（假設資料集 bbox 已是 xyxy；如果你的版本是 xywh，要在此轉換）
            bbox = ex.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = to_int_xyxy(bbox, w, h)

            # 5. captions（RefCOCO 是 referring expressions）
            caps = ex.get("captions") or ex.get("sentences") or []
            if not caps:
                continue

            # 6. 展開成多筆
            for cap in caps:
                if args.max_rows and num_rows >= args.max_rows:
                    break
                    
                cap = str(cap).strip()
                if not cap:
                    continue

                if not keep_caption_by_words(cap):
                    continue

                item = {
                    "query": QRY_FMT.format(cap=cap),
                    "response": BOX_FMT.format(x1=x1, y1=y1, x2=x2, y2=y2),
                    "images": [img_path],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                num_rows += 1

            if num_rows % 100 == 0 and num_rows > 0:
                print(f"written rows: {num_rows} | images: {len(downloaded)}")
            
    print("Done.")
    print("JSONL:", out_jsonl)
    print("rows:", num_rows)
    print("images downloaded:", len(downloaded))

if __name__ == "__main__":
    main()
