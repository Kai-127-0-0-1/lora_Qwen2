import json
import re
import os
import sys
import math
import random
from collections import Counter, defaultdict
from statistics import mean, median


BOX_RE = re.compile(r"\((\d+),(\d+)\),\((\d+),(\d+)\)")

def parse_line(line):
    ex = json.loads(line)
    img = ex["images"][0]
    q = ex["query"]
    r = ex["response"]

    # 取 object_ref 內文字（更準確的 caption 長度）
    # Find <|object_ref_start|> ... <|object_ref_end|>
    cap = q
    cap = cap.split("<|object_ref_start|>")[-1]
    cap = cap.split("<|object_ref_end|>")[0].strip()

    m = BOX_RE.search(r)
    box = None
    if m:
        x1,y1,x2,y2 = map(int, m.groups())
        box = (x1,y1,x2,y2)

    return img, cap, box, q, r

def box_area(b):
    x1,y1,x2,y2 = b
    return max(0, x2-x1) * max(0, y2-y1)

def main(jsonl_path, sample_imgs=2000, seed=42):
    random.seed(seed)

    n_rows = 0
    images = []
    caps = []
    cap_word_lens = []
    cap_char_lens = []
    short_1_2 = 0
    short_lt3 = 0

    bad_box = 0
    zero_area = 0
    box_areas = []

    # 用於「每張圖幾筆描述」
    img2cnt = Counter()

    # 重複 query/response 檢查
    qr_hash = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            n_rows += 1
            img, cap, box, q, r = parse_line(line)

            images.append(img)
            caps.append(cap)
            img2cnt[img] += 1

            words = cap.split()
            wlen = len(words)
            cap_word_lens.append(wlen)
            cap_char_lens.append(len(cap))

            if wlen <= 2: short_1_2 += 1
            if wlen < 3:  short_lt3 += 1

            if box is None:
                bad_box += 1
            else:
                a = box_area(box)
                box_areas.append(a)
                if a == 0: zero_area += 1

            qr_hash[(img, q, r)] += 1

    uniq_imgs = len(set(images))

    # 每張圖樣本數分佈
    per_img = list(img2cnt.values())
    per_img_sorted = sorted(per_img)

    # 重複樣本比例（完全相同 img+query+response）
    dup_rows = sum(v-1 for v in qr_hash.values() if v > 1)

    def pct(x): 
        return 100.0 * x / max(1, n_rows)

    print("=== BASIC ===")
    print("jsonl:", jsonl_path)
    print("rows:", n_rows)
    print("unique_images:", uniq_imgs)
    print("rows_per_image: mean=", round(mean(per_img),2), 
          "median=", median(per_img), 
          "p90=", per_img_sorted[int(0.9*len(per_img_sorted))-1])

    print("\n=== CAPTION LENGTH ===")
    print("words: mean=", round(mean(cap_word_lens),2),
          "median=", median(cap_word_lens),
          "p90=", sorted(cap_word_lens)[int(0.9*len(cap_word_lens))-1])
    print("chars: mean=", round(mean(cap_char_lens),2))
    print("short (<=2 words):", short_1_2, f"({pct(short_1_2):.2f}%)")
    print("short (<3 words):", short_lt3, f"({pct(short_lt3):.2f}%)")

    print("\n=== BOX QUALITY ===")
    print("bad_box_format:", bad_box, f"({pct(bad_box):.2f}%)")
    print("zero_area_boxes:", zero_area, f"({pct(zero_area):.2f}%)")
    if box_areas:
        areas_sorted = sorted(box_areas)
        print("box_area: mean=", round(mean(box_areas),2),
              "median=", median(box_areas),
              "p90=", areas_sorted[int(0.9*len(areas_sorted))-1])

    print("\n=== DUPLICATES ===")
    print("duplicate_rows (same img+query+response):", dup_rows, f"({pct(dup_rows):.2f}%)")

    # 顯示最常見的短 caption（幫你判斷要不要過濾）
    c_counter = Counter(caps)
    print("\n=== MOST COMMON CAPS (top 20) ===")
    for cap, cnt in c_counter.most_common(20):
        print(f"{cnt:6d}  {cap}")

if __name__ == "__main__":
    json_path = "./refcoco_grounding/train.jsonl"
    main(json_path)