import random
import torch
import sacrebleu
import math
import os
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from qwen_vl_utils import process_vision_info
from PIL import Image
from event.grounding_parse_box import parse_box, has_valid_xyxy, scale_box_xyxy, box_resize_to_original, draw_box


def iou(a: Optional[List[int]], b: Optional[List[int]]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union




@torch.inference_mode()
def predict_one(model, processor, resize_w, resize_h, SYSTEM_MESSAGE, ex: Dict[str, Any], max_new_tokens: int = 64) -> str:
    img_path = ex["images"][0]
    user_content = [{
        "type":"image", "image": f"file://{img_path}",
        "resized_height": resize_h, "resized_width": resize_w
    }]
    user_content.append({"type":"text", "text": ex["query"].strip()})

    messages = [
        {"role":"system","content":[{"type":"text","text": SYSTEM_MESSAGE}]},
        {"role":"user","content": user_content},
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    ).to(model.device)


    # 轉 vision tensor dtype（補強）
    model_dtype = next(model.parameters()).dtype
    for k in ("pixel_values", "pixel_values_videos"):
        if k in inputs and inputs[k].dtype == torch.float32:
            inputs[k] = inputs[k].to(model_dtype)


    autocast_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    
        
    gen = out_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=False)
    



## 增加圖片畫框

    
def generate_k(eval_dataset, out_dir, model, processor, resize_w, resize_h, max_new_tokens=128, k=8, SYSTEM_MESSAGE="You are a helpful assistant."):

    out_dir_img = os.path.join(out_dir, "pred_images")
    os.makedirs(out_dir_img, exist_ok=True)
    

    # 抽幾筆測試
    kn = min(k, len(eval_dataset))
    sample_idx = random.sample(range(len(eval_dataset)), k=kn)
    ious = []

    print("\n=== SAMPLE PRED (val) ===")
    rows = []
    for ni, idx in enumerate(sample_idx):
        ex = eval_dataset[idx]
        pred = predict_one(model, processor, resize_w, resize_h, SYSTEM_MESSAGE, ex)
        pred_box = parse_box(pred)

        QUERY = ex["query"]
        gt = parse_box(ex["response"])
        image_path = ex["images"][0]
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            src_w, src_h = im.size
            
        gt_scaled = scale_box_xyxy(gt, src_w, src_h, resize_w, resize_h) if gt else None

        # 物件畫框
        if ni < 100 and has_valid_xyxy(pred_box):
            pred_box_orig = box_resize_to_original(pred_box, src_w, src_h, resize_w, resize_h)
            img_drawn = draw_box(im, pred_box_orig, label=f"pred", color=(255,0,0))
            img_drawn = draw_box(img_drawn, gt, label=f"{QUERY} (orig)", color=(0,0,255))
            file_name = Path(image_path).name
            out_dir_img_name = os.path.join(out_dir_img, file_name)
            img_drawn.save(out_dir_img_name, quality=95)


        if has_valid_xyxy(pred_box):
            score = iou(pred_box, gt_scaled)
            ious.append(score)
            print(f"\n--- idx {idx} ---")
            print("Q:", QUERY)
            print("Ground Truth (scaled):", gt_scaled) # resize 縮放到同一個尺寸
            print("PRED_BOX:", pred_box, "IoU:", round(score, 4))
            print("PRED_RAW:", pred)
            rows.append({"idx": idx, "question": ex["query"], "GT(scaled)": gt_scaled, "pred_box": pred_box, "IoU": round(score, 4), "pred": pred})
        else:
            print(f"\n--- idx {idx} ---")
            print("The answer is Error")
            print("Q:", QUERY)
            print("PRED_RAW:", pred)
            
            
    print("\nMean IoU (sample):", sum(ious)/len(ious) if ious else 0.0)

        
        
        

    
    # 存下來方便比對
    with open(os.path.join(out_dir, "qualitative_samples.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)