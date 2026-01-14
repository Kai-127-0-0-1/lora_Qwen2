import os
import re
from typing import Optional, List, Tuple

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from event.grounding_parse_box import parse_box, has_valid_xyxy, box_resize_to_original, draw_box

SYSTEM_MESSAGE = "You are a helpful assistant."

# Qwen2-VL 圖片 Resize 官方預設
# min_pixels: 預設通常是 224 x 224 總像素 (約 50,176)
# max_pixels: 預設通常是 1280 x 1280 總像素 (約 1,638,400)


@torch.inference_mode()
def predict_box(
    model,
    processor,
    image_path: str,
    query: str,
    resize_hw: Tuple[int,int]=(560,560),
    max_new_tokens: int = 64,
) -> str:
    resize_h, resize_w = resize_hw

    user_content = [{
        "type": "image",
        "image": f"file://{image_path}",
        "resized_height": resize_h,
        "resized_width": resize_w,
    }]
    user_content.append({"type": "text", "text": query})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": user_content},
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_resize=False,  # 避免二次 resize
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    gen = out_ids[0, inputs["input_ids"].shape[1]:]
    out_text = processor.decode(gen, skip_special_tokens=False)
    return out_text


def main():
    # ---- 參數 ----
    CACHE_DIR = "../models"
    BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    LORA_DIR   = "./models/qwen2vl-2b-lora-grounding"
    IMAGE_PATH = "../dataset/refcoco_grounding/images/581857.jpg"
    QUERY      = "the lady with the blue shirt"
    RESIZE_H   = 560
    RESIZE_W   = 560
    


    text = "Find <|object_ref_start|>{cap}<|object_ref_end|>"
    QUERY_text = text.format(cap=QUERY)

    # ---- 載入 processor ----
    processor = Qwen2VLProcessor.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)


    # ---- 載入 base model + LoRA adapter ----
    # 若你訓練是 QLoRA(4bit)，推論也可以 4bit 載入（省顯存）
    use_4bit = False
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            cache_dir=CACHE_DIR
        )
    else:
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR
        )

    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    # pad_token 對齊 (可選，但建議)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    # ---- 推論 ----
    out = predict_box(
        model=model,
        processor=processor,
        image_path=IMAGE_PATH,
        query=QUERY_text,
        resize_hw=(RESIZE_H, RESIZE_W),
        max_new_tokens=64,
    )

    box_resized = parse_box(out)

    print("QUERY:", QUERY_text)
    print("RAW:", out)
    print("BOX:", box_resized)


    if box_resized is None:
        print("No box parsed. Nothing to draw.")
        return


    if has_valid_xyxy(box_resized):
        OUT_RESIZED = "pred_on_resized.jpg"
        OUT_ORIG    = "pred_on_original.jpg"

        # ---- 在 resized 圖上畫框 ----
        img_orig = Image.open(IMAGE_PATH).convert("RGB")
        img_resized = img_orig.resize((RESIZE_W, RESIZE_H), resample=Image.BICUBIC)
        img_resized_drawn = draw_box(img_resized, box_resized, label=f"{QUERY}")
        img_resized_drawn.save(OUT_RESIZED, quality=95)
        print("Saved:", OUT_RESIZED)

        # ---- 反推回原圖座標，在原圖上畫框 ----
        ow, oh = img_orig.size
        box_orig = box_resize_to_original(box_resized, ow, oh, RESIZE_W, RESIZE_H)
        img_orig_drawn = draw_box(img_orig, box_orig, label=f"{QUERY}")
        img_orig_drawn.save(OUT_ORIG, quality=95)
        print("Saved:", OUT_ORIG)
        print("BOX(original):", box_orig)
    else: 
        print("No box parsed. Nothing to draw. |", box_resized)


if __name__ == "__main__":
    main()
