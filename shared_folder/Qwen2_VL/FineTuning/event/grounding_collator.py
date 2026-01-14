import os
import re
import json
import math
import random
import torch
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from functools import lru_cache
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

from event.grounding_parse_box import parse_box, scale_box_xyxy


class GroundingCollator:
    """
    1) 以 resized_height/resized_width 統一 resize（避免 patch 數飄）
    2) 同步縮放 GT bbox（你的 JSONL response 是原圖像素座標時必須做）
    3) 只對 assistant 區段算 loss（mask prompt）
    """
    def __init__(self, processor: Qwen2VLProcessor, resize_hw: Tuple[int, int]=(560,560), add_system: bool=True, SYSTEM_MESSAGE="You are a helpful assistant."):
        self.processor = processor
        self.resize_h, self.resize_w = resize_hw
        self.add_system = add_system
        self.SYSTEM_MESSAGE = SYSTEM_MESSAGE

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_messages = []
        #print(examples[0].keys())
        
        for ex in examples:
            img_paths: List[str] = ex.get("images", [])
            if not img_paths:
                raise ValueError("Missing images")
            img_path = img_paths[0]
            
            '''
            # 讀原圖尺寸，用於縮放 GT bbox
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                src_w, src_h = im.size
            '''
            
            src_w, src_h = self.load_pil(img_path)
            
            
            # 解析 GT bbox (原圖座標)
            gt = parse_box(ex["response"])
            if gt is None:
                raise ValueError(f"Bad response format (no box): {ex['response']}")

            # 縮放到 resize 後座標（與訓練視覺輸入一致）
            gt_s = scale_box_xyxy(gt, src_w, src_h, self.resize_w, self.resize_h)
            response_scaled = f"<|box_start|>({gt_s[0]},{gt_s[1]}),({gt_s[2]},{gt_s[3]})<|box_end|>"

            # user content：image + text
            user_content = [{
                "type": "image",
                "image": f"file://{img_path}",
                "resized_height": self.resize_h,
                "resized_width": self.resize_w,
            }]
            user_content.append({"type":"text", "text": ex["query"].strip()})

            if self.add_system:
                messages = [
                    {"role":"system","content":[{"type":"text","text": self.SYSTEM_MESSAGE}]},
                    {"role":"user","content": user_content},
                    {"role":"assistant","content":[{"type":"text","text": response_scaled}]},
                ]
            else:
                messages = [
                    {"role":"user","content": user_content},
                    {"role":"assistant","content":[{"type":"text","text": response_scaled}]},
                ]

            batch_messages.append(messages)

        # full texts (含 assistant) → 用來算 labels
        full_texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch_messages
        ]

        # prompt texts (只到 user，add_generation_prompt=True) → 用來算 prompt_len 做 mask
        prompt_texts = []
        for m in batch_messages:
            prompt_msgs = m[:2] if self.add_system else m[:1]
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            )

        # vision inputs（讓 qwen-vl-utils 吃 messages）
        image_inputs, video_inputs = process_vision_info(batch_messages)

        model_inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            do_resize=False,  # 避免二次 resize
        )

        prompt_inputs = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )

        labels = model_inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        for i in range(labels.size(0)):
            prompt_len = prompt_inputs["input_ids"][i].ne(pad_id).sum().item()
            labels[i, :prompt_len] = -100

        model_inputs["labels"] = labels
        return model_inputs
        
    @lru_cache(maxsize=2048)  # 依 RAM 調整：512/1024/2048
    def load_pil(self, path: str):
        with Image.open(path) as im:
            src_w, src_h = im.size
            return src_w, src_h