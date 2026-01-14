import os, json, random, math
from typing import List, Dict, Any, Tuple

import torch
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info


class CaptionCollator:
    """
    - 統一 resize (可關)
    - 用 qwen-vl-utils 建 vision inputs
    - labels 只算 assistant (caption) 區段
    """
    def __init__(self, processor: Qwen2VLProcessor, resize_hw: Tuple[int,int]=(560,560), do_resize: bool=True, add_system: bool=True, SYSTEM_MESSAGE: str= "You are a helpful assistant."):
        self.processor = processor
        self.resize_h, self.resize_w = resize_hw
        self.do_resize = do_resize
        self.add_system = add_system
        self.SYSTEM_MESSAGE = SYSTEM_MESSAGE
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_messages = []

        for ex in examples:
            img_paths: List[str] = ex.get("images", [])
            if not img_paths:
                raise ValueError("Missing images")
            img_path = img_paths[0]

            user_content = []
            img_item = {"type": "image", "image": f"{img_path}"}
            if self.do_resize:
                img_item["resized_height"] = self.resize_h
                img_item["resized_width"]  = self.resize_w
            user_content.append(img_item)

            # caption 的 prompt（不需要 <image> token）
            query = ex.get("query", "Please describe the image.").replace("<image>", "").strip()
            answer = ex["response"].strip()
            user_content.append({"type":"text", "text": query})

            if self.add_system:
                messages = [
                    {"role":"system","content":[{"type":"text","text": self.SYSTEM_MESSAGE}]},
                    {"role":"user","content": user_content},
                    {"role":"assistant","content":[{"type":"text","text": answer}]},
                ]
            else:
                messages = [
                    {"role":"user","content": user_content},
                    {"role":"assistant","content":[{"type":"text","text": answer}]},
                ]

            batch_messages.append(messages)

        # full texts (含 assistant)
        full_texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch_messages
        ]

        # prompt texts（只到 user）
        prompt_texts = []
        for m in batch_messages:
            prompt_msgs = m[:2] if self.add_system else m[:1]
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            )

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