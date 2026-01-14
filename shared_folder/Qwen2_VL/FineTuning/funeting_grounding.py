import os
import json
import random
from tqdm import tqdm
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sacrebleu

from qwen_vl_utils import process_vision_info

from event.evaluate import eval_metrics
from event.generate_eval_grounding import generate_k
from event.processor_inspect import list_linear_suffixes, test_processor, test_processor_decode
from event.grounding_collator import GroundingCollator
from event.grounding_parse_box import parse_box, box_resize_to_original, draw_box


cwd = os.getcwd()
print(f"目前工作目錄是: {cwd}")

# ---------
# JSONL 格式假設：每行類似
# {"query":"<image>....", "response":"....", "images":["/abs/path/a.jpg", ...]}
# ---------

# 自訂提示詞
SYSTEM_MESSAGE = (
    "You are a helpful vision-language assistant. "
    "Answer the user based on the image(s) and the question."
)

# 官方預設 提示詞
SYSTEM_MESSAGE = "You are a helpful assistant."


def format_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    # 1. 讀圖 (JSONL 裡 images 是路徑list)
    img_paths: List[str] = ex.get("images", [])
    pil_images = [load_pil(p) for p in img_paths]

    # 2. 把 query 內的 <image> token 拿掉(因為會用 content 里的 image 物件表示)
    query = ex["query"].replace("<image>", "").strip()
    answer = ex["response"].strip()

    # 3. 組 messages (system / user(多圖+文字) / assistant)
    user_content = []
    for im in pil_images:
        user_content.append({"type": "image", "image": im})
    user_content.append({"type": "text", "text": query})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]

    # TRL Cookbook 的格式：同時放 "images" 與 "messages"（images 也放 PIL 物件）
    return {"images": pil_images, "messages": messages}


def main():

    # ---- paths ----
    CACHE_DIR = "../models"
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    data_path = "../dataset/refcoco_grounding/train.jsonl"
    out_dir = "./models/qwen2vl-2b-lora-grounding"
    
    # ---- training knobs ----
    use_qlora = True  # 預設開 (省顯存)
    resize_h  = 560
    resize_w  = 560
    seed      = 42

    random.seed(seed)
    torch.manual_seed(seed)


    # 1. 讀你的 JSONL
    raw = load_dataset("json", data_files=data_path, split="train")
    # 可選：只保留有 box token 的樣本（避免混到 caption 資料）
    raw = raw.filter(lambda x: "<|box_start|>" in x["response"] and "<|box_end|>" in x["response"], num_proc=4)
    
    # 2. 簡單切一點當 eval
    raw = raw.train_test_split(test_size=0.05, seed=42) # 資料集會打亂
    train_raw, eval_raw = raw["train"], raw["test"]
    print("train rows:", len(train_raw), "| val rows:", len(eval_raw))
    #print(next(iter(train_raw)))
    
    # 3. 載入模型 & processor
    if use_qlora:
        # Cookbook 的 4-bit bitsandbytes 設定 :contentReference[oaicite:3]{index=3}
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # 把 base model 權重以 4-bit 形式載入(QLoRA 的核心)
            bnb_4bit_use_double_quant=True, # Double Quant：再量化量化器的統計值，通常更省顯存/更穩
            bnb_4bit_quant_type="nf4", # 量化型態：NF4 常見於 QLoRA (泛用、效果好)
            bnb_4bit_compute_dtype=torch.bfloat16, # 計算用 dtype (建議 bf16；不支援 bf16 時用 fp16)
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            cache_dir=CACHE_DIR
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR
        )

    '''
    # 確認模型 線性層名稱
    for n, m in model.named_modules():
        print(n)
    print("-"*50)
    exit()
    '''
    
    show_model_name = list_linear_suffixes(model)
    print(f"模型框架: {show_model_name}\n\n")
    
    model.config.use_cache = False # 關閉快取以進行訓練
    # model.gradient_checkpointing_enable() # 顯存不夠再開，訓練會變慢一些
    
    processor = Qwen2VLProcessor.from_pretrained(
        model_id, 
        cache_dir=CACHE_DIR
        # min_pixels=256*256,
        # max_pixels=512*512,
        )
                
    test_processor_decode(processor)

    
    # 4. LoRA 設定（Cookbook 範例 target_modules=["q_proj","v_proj"]）
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # 5. 訓練參數（照 Cookbook 的 SFTConfig 思路改小一點，避免爆顯存） :contentReference[oaicite:5]{index=5}
    training_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=10,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        max_length=2048,
        remove_unused_columns=False, # 把「模型用不到的欄位」從 dataset 中移除。
        group_by_length=False, # 如果設為 True，會按長度分組，反而會降低隨機性
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


    collator = GroundingCollator(processor, resize_hw=(resize_h, resize_w), add_system=True, SYSTEM_MESSAGE=SYSTEM_MESSAGE)

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_raw,
        eval_dataset=eval_raw,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(out_dir)
    print(f"saved to: {out_dir}")
    
    
    # 測試模型
    trainer.model.config.use_cache = True  # 開啟快取以加速生成
    trainer.model.eval()
    eval_metrics(trainer, out_dir)
    
    generate_k(
        eval_dataset=eval_raw,
        out_dir=out_dir,
        model=trainer.model,
        processor=processor,
        resize_w=resize_w,
        resize_h=resize_h,
        max_new_tokens=128,
        k=8,
        SYSTEM_MESSAGE=SYSTEM_MESSAGE,
        )
    
    
    
if __name__ == "__main__":
    main()