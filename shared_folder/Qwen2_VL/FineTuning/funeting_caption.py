import os, json
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from functools import lru_cache

import torch
import torch.nn as nn
import sacrebleu
from PIL import Image
from dataclasses import dataclass
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info

from event.evaluate import eval_metrics
from event.generate_eval_caption import generate_k
from event.processor_inspect import list_linear_suffixes, test_processor, test_processor_decode
from event.caption_collator import CaptionCollator
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



@lru_cache(maxsize=2048)  # 依 RAM 調整：512/1024/2048
def load_pil(path: str)-> Image.Image:
    img = Image.open(path).convert("RGB")
    return img
    
    
    
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
    CACHE_DIR = "../models"
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    data_path = "../dataset/lmms_lab_coco_caption/train.jsonl"
    out_dir = "./models/qwen2vl-2b-lora-caption"
    seed = 42
    use_qlora = True  # 預設開 (省顯存)
    use_qlora = True
    do_resize = True
    resize_h  = 560
    resize_w  = 560

    random.seed(seed)
    torch.manual_seed(seed)


    # 1. 讀你的 JSONL
    raw = load_dataset("json", data_files=data_path, split="train")
    # 簡單切一點當 eval
    raw = raw.train_test_split(test_size=0.05, seed=42) # 資料集會打亂
    train_raw, eval_raw = raw["train"], raw["test"]


    # 2. 載入模型 & processor
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
    # model.gradient_checkpointing_enable()
    
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
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        max_length=2048,
        remove_unused_columns=False,
        group_by_length=False, # 如果設為 True，會按長度分組，反而會降低隨機性
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


    collator = CaptionCollator(processor, resize_hw=(resize_h, resize_w), do_resize=do_resize, add_system=True)

    # 6. Trainer（Cookbook：SFTTrainer + processing_class=processor；會自動用 VLM 的 collator） :contentReference[oaicite:6]{index=6}
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
        resize_h=resize_h, 
        resize_w=resize_w,
        max_new_tokens=128,
        k=8,
        SYSTEM_MESSAGE=SYSTEM_MESSAGE,
        do_resize=do_resize,
    )

    
if __name__ == "__main__":
    main()