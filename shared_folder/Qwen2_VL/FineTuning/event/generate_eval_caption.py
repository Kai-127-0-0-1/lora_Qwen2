import random
import torch
import sacrebleu
import math
import os
import json
from qwen_vl_utils import process_vision_info


def bleu_socre(rows):
    def norm(s: str) -> str:
        return " ".join((s or "").strip().split())

    refs = [norm(r["ref"]) for r in rows]
    preds = [norm(r["pred"]) for r in rows]
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    print("BLEU:", bleu)


def build_user_only_messages(example):
    """把訓練用的 messages 拿掉 assistant 回答，變成推論用 prompt"""
    msgs = example["messages"]
    # 保留 system + user（如果你訓練時沒放 system，就只會有 user/assistant）
    out = []
    for m in msgs:
        if m["role"] in ("system", "user"):
            out.append(m)
    return out

def get_reference_answer(example):
    """從原 messages 取出參考答案（assistant 那段）"""
    for m in example["messages"]:
        if m["role"] == "assistant":
            # content 通常 [{"type":"text","text": "..."}]
            for c in m.get("content", []):
                if c.get("type") == "text":
                    return c.get("text", "").strip()
    return ""


def cast_batch_to_dtype(batch, dtype):
    for k, v in batch.items():
        if torch.is_tensor(v) and v.is_floating_point():
            batch[k] = v.to(dtype)
    return batch



def get_model_compute_dtype(model):
    """
    以 lm_head 權重 dtype 為準 (最容易出現 mismatch 的地方就是 lm_head)
    PEFT 模型也適用：model.lm_head 一般仍存在
    """
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.dtype
    # 後備：找第一個浮點參數 dtype
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    # 最後後備
    return torch.float32



@torch.inference_mode()
def generate_one(ex, model, processor, resize_h, resize_w, max_new_tokens, SYSTEM_MESSAGE, do_resize):
    
    # 避免一些 generation_config 殘留 sampling 參數造成 warning
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    
    img_path = ex["images"][0]
    query = ex["query"].strip()
    user_content = []
    img_item = {"type":"image", "image": f"file://{img_path}"}
    if do_resize:
        img_item["resized_height"] = resize_h
        img_item["resized_width"]  = resize_w
    user_content.append(img_item)
    user_content.append({"type":"text", "text": query})
    messages = [
        {"role":"system","content":[{"type":"text","text": SYSTEM_MESSAGE}]},
        {"role":"user","content": user_content},
    ]
    
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # example["images"] 是 PIL list（你前面 format_example 已經準備好）
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_resize=False,  # 避免二次 Resize
    ).to(model.device)

    target_dtype = get_model_compute_dtype(model)
    # CPU 上不建議用 bf16/fp16 autocast，直接 float32
    if model.device.type != "cuda":
        target_dtype = torch.float32
    
    # 把 float32 的 pixel_values / pixel_values_videos 等轉成 bf16
    inputs = cast_batch_to_dtype(inputs, target_dtype)

    if model.device.type == "cuda" and target_dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type="cuda", dtype=target_dtype):
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
    else:
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    ans = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    
    return ans


def generate_k(eval_dataset, out_dir, model, processor, resize_h, resize_w, max_new_tokens=128, k=8,SYSTEM_MESSAGE="You are a helpful assistant.", do_resize=False):
    # 抽幾筆測試
    kn = min(k, len(eval_dataset))
    sample_idx = random.sample(range(len(eval_dataset)), k=kn)
    rows = []
    print("\n=== SAMPLE PRED (val) ===")
    for idx in sample_idx:
        ex = eval_dataset[idx]
        img_path = ex["images"][0]
        query = ex["query"].strip()
        ref = ex["response"].strip()
        
        pred = generate_one(ex, model, processor, resize_h, resize_w, max_new_tokens=max_new_tokens, SYSTEM_MESSAGE=SYSTEM_MESSAGE, do_resize=do_resize)
        print("\nIMG:", ex["images"][0])
        print("Q:", ex.get("query"))
        print("GT:", ex["response"])
        print("PRED:", pred)
        
        
        
        rows.append({"idx": idx, "question": query, "ref": ref, "pred": pred})

    print(f"\n")
    print(f"-"*50)
    bleu_socre(rows)
    
    # 存下來方便比對
    with open(os.path.join(out_dir, "qualitative_samples.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
