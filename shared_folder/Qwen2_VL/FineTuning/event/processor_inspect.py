import os
import torch
import torch.nn as nn

def list_linear_suffixes(model):
    s = set()
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            s.add(name.split(".")[-1])  # 只取最後一段，例如 q_proj
    return sorted(s)


def test_processor(ex, processor, model):
    print([m for m in ex["messages"] if m["role"]=="assistant"][0])
    print("*"*50)
    msgs = [m for m in ex["messages"] if m["role"] in ("system","user")]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt],
        images=ex["images"],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    #print(inputs.keys())
    print("pixel_values:", inputs['input_ids'].shape)
    print("attention_mask:", inputs['attention_mask'].shape)
    print("pixel_values:", inputs['pixel_values'].shape)

    

def test_processor_decode(processor):
    # 1. 確認特殊 token 在 tokenizer 裡不會被亂切
    s = 'Find <|object_ref_start|>standing<|object_ref_end|>'
    ids = processor.tokenizer(s).input_ids
    print("tokenized len:", len(ids))
    print("decoded:", processor.tokenizer.decode(ids))

    # 2. 確認 box 格式同理
    t = '<|box_start|>(171,92),(255,337)<|box_end|>'
    ids2 = processor.tokenizer(t).input_ids
    print("decoded box:", processor.tokenizer.decode(ids2))
    