import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
import time


def load_finetuned_model(CACHE_DIR, model_id, finetuned_model):
    # 1. 載入原始模型 (與訓練時的設定一致)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    
    # 2. 載入 LoRA 適配器權重
    model = PeftModel.from_pretrained(model, finetuned_model)
    model.eval()
    
    # 3. 載入 Processor
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)
    
    return model, processor

# 使用方式
cache_dir = '../models'
model_id = "Qwen/Qwen2-VL-2B-Instruct"
finetuned_model = "./models/qwen2vl-2b-lora" # 這是您訓練完 save_model 的路徑

model, processor = load_finetuned_model(cache_dir, model_id, finetuned_model)
print("eos_token:", processor.tokenizer.eos_token)
print("eos_token_id:", processor.tokenizer.eos_token_id)
print("-" * 50) 

start_time = time.perf_counter()

# 準備與訓練時一致的 Message 格式
local_image_path = "../img/cat.jpg"
raw_image = Image.open(local_image_path).convert("RGB")
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [
        {"type": "image", "image": raw_image},
        {"type": "text", "text": "這張圖片顯示了什麼?"}
    ]}
]

# 1. 處理 Prompt 模板
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 2. 處理圖像資訊
image_inputs, video_inputs = process_vision_info(messages)

# 3. 轉換為 Tensor
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

# 4. 生成結果
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]


output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output_text)


end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"執行時間：{execution_time:.4f} 秒")