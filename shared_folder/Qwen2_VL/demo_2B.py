import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from PIL import Image
import time

'''
model_path = "./models/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype="auto",
    local_files_only = True,
    # attn_implementation="flash_attention_2",
    
)
processor = AutoProcessor.from_pretrained(
    model_path, 
    local_files_only = True,
    )
'''

model_id = "Qwen/Qwen2-VL-2B-Instruct"  # 或改 "Qwen/Qwen2-VL-7B-Instruct"
CACHE_DIR = "./models"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto",
    torch_dtype="auto",
    cache_dir=CACHE_DIR,
    # attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(
    model_id, 
    cache_dir=CACHE_DIR, 
    )
print(processor.tokenizer.chat_template[:300])


local_image_path = "img/cat.jpg"
raw_image = Image.open(local_image_path).convert("RGB")
raw_text = "請描述這張圖片。用繁體中文"


'''
local_image_path = "./dataset/refcoco_grounding/images/581857.jpg"
raw_image = Image.open(local_image_path).convert("RGB")
raw_text = "Find <|object_ref_start|>the lady with the blue shirt<|object_ref_end|>"
'''


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": local_image_path},
            {"type": "text", "text": raw_text},
        ],
    }
]

start_time = time.perf_counter()

# 1. 生成文字 prompt
text = processor.apply_chat_template(
                messages, 
                tokenize=False,  # 只回傳字串，不做 tokenization
                add_generation_prompt=True # 在尾端加上讓模型開始回答的提示
                )
# 2. qwen-vl-utils 抽取/處理視覺輸入(圖 / 影片)
# 這邊會自行做 resize
image_inputs, video_inputs = process_vision_info(messages)

# 3. 檢查格式
print("image_inputs:", type(image_inputs), "len:", 0 if image_inputs is None else len(image_inputs))
if image_inputs:
    print(" first image item:", type(image_inputs[0]))

print("video_inputs:", type(video_inputs), "len:", 0 if video_inputs is None else len(video_inputs))
if video_inputs:
    # video_inputs[0] 可能是 frames / list / 其他結構
    print(" first video item:", type(video_inputs[0]))


inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    do_resize=False,   # 關掉二次 resize
).to(model.device)

with torch.inference_mode():
    out_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            )


# 把 prompt 部分切掉，只保留新生成 tokens（推薦）
gen_ids = out_ids[:, inputs.input_ids.shape[1]:]

ans = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
print("ans:", ans)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"執行時間：{execution_time:.4f} 秒")


