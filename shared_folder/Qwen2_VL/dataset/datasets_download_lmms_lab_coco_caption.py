import json, os
from datasets import load_dataset
from PIL import Image
import time

out_img_dir = "./lmms_lab_coco_caption/images"
out_jsonl = "./lmms_lab_coco_caption/train.jsonl"
os.makedirs(out_img_dir, exist_ok=True)

'''
# 查看資料集 train、val、test
builder = load_dataset("lmms-lab/COCO-Caption", cache_dir="./lmms-lab_coco-caption")
print(builder)           # 會列出 DatasetDict keys
print(builder.keys())    # 例如 dict_keys(['val','test'])
'''

# 下載資料集
custom_path = "./lmms_lab_coco_caption"
os.makedirs(custom_path, exist_ok=True)
ds = load_dataset("lmms-lab/COCO-Caption", cache_dir=custom_path, split="val[:100]")  # 先取小一點跑通

sample1 = next(iter(ds))
print(sample1)
print(sample1.items())


'''
{'question_id': 'COCO_val2014_000000203564.jpg',
 
 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=400x400 at 0x79FDC2E77530>, 
 
 'question': 'Please carefully observe the image and come up with a caption for the image.', 
 
 'answer': ['A bicycle replica with a clock as the front wheel.', 'The bike has a clock as a tire.', 'A black metal bicycle with a clock inside the front wheel.', 'A bicycle figurine in which the front wheel is replaced with a clock\n', 'A clock with the appearance of the wheel of a bicycle '], 'id': 37, 'license': 4,
 
 'file_name': 'COCO_val2014_000000203564.jpg', 
 
 'coco_url': 'http://images.cocodataset.org/val2014/COCO_val2014_000000203564.jpg', 
 'height': 400, 
 'width': 400, 
 'date_captured': '2013-11-15 03:12:47'}
'''


with open(out_jsonl, "w", encoding="utf-8") as f:
    for ex in ds:
        time.sleep(0.01)
        img_path = os.path.join(out_img_dir, ex["file_name"])
        ex["image"].convert("RGB").save(img_path, quality=95)
        img_path = os.path.abspath(img_path)

        for ans in ex["answer"]:
            item = {
                "query": f"<image>{ex['question']}",
                "response": ans.strip(),
                "images": [img_path],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            time.sleep(0.001)
            
print("wrote:", out_jsonl)