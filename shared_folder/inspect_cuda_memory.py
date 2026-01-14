import torch

# 取得 GPU 總量 (單位轉換成 GB)
total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU 總顯存: {total_memory:.2f} GB")

# 取得目前剩餘可用的顯存
free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
print(f"目前可用顯存: {free_memory:.2f} GB")