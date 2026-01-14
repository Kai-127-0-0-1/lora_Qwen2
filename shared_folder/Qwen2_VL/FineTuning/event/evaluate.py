import math
import json, os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from qwen_vl_utils import process_vision_info

def eval_metrics(trainer, out_dir):

    eval_metrics_data = trainer.evaluate()
    
    # 常見會有 eval_loss
    if "eval_loss" in eval_metrics_data:
        eval_metrics_data["eval_ppl"] = math.exp(eval_metrics_data["eval_loss"])

    print("=== EVAL METRICS ===")
    for k, v in eval_metrics_data.items():
        print(f"{k}: {v}")

    # 存成 json
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_metrics_data, f, ensure_ascii=False, indent=2)

