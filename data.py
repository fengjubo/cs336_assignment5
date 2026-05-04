import json
from pathlib import Path


data_path = Path(__file__).resolve().parents[1] / "data" / "gsm8k" / "train.jsonl"

with data_path.open() as f:
    ds = [json.loads(line) for line in f]

# 打印第一条看看，确认是否靠谱
print(ds[0])
