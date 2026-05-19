"""
导出 RoBERTa-BiLSTM 在测试集上的逐条预测结果，
包含 gold label、pred label、推理延迟，用于与 Qwen 对比。
"""
import json
import os
import time
import sys

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt")

from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import ECSADataset, get_num_classes, get_label_name

MODEL_PATH = "/mnt/checkpoints/roberta_bilstm_full/best_model.pt"
TEST_CSV = "/mnt/datasets/csv/dreaddit_test_processed.csv"
OUTPUT_PRED = "/mnt/results/compare/roberta_predictions.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PRED), exist_ok=True)

    print(f"设备: {DEVICE}")
    print(f"加载模型: {MODEL_PATH}")

    num_classes = get_num_classes()
    model = RoBERTaBiLSTM(num_classes=num_classes, dropout_rate=0.1, lstm_hidden_size=256)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    print(f"✓ 模型加载成功 ({num_classes} 类)")

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    test_dataset = ECSADataset(
        csv_path=TEST_CSV,
        tokenizer=tokenizer,
        max_length=128,
        text_column="transformed_text",
    )

    # 逐条推理以精确测量延迟
    records = []
    for i in tqdm(range(len(test_dataset)), desc="RoBERTa inference"):
        item = test_dataset[i]
        text = item["raw_text"]
        gold_id = item["labels"].item()
        gold_label = get_label_name(gold_id)

        input_ids = item["input_ids"].unsqueeze(0).to(DEVICE)
        attention_mask = item["attention_mask"].unsqueeze(0).to(DEVICE)

        start = time.time()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
        latency = (time.time() - start) * 1000  # ms

        pred_id = predicted.item()
        pred_label = get_label_name(pred_id)

        records.append({
            "text": text,
            "gold": gold_label,
            "pred": pred_label,
            "latency_ms": round(latency, 2),
        })

    with open(OUTPUT_PRED, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ 预测结果已保存: {OUTPUT_PRED} ({len(records)} 条)")


if __name__ == "__main__":
    main()
