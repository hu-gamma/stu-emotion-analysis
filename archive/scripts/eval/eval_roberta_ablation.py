"""
评估消融模型（无 Dreaddit），导出逐条预测结果用于对比。
"""
import json
import os
import time
import sys

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/mnt")

from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import ECSADataset, get_num_classes, get_label_name


def eval_and_export(model_path, test_csv, output_pred, device):
    os.makedirs(os.path.dirname(output_pred), exist_ok=True)

    print(f"[消融评估] 模型: {model_path}")
    print(f"测试集: {test_csv}")

    num_classes = get_num_classes()
    model = RoBERTaBiLSTM(num_classes=num_classes, dropout_rate=0.1, lstm_hidden_size=256)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    test_dataset = ECSADataset(
        csv_path=test_csv,
        tokenizer=tokenizer,
        max_length=128,
        text_column="transformed_text",
    )

    records = []
    for i in tqdm(range(len(test_dataset)), desc="Inference"):
        item = test_dataset[i]
        text = item["raw_text"]
        gold_id = item["labels"].item()
        gold_label = get_label_name(gold_id)

        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)

        start = time.time()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
        latency = (time.time() - start) * 1000

        pred_id = predicted.item()
        pred_label = get_label_name(pred_id)

        records.append({
            "text": text,
            "gold": gold_label,
            "pred": pred_label,
            "latency_ms": round(latency, 2),
        })

    with open(output_pred, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✓ 预测结果已保存: {output_pred} ({len(records)} 条)")
    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="/mnt/datasets/csv/dreaddit_test_processed.csv")
    parser.add_argument("--output_pred", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_and_export(args.model, args.test_csv, args.output_pred, device)
