"""
RoBERTa-BiLSTM 在测试集上逐条预测，生成 compare_models.py 需要的 jsonl 格式
每行: {"gold": ..., "pred": ..., "latency_ms": ...}
"""
import os
import sys
import json
import time
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, "/mnt")

from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import ECSADataset, get_num_classes, get_label_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/mnt/checkpoints/roberta_bilstm_split/best_model.pt")
    parser.add_argument("--test_csv", type=str,
                        default="/mnt/data_processed/split_10class/test.csv")
    parser.add_argument("--output", type=str,
                        default="/mnt/results/compare/roberta_predictions.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"模型: {args.model_path}")
    print(f"测试集: {args.test_csv}")

    # 加载模型
    num_classes = get_num_classes()
    model = RoBERTaBiLSTM(num_classes=num_classes, dropout_rate=0.1, lstm_hidden_size=256)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"✓ 模型加载成功 (类别数: {num_classes})")

    # 加载测试数据
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    test_dataset = ECSADataset(
        csv_path=args.test_csv,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_column="transformed_text",
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 逐条预测（测量延迟）
    results = []
    correct = 0
    total = 0

    print(f"\n开始预测 ({len(test_loader)} 条)...")
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["labels"].item()
        gold = get_label_name(label)

        start = time.time()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        latency = (time.time() - start) * 1000  # ms

        pred_id = torch.argmax(outputs, dim=1).item()
        pred = get_label_name(pred_id)

        if pred == gold:
            correct += 1
        total += 1

        results.append({
            "gold": gold,
            "pred": pred,
            "latency_ms": round(latency, 2),
        })

        if total % 1000 == 0:
            print(f"  已处理 {total}/{len(test_loader)} 条 (acc={correct/total:.4f})")

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = correct / total
    print(f"\n预测完成!")
    print(f"  总样本: {total}")
    print(f"  准确率: {acc:.4f}")
    print(f"  平均延迟: {sum(r['latency_ms'] for r in results)/len(results):.2f} ms")
    print(f"  结果已保存: {args.output}")


if __name__ == "__main__":
    main()
