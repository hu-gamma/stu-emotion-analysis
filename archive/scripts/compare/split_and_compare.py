"""
重新划分测试集并对比新旧模型
1. 从 all_merged_v2.csv 分层划分 80% 训练 / 20% 测试
2. 用训练集训练 v4 模型
3. 在测试集上同时评估 v4（新模型）和旧 16 类模型（映射到 10 类）
"""
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, '/mnt')
from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import ECSADataset, get_num_classes, get_label_name


def split_dataset():
    """分层划分 all_merged_v2.csv"""
    print("=" * 60)
    print("1. 分层划分数据集")
    print("=" * 60)

    df = pd.read_csv('/mnt/datasets/csv/all_merged_v2.csv')
    print(f"原始数据: {len(df)} 条")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df['fine_grained_emotion']
    )
    print(f"训练集: {len(train_df)} 条")
    print(f"测试集: {len(test_df)} 条")

    train_path = '/mnt/datasets/csv/split_train.csv'
    test_path = '/mnt/datasets/csv/split_test.csv'
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    print("\n训练集标签分布:")
    for emo, cnt in train_df['fine_grained_emotion'].value_counts().sort_index().items():
        print(f"  {emo}: {cnt}")
    print("\n测试集标签分布:")
    for emo, cnt in test_df['fine_grained_emotion'].value_counts().sort_index().items():
        print(f"  {emo}: {cnt}")

    return train_path, test_path


def evaluate_model_direct(model_path, test_csv, num_classes=10, is_16class=False):
    """直接评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n加载模型: {model_path} ({'16类映射到10类' if is_16class else '10类'})")

    model = RoBERTaBiLSTM(num_classes=16 if is_16class else num_classes,
                          dropout_rate=0.1, lstm_hidden_size=256)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    test_dataset = ECSADataset(
        csv_path=test_csv,
        tokenizer=tokenizer,
        max_length=128,
        text_column='transformed_text'
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # 16→10 映射组
    old_to_new_groups = [
        [0, 13, 6], [1, 12, 15], [2], [3], [4, 14],
        [5], [7], [8], [9], [11, 10]
    ]

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)

            if is_16class:
                old_logits = outputs.cpu().numpy()
                new_logits = np.zeros((old_logits.shape[0], 10), dtype=np.float32)
                for new_id, old_ids in enumerate(old_to_new_groups):
                    new_logits[:, new_id] = old_logits[:, old_ids].sum(axis=1)
                predicted = torch.tensor(new_logits.argmax(axis=1)).to(device)
            else:
                _, predicted = torch.max(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = correct / total
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    print("\n  分类报告:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(10)),
        target_names=[get_label_name(i) for i in range(10)],
        digits=4, zero_division=0
    ))

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'y_true': y_true,
        'y_pred': y_pred
    }


def main():
    train_csv, test_csv = split_dataset()

    print("\n" + "=" * 60)
    print("2. 在新测试集上评估旧模型 (16类→10类)")
    print("=" * 60)
    old_results = evaluate_model_direct(
        '/mnt/checkpoints/roberta_bilstm_full/best_model.pt',
        test_csv, is_16class=True
    )

    print("\n" + "=" * 60)
    print("3. 在新测试集上评估 V3 模型 (当前已训练好的10类)")
    print("=" * 60)
    v3_results = evaluate_model_direct(
        '/mnt/checkpoints/roberta_bilstm_v3/best_model.pt',
        test_csv, is_16class=False
    )

    print("\n" + "=" * 60)
    print("4. 对比汇总")
    print("=" * 60)
    print(f"{'指标':<20} {'旧模型(16→10)':<18} {'V3模型(10类)':<18} {'变化':<10}")
    print("-" * 70)
    for key in ['accuracy', 'macro_f1', 'weighted_f1']:
        old_v = old_results[key]
        new_v = v3_results[key]
        diff = new_v - old_v
        diff_pct = (diff / old_v * 100) if old_v != 0 else 0
        old_str = f"{old_v*100:.2f}%" if key == 'accuracy' else f"{old_v:.4f}"
        new_str = f"{new_v*100:.2f}%" if key == 'accuracy' else f"{new_v:.4f}"
        print(f"{key:<20} {old_str:<18} {new_str:<18} {diff_pct:+.1f}%")


if __name__ == "__main__":
    main()
