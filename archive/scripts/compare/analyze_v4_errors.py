"""
V4 模型错误案例分析
1. 混淆矩阵热力图
2. 各类别主要误分类方向
3. 抽样展示典型错误案例
"""
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, '/mnt')
from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import ECSADataset, get_label_name


def analyze_errors(model_path, test_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"加载模型: {model_path}")

    model = RoBERTaBiLSTM(num_classes=10, dropout_rate=0.1, lstm_hidden_size=256)
    cp = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    ds = ECSADataset(csv_path=test_csv, tokenizer=tokenizer, max_length=128, text_column='transformed_text')
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_texts = []
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_texts.extend(batch['raw_text'])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    label_names = [get_label_name(i) for i in range(10)]

    # 1. 混淆矩阵
    print("\n" + "="*70)
    print("1. 混淆矩阵（真实标签 × 预测标签）")
    print("="*70)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    # 打印表头
    header = f"{'真实\\预测':<10}"
    for name in label_names:
        header += f"{name:<8}"
    print(header)
    print("-" * 90)

    for i, true_name in enumerate(label_names):
        row = f"{true_name:<10}"
        for j in range(10):
            row += f"{cm[i][j]:<8}"
        print(row)

    # 2. 各类别误分类分析
    print("\n" + "="*70)
    print("2. 各类别主要误分类方向")
    print("="*70)

    for true_id in range(10):
        true_name = label_names[true_id]
        total = cm[true_id].sum()
        correct = cm[true_id][true_id]
        errors = [(pred_id, cm[true_id][pred_id]) for pred_id in range(10) if pred_id != true_id]
        errors.sort(key=lambda x: x[1], reverse=True)

        print(f"\n[{true_name}] 共 {total} 条，正确 {correct} 条 ({correct/total*100:.1f}%)，错误 {total-correct} 条")
        print(f"  最易被误判为:")
        for pred_id, count in errors[:3]:
            if count > 0:
                pct = count / total * 100
                print(f"    → {label_names[pred_id]:<8} {count:>4} 条 ({pct:>5.1f}%)")

    # 3. 错误案例抽样
    print("\n" + "="*70)
    print("3. 典型错误案例抽样")
    print("="*70)

    df_test = pd.read_csv(test_csv)
    error_mask = (y_true != y_pred)
    error_indices = np.where(error_mask)[0]

    print(f"总错误数: {len(error_indices)} / {len(y_true)} ({len(error_indices)/len(y_true)*100:.1f}%)")

    # 按 (真实, 预测) 分组抽样
    for true_id in range(10):
        for pred_id in range(10):
            if true_id == pred_id:
                continue
            mask = (y_true == true_id) & (y_pred == pred_id)
            count = mask.sum()
            if count == 0:
                continue

            # 抽样最多3个
            indices = np.where(mask)[0]
            sample_idx = np.random.choice(indices, min(3, len(indices)), replace=False)

            print(f"\n--- [{label_names[true_id]}] 被误判为 [{label_names[pred_id]}] (共 {count} 条) ---")
            for idx in sample_idx:
                text = all_texts[idx]
                true_prob = y_prob[idx][true_id]
                pred_prob = y_prob[idx][pred_id]
                print(f"  原文: {text}")
                print(f"  真实标签概率: {true_prob:.3f}, 预测标签概率: {pred_prob:.3f}")
                print()


if __name__ == "__main__":
    analyze_errors(
        '/mnt/checkpoints/roberta_bilstm_v4/best_model.pt',
        '/mnt/datasets/csv/split_test.csv'
    )
