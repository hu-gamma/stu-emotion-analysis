"""
独立测试脚本：对已训练模型进行多轮/多数据集测试评估
支持：
  - 在指定 CSV 测试集上评估
  - 输出 Accuracy / Macro F1 / Weighted F1 / 分类报告
  - 自动生成 ROC 曲线、混淆矩阵、Markdown 报告
  - 支持多次测试不同数据集，结果按时间戳分目录保存
"""
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from models.model import RoBERTaBiLSTM
from models.model_v6b import RoBERTaBiLSTMDropoutEnhanced
from data.dataset_csv_loader_v2 import (
    ECSADataset, get_num_classes, get_label_name
)

# 可视化设置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """
    在测试集上评估模型，返回详细结果字典
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(get_num_classes())),
        target_names=[get_label_name(i) for i in range(get_num_classes())],
        digits=4, zero_division=0
    )

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'report': report,
    }


def plot_roc(y_true, y_prob, n_classes, result_dir, label_names, prefix=''):
    """绘制并保存 ROC 曲线"""
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr['micro'], tpr['micro'],
            label=f"Micro-average ROC (AUC = {roc_auc['micro']:.2f})",
            color='deeppink', linestyle=':', linewidth=2)
    ax.plot(fpr['macro'], tpr['macro'],
            label=f"Macro-average ROC (AUC = {roc_auc['macro']:.2f})",
            color='navy', linestyle=':', linewidth=2)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=1.5,
                label=f"{label_names[i]} (AUC = {roc_auc[i]:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Multi-class ROC Curve (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)

    roc_path = os.path.join(result_dir, f'{prefix}roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC 曲线已保存: {roc_path}")
    return roc_path


def plot_confusion_matrix(y_true, y_pred, n_classes, result_dir, label_names, prefix=''):
    """绘制并保存混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=label_names,
           yticklabels=label_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    cm_path = os.path.join(result_dir, f'{prefix}confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  混淆矩阵已保存: {cm_path}")
    return cm_path


def save_report_md(result_dir: str, overall: Dict, per_class_lines: List[str],
                   roc_path: str, cm_path: str, label_names: List[str],
                   model_path: str, test_csv: str, prefix=''):
    """生成 Markdown 结果报告"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md_path = os.path.join(result_dir, f'{prefix}report.md')

    per_class_table = "| 情感 | Precision | Recall | F1-Score | Support |\n"
    per_class_table += "|------|-----------|--------|----------|----------|\n"
    for line in per_class_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            emotion = parts[0]
            per_class_table += f"| {emotion} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]} |\n"

    md_content = f"""# 情感分析测试报告

**生成时间**: {timestamp}

## 1. 配置信息

- **模型**: {model_path}
- **测试集**: {test_csv}
- **类别数**: {len(label_names)} 分类

## 2. 整体指标

| 指标 | 数值 |
|------|------|
| 测试准确率 (Accuracy) | {overall['accuracy']*100:.2f}% |
| 测试损失 (Loss) | {overall['loss']:.4f} |
| Macro F1 | {overall['macro_f1']:.4f} |
| Weighted F1 | {overall['weighted_f1']:.4f} |

## 3. 各类别详细指标

{per_class_table}

## 4. 可视化结果

### ROC 曲线
![ROC Curve](./roc_curve.png)

### 混淆矩阵
![Confusion Matrix](./confusion_matrix.png)
"""

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  报告已保存: {md_path}")


def test(
    model_path: str,
    test_csv: str,
    result_base_dir: str = './result',
    batch_size: int = 16,
    max_length: int = 128,
    text_column: str = 'transformed_text',
):
    """
    在指定测试集上评估已训练模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    print(f"测试集: {test_csv}")

    # 加载模型
    num_classes = get_num_classes()
    if 'v6b' in model_path.lower():
        model = RoBERTaBiLSTMDropoutEnhanced(num_classes=num_classes)
        print("✓ 使用 V6b 增强 Dropout 模型")
    else:
        model = RoBERTaBiLSTM(num_classes=num_classes, dropout_rate=0.1, lstm_hidden_size=256)
        print("✓ 使用标准 RoBERTa-BiLSTM 模型")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✓ 模型加载成功 (类别数: {num_classes})")

    # 加载测试数据
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    test_dataset = ECSADataset(
        csv_path=test_csv,
        tokenizer=tokenizer,
        max_length=max_length,
        text_column=text_column
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 评估
    print("\n" + "=" * 60)
    print("开始测试评估")
    print("=" * 60)
    results = evaluate_model(model, test_loader, device)

    print(f"\n测试损失: {results['loss']:.4f}")
    print(f"测试准确率: {results['accuracy']*100:.2f}%")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print("\n详细分类报告:")
    print(results['report'])

    # 创建结果目录：按日期统一存放
    result_dir = os.path.join(
        result_base_dir,
        datetime.now().strftime("%Y-%m-%d")
    )
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n保存结果到: {result_dir}")

    label_names = [get_label_name(i) for i in range(num_classes)]
    per_class_lines = results['report'].strip().split('\n')[2:-3]

    # 文件前缀：时分秒，用于区分同一天的不同测试
    prefix = datetime.now().strftime('%H%M%S_')

    roc_path = plot_roc(
        results['y_true'], results['y_prob'],
        num_classes, result_dir, label_names, prefix
    )
    cm_path = plot_confusion_matrix(
        results['y_true'], results['y_pred'],
        num_classes, result_dir, label_names, prefix
    )
    save_report_md(
        result_dir, results, per_class_lines,
        roc_path, cm_path, label_names,
        model_path, test_csv, prefix
    )

    print(f"\n✓ 测试完成，结果保存在: {result_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='RoBERTa-BiLSTM 模型测试脚本')
    parser.add_argument('--model', '-m', type=str, default='/mnt/checkpoints/roberta_bilstm_full/best_model.pt',
                        help='模型文件路径')
    parser.add_argument('--test-csv', '-t', type=str, required=True,
                        help='测试集 CSV 文件路径')
    parser.add_argument('--output', '-o', type=str, default='./result',
                        help='结果输出目录')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--max-length', type=int, default=128,
                        help='最大序列长度')
    parser.add_argument('--text-column', type=str, default='transformed_text',
                        help='CSV 中文本列名')

    args = parser.parse_args()

    test(
        model_path=args.model,
        test_csv=args.test_csv,
        result_base_dir=args.output,
        batch_size=args.batch_size,
        max_length=args.max_length,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
