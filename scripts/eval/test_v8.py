"""
V8 模型测试脚本（7类，去掉尴尬类）
生成 ROC 曲线、混淆矩阵、分类报告
"""
import os
import sys
import argparse
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, '/mnt')
from models.model_v8 import RoBERTaBiLSTMV8
from data.dataset_csv_loader_v5 import (
    ECSADataset, get_num_classes, get_label_name
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

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
        'loss': avg_loss, 'accuracy': accuracy,
        'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob,
        'report': report,
    }


def plot_roc(y_true, y_prob, n_classes, result_dir, label_names, prefix=''):
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = dict(), dict(), dict()
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
    fpr['macro'], tpr['macro'] = all_fpr, mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(fpr['micro'], tpr['micro'],
            label=f"Micro-average (AUC={roc_auc['micro']:.4f})",
            color='deeppink', linestyle=':', linewidth=2.5)
    ax.plot(fpr['macro'], tpr['macro'],
            label=f"Macro-average (AUC={roc_auc['macro']:.4f})",
            color='navy', linestyle=':', linewidth=2.5)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2.0,
                label=f"{label_names[i]} (AUC={roc_auc[i]:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title('Multi-class ROC Curve (One-vs-Rest)', fontsize=18)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9, ncol=1)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, linestyle='--', alpha=0.6)

    roc_path = os.path.join(result_dir, f'{prefix}roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ROC 曲线已保存: {roc_path}")
    return roc_path


def plot_confusion_matrix(y_true, y_pred, n_classes, result_dir, label_names, prefix=''):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    # ---- Figure 1: Raw counts ----
    fig1, ax1 = plt.subplots(figsize=(14, 12))
    im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=12)
    ax1.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
            xticklabels=label_names, yticklabels=label_names,
            title='Confusion Matrix — Raw Counts',
            ylabel='True Label', xlabel='Predicted Label')
    ax1.set_title('Confusion Matrix — Raw Counts', fontsize=18, pad=15)
    ax1.set_ylabel('True Label', fontsize=16)
    ax1.set_xlabel('Predicted Label', fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold')

    fig1.tight_layout()
    cm_counts_path = os.path.join(result_dir, f'{prefix}confusion_matrix_counts.png')
    fig1.savefig(cm_counts_path, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    print(f"  混淆矩阵(计数)已保存: {cm_counts_path}")

    # ---- Figure 2: Normalized by row ----
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=12)
    ax2.set(xticks=np.arange(cm_norm.shape[1]), yticks=np.arange(cm_norm.shape[0]),
            xticklabels=label_names, yticklabels=label_names,
            title='Confusion Matrix — Normalized by Row (Recall)',
            ylabel='True Label', xlabel='Predicted Label')
    ax2.set_title('Confusion Matrix — Normalized by Row (Recall)', fontsize=18, pad=15)
    ax2.set_ylabel('True Label', fontsize=16)
    ax2.set_xlabel('Predicted Label', fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(j, i, f'{cm_norm[i, j]:.2f}',
                    ha="center", va="center", fontsize=14,
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontweight='bold')
    # Add recall values to y-axis labels
    recalls = [f'{label_names[i]}\n(recall={cm_norm[i,i]:.2%})' for i in range(len(label_names))]
    ax2.set_yticklabels(recalls)

    fig2.tight_layout()
    cm_norm_path = os.path.join(result_dir, f'{prefix}confusion_matrix_norm.png')
    fig2.savefig(cm_norm_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"  混淆矩阵(归一化)已保存: {cm_norm_path}")

    return cm_counts_path, cm_norm_path


def save_report_md(result_dir, overall, per_class_lines, roc_path, cm_counts_path, cm_norm_path,
                   label_names, model_path, test_csv, num_classes, prefix=''):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md_path = os.path.join(result_dir, f'{prefix}report.md')

    per_class_table = "| 情感 | Precision | Recall | F1-Score | Support |\n"
    per_class_table += "|------|-----------|--------|----------|----------|\n"
    for line in per_class_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            per_class_table += f"| {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]} |\n"

    md_content = f"""# V8 情感分析测试报告（7类，去掉尴尬）

**生成时间**: {timestamp}

## 1. 配置信息

- **模型架构**: RoBERTa-BiLSTM V8 (7分类 + 多层融合 + Attention Pooling)
- **模型路径**: {model_path}
- **测试集**: {test_csv}
- **类别数**: {num_classes} 分类
- **类别**: {', '.join(label_names)}

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

### 混淆矩阵（原始计数）
![Confusion Matrix Counts](./confusion_matrix_counts.png)

### 混淆矩阵（按行归一化，Recall）
![Confusion Matrix Normalized](./confusion_matrix_norm.png)
"""

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  报告已保存: {md_path}")


def test(model_path, test_csv, result_base_dir='/mnt/results',
         batch_size=32, max_length=128, text_column='transformed_text'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    print(f"测试集: {test_csv}")

    num_classes = get_num_classes()
    print(f"类别数: {num_classes}")

    model = RoBERTaBiLSTMV8(num_classes=num_classes)
    print("✓ 使用 V8 RoBERTa-BiLSTM + Attention Pooling 模型（7类）")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✓ 模型加载成功")
    if 'val_macro_f1' in checkpoint:
        print(f"  训练时最佳验证 Macro F1: {checkpoint['val_macro_f1']:.4f}")

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    test_dataset = ECSADataset(
        csv_path=test_csv,
        tokenizer=tokenizer,
        max_length=max_length,
        text_column=text_column
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n{'='*60}")
    print("开始测试评估")
    print(f"{'='*60}")
    results = evaluate_model(model, test_loader, device)

    print(f"\n测试损失: {results['loss']:.4f}")
    print(f"测试准确率: {results['accuracy']*100:.2f}%")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print("\n详细分类报告:")
    print(results['report'])

    result_dir = os.path.join(result_base_dir, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n保存结果到: {result_dir}")

    label_names = [get_label_name(i) for i in range(num_classes)]
    per_class_lines = results['report'].strip().split('\n')[2:-3]

    prefix = datetime.now().strftime('%H%M%S_')

    roc_path = plot_roc(results['y_true'], results['y_prob'],
                        num_classes, result_dir, label_names, prefix)
    cm_counts_path, cm_norm_path = plot_confusion_matrix(results['y_true'], results['y_pred'],
                                    num_classes, result_dir, label_names, prefix)
    save_report_md(result_dir, results, per_class_lines, roc_path, cm_counts_path, cm_norm_path,
                   label_names, model_path, test_csv, num_classes, prefix)

    print(f"\n✓ 测试完成，结果保存在: {result_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='V8 RoBERTa-BiLSTM 模型测试脚本（7类）')
    parser.add_argument('--model', '-m', type=str,
                        default='/mnt/checkpoints/roberta_bilstm_v8_7class_L4_d03_lw/best_model.pt',
                        help='模型文件路径')
    parser.add_argument('--test-csv', '-t', type=str,
                        default='/mnt/datasets/csv/split_test.csv',
                        help='测试集 CSV 文件路径')
    parser.add_argument('--output', '-o', type=str, default='/mnt/results',
                        help='结果输出目录')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--text-column', type=str, default='transformed_text')
    args = parser.parse_args()

    test(model_path=args.model, test_csv=args.test_csv,
         result_base_dir=args.output, batch_size=args.batch_size,
         max_length=args.max_length, text_column=args.text_column)


if __name__ == "__main__":
    main()
