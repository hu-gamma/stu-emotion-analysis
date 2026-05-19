"""
分类相关关系与过拟合可视化分析脚本
支持：混淆矩阵、类别相似度、t-SNE分布、过拟合诊断、混淆方向分析

用法:
    cd /mnt && python analysis/analyze_overfitting.py \
        --model /mnt/checkpoints/roberta_bilstm_v6b/best_model.pt \
        --test-csv /mnt/datasets/csv/split_test.csv \
        --num-classes 8 \
        --output /mnt/analysis/results_v6b
"""
import os
import sys
import argparse
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, silhouette_score
)
from sklearn.manifold import TSNE

sys.path.insert(0, '/mnt')
from models.model import RoBERTaBiLSTM
from models.model_v6b import RoBERTaBiLSTMDropoutEnhanced

# 可视化设置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 1. 数据加载（通用：适配不同版本的数据加载器）
# ============================================================================

def load_data(test_csv: str, num_classes: int, batch_size: int = 32, max_length: int = 128):
    """通用数据加载：根据类别数自动选择数据加载器"""
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

    if num_classes == 6:
        from data.dataset_csv_loader_v5 import ECSADataset, get_num_classes, get_label_name
    elif num_classes == 8:
        from data.dataset_csv_loader_v4 import ECSADataset, get_num_classes, get_label_name
    elif num_classes == 10:
        from data.dataset_csv_loader_v2 import ECSADataset, get_num_classes, get_label_name
    else:
        raise ValueError(f"不支持的类别数: {num_classes}，当前支持 6/8/10")

    test_dataset = ECSADataset(
        csv_path=test_csv,
        tokenizer=tokenizer,
        max_length=max_length,
        text_column='transformed_text'
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    label_names = [get_label_name(i) for i in range(num_classes)]
    return test_loader, label_names, ECSADataset


def load_train_data(train_csv: str, num_classes: int, batch_size: int = 32, max_length: int = 128):
    """加载训练数据用于过拟合对比"""
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

    if num_classes == 6:
        from data.dataset_csv_loader_v5 import ECSADataset
    elif num_classes == 8:
        from data.dataset_csv_loader_v4 import ECSADataset
    elif num_classes == 10:
        from data.dataset_csv_loader_v2 import ECSADataset
    else:
        raise ValueError(f"不支持的类别数: {num_classes}")

    train_dataset = ECSADataset(
        csv_path=train_csv,
        tokenizer=tokenizer,
        max_length=max_length,
        text_column='transformed_text'
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader


# ============================================================================
# 2. 模型评估与特征提取
# ============================================================================

def evaluate_with_features(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    """评估模型并提取中间特征（BiLSTM输出512维）"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []  # BiLSTM拼接后的特征 [batch, 512]

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 手动前向传播以提取特征（兼容 V6a 和 V6b）
            outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
            # V6b 使用 roberta_dropout，V6a 使用 dropout
            if hasattr(model, 'roberta_dropout'):
                roberta_out = model.roberta_dropout(outputs.last_hidden_state)
            else:
                roberta_out = model.dropout(outputs.last_hidden_state)
            lstm_out, (hidden, cell) = model.bilstm(roberta_out)
            forward_hidden = hidden[0]
            backward_hidden = hidden[1]
            features = torch.cat((forward_hidden, backward_hidden), dim=1)  # [batch, 512]
            # V6b 有额外的 lstm_dropout
            if hasattr(model, 'lstm_dropout'):
                features = model.lstm_dropout(features)

            # 继续分类
            logits = model.classifier(features)

            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, dim=1)

            total_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.append(features.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    features = np.vstack(all_features)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    per_class = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
    }

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'features': features,
        'per_class': per_class,
    }


# ============================================================================
# 3. 可视化函数
# ============================================================================

def plot_confusion_matrix_both(y_true, y_pred, n_classes, label_names, output_dir, prefix=''):
    """绘制原始和归一化混淆矩阵"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # 原始混淆矩阵
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    ax = axes[0]
    im = ax.imshow(cm_raw, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=label_names, yticklabels=label_names,
           title='Confusion Matrix (Raw Counts)', ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_raw.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm_raw[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_raw[i, j] > thresh else "black", fontsize=9)

    # 归一化混淆矩阵（按行）
    cm_norm = cm_raw.astype('float') / (cm_raw.sum(axis=1, keepdims=True) + 1e-8)
    ax = axes[1]
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Reds, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=label_names, yticklabels=label_names,
           title='Confusion Matrix (Row-Normalized)', ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, f'{prefix}confusion_matrix_both.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  混淆矩阵已保存: {path}")
    return cm_raw, cm_norm


def plot_class_similarity_matrix(model: nn.Module, label_names: List[str], output_dir: str, prefix=''):
    """基于分类器最后一层权重计算类别间余弦相似度"""
    # 提取最后一层 Linear 的权重: shape [num_classes, 256]
    classifier = model.classifier
    # classifier[-1] 是最后的 Linear(256, num_classes)
    weight = classifier[-1].weight.detach().cpu().numpy()  # [num_classes, 256]

    n_classes = len(label_names)
    similarity = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            w_i = weight[i]
            w_j = weight[j]
            sim = np.dot(w_i, w_j) / (np.linalg.norm(w_i) * np.linalg.norm(w_j) + 1e-8)
            similarity[i, j] = sim

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(similarity, interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=label_names, yticklabels=label_names,
           title='Class Similarity (Cosine of Classifier Weights)',
           ylabel='Class', xlabel='Class')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if abs(similarity[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{similarity[i, j]:.2f}",
                    ha="center", va="center", color=color, fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, f'{prefix}class_similarity_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  类别相似度矩阵已保存: {path}")
    return similarity


def plot_tsne(features: np.ndarray, labels: np.ndarray, label_names: List[str],
              output_dir: str, prefix=''):
    """t-SNE 特征分布可视化"""
    print("  正在计算 t-SNE（样本数=%d，特征维=%d）..." % (features.shape[0], features.shape[1]))
    # 如果样本太多，随机采样一部分加速
    n_samples = features.shape[0]
    if n_samples > 5000:
        indices = np.random.choice(n_samples, 5000, replace=False)
        features_sub = features[indices]
        labels_sub = labels[indices]
    else:
        features_sub = features
        labels_sub = labels

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_sub)-1),
                max_iter=1000, learning_rate='auto', init='pca')
    emb_2d = tsne.fit_transform(features_sub)

    fig, ax = plt.subplots(figsize=(12, 10))
    n_classes = len(label_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in enumerate(colors[:n_classes]):
        mask = labels_sub == i
        if mask.sum() > 0:
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=[color], label=label_names[i], alpha=0.6, s=15, edgecolors='none')

    ax.set_title('t-SNE Visualization of BiLSTM Features (Test Set)', fontsize=14)
    ax.legend(loc='best', fontsize=10, markerscale=2)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f'{prefix}tsne_feature_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE 分布图已保存: {path}")

    # 计算 silhouette score（特征空间分离度）
    sil_score = silhouette_score(features_sub, labels_sub)
    print(f"  整体 Silhouette Score: {sil_score:.4f} (越接近1越好)")
    return emb_2d, sil_score


def plot_overfitting_diagnosis(test_per_class: Dict, train_per_class: Dict,
                                label_names: List[str], output_dir: str, prefix=''):
    """训练集 vs 测试集 per-class F1 对比（过拟合信号）"""
    n_classes = len(label_names)
    train_f1 = train_per_class['f1']
    test_f1 = test_per_class['f1']
    gaps = train_f1 - test_f1

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 柱状图对比
    ax = axes[0]
    x = np.arange(n_classes)
    width = 0.35
    bars1 = ax.bar(x - width/2, train_f1, width, label='Train F1', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_f1, width, label='Test F1', color='coral', alpha=0.8)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1: Train vs Test (Overfitting Indicator)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # 过拟合差距柱状图
    ax = axes[1]
    colors = ['green' if g < 0.05 else 'orange' if g < 0.15 else 'red' for g in gaps]
    bars = ax.bar(label_names, gaps, color=colors, alpha=0.8)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1 Gap (Train - Test)', fontsize=12)
    ax.set_title('Overfitting Gap per Class', fontsize=14)
    ax.axhline(y=0.05, color='orange', linestyle='--', label='Mild threshold (0.05)')
    ax.axhline(y=0.15, color='red', linestyle='--', label='Severe threshold (0.15)')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax.annotate(f'{gap:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=9, fontweight='bold')

    fig.tight_layout()
    path = os.path.join(output_dir, f'{prefix}overfitting_per_class_f1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  过拟合诊断图已保存: {path}")
    return gaps


def plot_confusion_flow(cm_norm: np.ndarray, label_names: List[str], output_dir: str, prefix='', top_n=10):
    """混淆流向分析：找出最常见的错分对"""
    n_classes = len(label_names)
    errors = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                errors.append((label_names[i], label_names[j], cm_norm[i, j], cm_norm[i, i]))

    # 按错分比例排序
    errors.sort(key=lambda x: x[2], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    top_errors = errors[:top_n]
    y_labels = [f"{e[0]} → {e[1]}" for e in top_errors]
    values = [e[2] * 100 for e in top_errors]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_errors)))

    bars = ax.barh(y_labels[::-1], values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel('Misclassification Rate (%)', fontsize=12)
    ax.set_title(f'Top-{top_n} Misclassification Pairs (Normalized)', fontsize=14)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=10)

    fig.tight_layout()
    path = os.path.join(output_dir, f'{prefix}confusion_flow_top{top_n}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  混淆流向图已保存: {path}")
    return errors


# ============================================================================
# 4. 报告生成
# ============================================================================

def generate_report(output_dir: str, label_names: List[str], test_results: Dict,
                    train_results: Dict, cm_norm: np.ndarray, similarity: np.ndarray,
                    errors: List, gaps: np.ndarray, sil_score: float, prefix=''):
    """生成 Markdown 综合分析报告"""
    n_classes = len(label_names)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 最易混淆的类别对
    sim_pairs = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            sim_pairs.append((label_names[i], label_names[j], similarity[i, j]))
    sim_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # 过拟合排名
    gap_rank = sorted(zip(label_names, gaps), key=lambda x: x[1], reverse=True)

    # per-class F1
    test_f1 = test_results['per_class']['f1']
    train_f1 = train_results['per_class']['f1'] if train_results else [0]*n_classes

    md = f"""# 分类相关关系与过拟合诊断报告

**生成时间**: {timestamp}

---

## 1. 整体指标

| 指标 | 测试集 | 训练集 |
|------|--------|--------|
| Accuracy | {test_results['accuracy']*100:.2f}% | {train_results['accuracy']*100:.2f}% |
| Loss | {test_results['loss']:.4f} | {train_results['loss']:.4f}% |
| Macro F1 | {test_results['macro_f1']:.4f} | {train_results['macro_f1']:.4f} |
| Weighted F1 | {test_results['weighted_f1']:.4f} | {train_results['weighted_f1']:.4f} |
| Silhouette Score | {sil_score:.4f} | - |

## 2. 各类别详细指标

| 类别 | 测试 F1 | 训练 F1 | 过拟合差距 | Support |
|------|---------|---------|-----------|---------|
"""
    for i, name in enumerate(label_names):
        gap = gaps[i] if train_results else 0
        sup = int(test_results['per_class']['support'][i])
        md += f"| {name} | {test_f1[i]:.4f} | {train_f1[i]:.4f} | {gap:+.4f} | {sup} |\n"

    md += f"""
## 3. 最易混淆的类别对 Top-5（基于分类器权重相似度）

| 排名 | 类别 A | 类别 B | 余弦相似度 | 解读 |
|------|--------|--------|-----------|------|
"""
    for rank, (a, b, sim) in enumerate(sim_pairs[:5], 1):
        level = "极高" if abs(sim) > 0.5 else "高" if abs(sim) > 0.3 else "中等"
        md += f"| {rank} | {a} | {b} | {sim:+.3f} | {level}相似，决策边界接近 |\n"

    md += f"""
## 4. 最常见错分流向 Top-10

| 排名 | 真实类别 | 被错分为 | 错分比例 | 原类召回率 |
|------|---------|---------|---------|-----------|
"""
    for rank, (true_lbl, pred_lbl, rate, recall) in enumerate(errors[:10], 1):
        md += f"| {rank} | {true_lbl} | {pred_lbl} | {rate*100:.1f}% | {recall*100:.1f}% |\n"

    md += f"""
## 5. 过拟合严重程度排名

| 排名 | 类别 | 过拟合差距 (Train-Test F1) | 严重程度 |
|------|------|---------------------------|---------|
"""
    for rank, (name, gap) in enumerate(gap_rank, 1):
        if gap > 0.15:
            severity = "🔴 严重"
        elif gap > 0.05:
            severity = "🟠 中等"
        elif gap > 0.01:
            severity = "🟡 轻微"
        else:
            severity = "🟢 正常"
        md += f"| {rank} | {name} | {gap:+.4f} | {severity} |\n"

    md += f"""
## 6. 诊断建议

"""
    # 根据分析结果自动生成建议
    severe_overfit = [(n, g) for n, g in gap_rank if g > 0.15]
    mild_overfit = [(n, g) for n, g in gap_rank if 0.05 < g <= 0.15]
    high_sim = [(a, b, s) for a, b, s in sim_pairs[:3]]
    top_errors_3 = errors[:3]

    if severe_overfit:
        md += "### ⚠️ 严重过拟合类别\n\n"
        for name, gap in severe_overfit:
            md += f"- **{name}**：训练集和测试集 F1 差距 {gap:.4f}，模型对该类别严重过拟合。"
            md += "建议增加该类别的训练数据或使用更强的数据增强。\n"
        md += "\n"

    if mild_overfit:
        md += "### 中等过拟合类别\n\n"
        for name, gap in mild_overfit:
            md += f"- **{name}**：F1 差距 {gap:.4f}，存在轻微过拟合。\n"
        md += "\n"

    if high_sim:
        md += "### 📊 易混淆类别对\n\n"
        for a, b, sim in high_sim:
            md += f"- **{a} vs {b}**：分类器权重余弦相似度 {sim:+.3f}，决策边界非常接近。"
            md += f"在测试集中，{a} 有 {next((e[2]*100 for e in errors if e[0]==a and e[1]==b), 0):.1f}% 被错分为 {b}。"
            md += "建议：如果语义确实接近，可考虑合并这两个类别。\n"
        md += "\n"

    if top_errors_3:
        md += "### 🔀 主要错分模式\n\n"
        for true_lbl, pred_lbl, rate, recall in top_errors_3:
            md += f"- **{true_lbl} → {pred_lbl}**：{rate*100:.1f}% 的 {true_lbl} 样本被错分，"
            md += f"是第 {next((i+1 for i,e in enumerate(errors) if e[0]==true_lbl and e[1]==pred_lbl), '?')} 大错分对。\n"
        md += "\n"

    md += f"""### 💡 综合建议

1. **数据层面**：过拟合最严重的类别应优先补充数据。
2. **模型层面**：若某对类别相似度极高且持续混淆，考虑合并或设计层级分类器。
3. **正则化层面**：对 silhouette score 较低的类别簇，可尝试增大 dropout 或添加类别间间隔损失（如 Center Loss）。

---
*报告由 analyze_overfitting.py 自动生成*
"""

    path = os.path.join(output_dir, f'{prefix}report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"  综合分析报告已保存: {path}")
    return path


# ============================================================================
# 5. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='分类相关关系与过拟合可视化分析')
    parser.add_argument('--model', '-m', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--test-csv', '-t', type=str, required=True, help='测试集 CSV')
    parser.add_argument('--train-csv', type=str, default=None, help='训练集 CSV（用于过拟合对比，可选）')
    parser.add_argument('--num-classes', '-n', type=int, required=True, help='类别数 (6/8/10)')
    parser.add_argument('--output', '-o', type=str, default='/mnt/analysis/results', help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--no-train-eval', action='store_true', help='跳过训练集评估（仅测试集分析）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"模型: {args.model}")
    print(f"测试集: {args.test_csv}")
    print(f"类别数: {args.num_classes}")

    # 加载模型（自动检测 V6b）
    if 'v6b' in args.model.lower():
        model = RoBERTaBiLSTMDropoutEnhanced(num_classes=args.num_classes)
        print("✓ 使用 V6b 增强 Dropout 模型")
    else:
        model = RoBERTaBiLSTM(num_classes=args.num_classes, dropout_rate=0.1, lstm_hidden_size=256)
        print("✓ 使用标准 RoBERTa-BiLSTM 模型")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ 模型加载成功")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    prefix = datetime.now().strftime('%H%M%S_')
    print(f"\n输出目录: {args.output}")

    # 加载测试数据
    test_loader, label_names, _ = load_data(args.test_csv, args.num_classes, args.batch_size, args.max_length)

    # 1. 测试集评估 + 特征提取
    print("\n" + "=" * 60)
    print("1. 测试集评估与特征提取")
    print("=" * 60)
    test_results = evaluate_with_features(model, test_loader, device, args.num_classes)
    print(f"  Test Accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"  Test Macro F1: {test_results['macro_f1']:.4f}")

    # 2. 训练集评估（可选，用于过拟合对比）
    train_results = None
    if not args.no_train_eval and args.train_csv:
        print("\n" + "=" * 60)
        print("2. 训练集评估（过拟合对比）")
        print("=" * 60)
        train_loader = load_train_data(args.train_csv, args.num_classes, args.batch_size, args.max_length)
        train_results = evaluate_with_features(model, train_loader, device, args.num_classes)
        print(f"  Train Accuracy: {train_results['accuracy']*100:.2f}%")
        print(f"  Train Macro F1: {train_results['macro_f1']:.4f}")
    else:
        print("\n[跳过训练集评估，仅做测试集分析]")
        # 构造空的 train_results
        train_results = {
            'accuracy': 0, 'loss': 0, 'macro_f1': 0, 'weighted_f1': 0,
            'per_class': {'f1': np.zeros(args.num_classes)}
        }

    # 3. 可视化分析
    print("\n" + "=" * 60)
    print("3. 生成可视化图表")
    print("=" * 60)

    # 混淆矩阵
    cm_raw, cm_norm = plot_confusion_matrix_both(
        test_results['y_true'], test_results['y_pred'], args.num_classes,
        label_names, args.output, prefix
    )

    # 类别相似度
    similarity = plot_class_similarity_matrix(model, label_names, args.output, prefix)

    # t-SNE
    emb_2d, sil_score = plot_tsne(
        test_results['features'], test_results['y_true'], label_names, args.output, prefix
    )

    # 过拟合诊断
    gaps = plot_overfitting_diagnosis(
        test_results['per_class'], train_results['per_class'], label_names, args.output, prefix
    )

    # 混淆流向
    errors = plot_confusion_flow(cm_norm, label_names, args.output, prefix, top_n=10)

    # 4. 生成报告
    print("\n" + "=" * 60)
    print("4. 生成综合报告")
    print("=" * 60)
    generate_report(args.output, label_names, test_results, train_results,
                    cm_norm, similarity, errors, gaps, sil_score, prefix)

    print("\n" + "=" * 60)
    print(f"✓ 分析完成！所有结果保存在: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
