"""
训练脚本：RoBERTa-BiLSTM 不使用 Dreaddit 数据集
训练集：OCEMOTION + Dataset_ECSA 各自切分训练部分后合并
测试集：OCEMOTION + Dataset_ECSA 各自切分测试部分后合并
验证集：从合并训练集中切分 10%
改进点:
- 多数据集各自切分后合并训练/测试
- Focal Loss + 类别权重
- WeightedRandomSampler 动态重采样
- Macro F1 作为早停指标
"""
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

sys.path.insert(0, "/mnt")

from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import (
    load_split_merged_v2,
    get_num_classes,
    get_subset_labels,
)


class FocalLoss(nn.Module):
    """Focal Loss，用于缓解类别不平衡"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class EarlyStopping:
    """基于 Macro F1 的早停机制 (mode='max')"""
    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    return total_loss / len(train_loader), correct / total


def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    detailed: bool = False
) -> Tuple[float, float, float, Dict]:
    """返回 (loss, accuracy, macro_f1, details)"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
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

            if detailed:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    macro_f1 = 0.0
    if detailed:
        all_preds_arr = np.array(all_preds) if all_preds else None
        all_labels_arr = np.array(all_labels) if all_labels else None
        if all_preds_arr is not None and len(all_preds_arr) > 0:
            macro_f1 = f1_score(all_labels_arr, all_preds_arr, average='macro', zero_division=0)

    details = {}
    if detailed:
        details['all_preds'] = all_preds
        details['all_labels'] = all_labels
        details['all_probs'] = all_probs

    return avg_loss, accuracy, macro_f1, details


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-5,
    save_dir: str = '/mnt/checkpoints/roberta_bilstm_split',
    use_focal_loss: bool = False,
    focal_gamma: float = 1.5,
    class_weights: torch.Tensor = None,
) -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    num_classes = get_num_classes()
    print(f"类别数: {num_classes}")

    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"类别权重: {class_weights.tolist()}")

    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        print(f"使用 Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("使用 CrossEntropyLoss (带类别权重)")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, mode='max')

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_macro_f1': []
    }
    best_macro_f1 = -1.0

    print(f"\n{'='*60}")
    print(f"开始训练（数据集内切分版）")
    print(f"设备: {device}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {learning_rate}")
    print(f"优化器: AdamW")
    print(f"类别数: {num_classes} (多分类)")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_macro_f1, _ = evaluate(model, val_loader, criterion, device, detailed=True)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_macro_f1)

        print(f"\n训练结果:")
        print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc*100:.2f}%")
        print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc*100:.2f}%")
        print(f"  验证 Macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_macro_f1': val_macro_f1,
                'num_classes': num_classes,
            }, checkpoint_path)
            print(f"  ✓ 最佳模型已保存 (Macro F1={val_macro_f1:.4f}): {checkpoint_path}")

        early_stopping(val_macro_f1)
        if early_stopping.early_stop:
            print(f"\n⏹️ 早停触发，停止训练")
            break

    print(f"\n{'='*60}")
    print(f"训练完成! 最佳验证 Macro F1: {best_macro_f1:.4f}")
    print(f"{'='*60}\n")
    return history


class Tee:
    """同时输出到控制台和文件"""
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self):
        return self.console.isatty()

    def close(self):
        self.file.close()


def main():
    save_dir = '/mnt/checkpoints/roberta_bilstm_split'
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    tee = Tee(log_path)
    sys.stdout = tee

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"训练日志将保存至: {log_path}")

    # 两个数据集各自切分训练/测试，不使用 dreaddit
    csv_paths = [
        '/mnt/datasets/csv/OCEMOTION_processed.csv',
        '/mnt/datasets/csv/Dataset_ECSA_processed.csv',
    ]

    print("\n加载数据集（各自切分训练/测试集）...")
    (
        train_subset, val_subset, test_dataset,
        train_loader, val_loader, test_loader,
        tokenizer, class_weights, train_labels
    ) = load_split_merged_v2(
        csv_paths=csv_paths,
        batch_size=16,
        max_length=128,
        val_ratio=0.1,
        test_ratio=0.2,
        random_seed=42,
    )

    # 使用 WeightedRandomSampler 动态重采样
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=16,
        sampler=sampler,
        num_workers=0,
    )
    print(f"已启用 WeightedRandomSampler，每个 epoch 采样 {len(sample_weights)} 条")

    # 打印各类别在验证集和测试集中的分布
    val_labels = get_subset_labels(val_subset)
    test_labels = get_subset_labels(test_dataset)
    print(f"\n验证集分布: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
    print(f"测试集分布: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

    # 初始化模型
    num_classes = get_num_classes()
    print(f"\n初始化模型 (类别数: {num_classes})...")
    model = RoBERTaBiLSTM(num_classes=num_classes, dropout_rate=0.1, lstm_hidden_size=256)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 训练
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        learning_rate=1e-5,
        save_dir=save_dir,
        use_focal_loss=False,
        focal_gamma=1.5,
        class_weights=class_weights,
    )

    print("\n训练历史:")
    for i, (tl, ta, vl, va, vmf1) in enumerate(zip(
        history['train_loss'], history['train_acc'],
        history['val_loss'], history['val_acc'], history['val_macro_f1']
    )):
        print(f"Epoch {i+1}: Train Loss={tl:.4f}, Train Acc={ta*100:.2f}%, "
              f"Val Loss={vl:.4f}, Val Acc={va*100:.2f}%, Val Macro F1={vmf1:.4f}")

    # 最终在测试集上评估
    print(f"\n{'='*60}")
    print("在测试集上进行最终评估...")
    print(f"{'='*60}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_loss, test_acc, test_macro_f1, details = evaluate(
        model, test_loader, criterion, device, detailed=True
    )
    print(f"\n测试结果:")
    print(f"  测试损失: {test_loss:.4f} | 测试准确率: {test_acc*100:.2f}%")
    print(f"  测试 Macro F1: {test_macro_f1:.4f}")

    tee.close()


if __name__ == "__main__":
    main()
