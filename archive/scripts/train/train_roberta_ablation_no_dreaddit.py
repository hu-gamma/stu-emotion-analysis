"""
消融实验：RoBERTa-BiLSTM 不使用 Dreaddit 训练数据
训练集：OCEMOTION + ECSA（去掉 dreaddit_train）
测试集：dreaddit_test_processed.csv（与完整模型完全一致）
模型架构：与当前 best_model.pt 完全一致（BiLSTM 不变）
"""
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

sys.path.insert(0, "/mnt")

from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import (
    load_merged_train_test_v2,
    get_num_classes,
    get_subset_labels,
)


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


def train_epoch(model, train_loader, optimizer, criterion, device):
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


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    macro_f1 = f1_score(np.array(all_labels), np.array(all_preds), average='macro', zero_division=0)
    return avg_loss, accuracy, macro_f1


def main():
    save_dir = '/mnt/checkpoints/roberta_bilstm_no_dreaddit'
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    tee = Tee(log_path)
    sys.stdout = tee

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[消融实验] 不使用 Dreaddit 训练数据")
    print(f"使用设备: {device}")
    print(f"训练日志: {log_path}")

    # 去掉 dreaddit_train
    train_csv_paths = [
        '/mnt/datasets/csv/OCEMOTION_processed.csv',
        '/mnt/datasets/csv/Dataset_ECSA_processed.csv',
    ]
    test_csv_path = '/mnt/datasets/csv/dreaddit_test_processed.csv'

    print("\n加载数据集...")
    (
        train_subset, val_subset, test_dataset,
        train_loader, val_loader, test_loader,
        tokenizer, class_weights, train_labels
    ) = load_merged_train_test_v2(
        train_csv_paths=train_csv_paths,
        test_csv_path=test_csv_path,
        batch_size=16,
        max_length=128,
        val_ratio=0.1,
        random_seed=42,
    )

    # WeightedRandomSampler
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_subset, batch_size=16, sampler=sampler, num_workers=0,
    )
    print(f"已启用 WeightedRandomSampler")

    val_labels = get_subset_labels(val_subset)
    test_labels = get_subset_labels(test_dataset)
    print(f"\n验证集分布: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
    print(f"测试集分布: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

    # 模型：与当前完全一致
    num_classes = get_num_classes()
    print(f"\n初始化模型 (类别数: {num_classes})...")
    model = RoBERTaBiLSTM(num_classes=num_classes, dropout_rate=0.1, lstm_hidden_size=256)
    model = model.to(device)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数、优化器、调度器（与当前完全一致）
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 5
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    best_macro_f1 = -1.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}

    print(f"\n{'='*60}")
    print(f"开始训练（5 epochs, lr=1e-5, CrossEntropyLoss + 类别权重）")
    print(f"{'='*60}\n")

    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5")
        print("-" * 60)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)
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
            print(f"  ✓ 最佳模型已保存 (Macro F1={val_macro_f1:.4f})")

    print(f"\n{'='*60}")
    print(f"训练完成! 最佳验证 Macro F1: {best_macro_f1:.4f}")
    print(f"{'='*60}\n")

    for i, (tl, ta, vl, va, vmf1) in enumerate(zip(
        history['train_loss'], history['train_acc'],
        history['val_loss'], history['val_acc'], history['val_macro_f1']
    )):
        print(f"Epoch {i+1}: Train Loss={tl:.4f}, Train Acc={ta*100:.2f}%, "
              f"Val Loss={vl:.4f}, Val Acc={va*100:.2f}%, Val Macro F1={vmf1:.4f}")

    tee.close()


if __name__ == "__main__":
    main()
