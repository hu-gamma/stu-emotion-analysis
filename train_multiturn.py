"""
快速训练多轮对话情感分析模型
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, '/mnt/stu-emotion-analysis')
from models.model_v8_multiturn import RoBERTaBiLSTMV8MultiTurn
from data.dataset_csv_loader_v5 import MultiTurnDataset, get_num_classes, get_label_name


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, desc="Training"):
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
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(np.array(all_labels), np.array(all_preds), average='macro', zero_division=0)
    return acc, macro_f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    save_dir = '/mnt/stu-emotion-analysis/checkpoints/multiturn'
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    df = pd.read_csv('/mnt/stu-emotion-analysis/datasets/csv/synthetic_multiturn.csv')
    dataset = MultiTurnDataset(df, tokenizer, max_length=128)

    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    # 类别权重
    train_labels = [dataset.samples[i]['label'].item() for i in train_ds.indices]
    counts = Counter(train_labels)
    weights = []
    for i in range(get_num_classes()):
        count = counts.get(i, 1)
        weights.append(1.0 / (count ** 0.5))
    class_weights = torch.tensor(weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.clamp(class_weights, min=0.1, max=5.0)
    print(f"类别权重: {class_weights.tolist()}")

    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 模型
    model = RoBERTaBiLSTMV8MultiTurn(
        num_classes=get_num_classes(),
        dropout_rate=0.3,
        lstm_hidden_size=256,
        num_fusion_layers=4,
        fusion_mode='weighted_sum',
    )
    model = model.to(device)
    print(f"总参数: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    param_groups = model.get_param_groups(base_lr=1e-5)
    optimizer = AdamW(param_groups, weight_decay=0.01)
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)

    # 训练
    best_f1 = -1.0
    epochs = 15
    patience = 5
    counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        lw = model.get_layer_weights()
        if lw is not None:
            print(f"  层权重: {', '.join([f'L{i+1}={w:.3f}' for i, w in enumerate(lw)])}")
        print(f"  训练: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
        print(f"  验证: Acc={val_acc*100:.2f}%, Macro F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_macro_f1': val_f1,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  ✓ 最佳模型已保存 (Macro F1={val_f1:.4f})")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("\n早停触发")
                break

    print(f"\n训练完成! 最佳验证 Macro F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
