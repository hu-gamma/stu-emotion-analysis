"""
训练脚本 — V8 多轮对话模型
使用 multiturn_train.csv（包含上下文窗口数据）进行训练

用法:
    python scripts/train/train_multiturn.py \
        --epochs 20 --batch-size 64 --use-layerwise-lr \
        --save-dir checkpoints/roberta_bilstm_v8_multiturn_L4_d03
"""
import os
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer

sys.path.insert(0, '/mnt')
from models.model_v8 import RoBERTaBiLSTMV8
from data.dataset_csv_loader_v5 import get_num_classes, get_label_name


class MultiTurnDataset(Dataset):
    """
    多轮对话数据集
    将 prev_text + curr_text 拼接为 [CLS] prev [SEP] curr [SEP]
    """
    EMOTION_MAP = {
        '悲伤': 0, '快乐': 1, '愤怒': 2, '焦虑': 3,
        '厌恶': 4, '惊讶': 5, '好奇': 6,
    }

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = []
        for _, row in df.iterrows():
            emotion = row['emotion']
            if emotion not in self.EMOTION_MAP:
                continue
            label = self.EMOTION_MAP[emotion]

            prev = str(row['prev_text']) if pd.notna(row['prev_text']) and row['prev_text'] else ''
            curr = str(row['curr_text'])

            # 拼接: [CLS] prev [SEP] curr [SEP]
            if prev:
                input_text = prev + tokenizer.sep_token + curr
            else:
                input_text = curr

            encoding = tokenizer(
                input_text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
            )

            self.samples.append({
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long),
                'has_context': 1 if prev else 0,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'input_ids': s['input_ids'],
            'attention_mask': s['attention_mask'],
            'labels': s['label'],
        }


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (score > self.best_score + self.min_delta if self.mode == 'max'
              else score < self.best_score - self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


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


def evaluate(model, loader, criterion, device, detailed=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            if detailed:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    macro_f1 = f1_score(np.array(all_labels), np.array(all_preds),
                        average='macro', zero_division=0) if detailed and all_preds else 0.0
    return avg_loss, accuracy, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-fusion-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--fusion-mode', type=str, default='weighted_sum')
    parser.add_argument('--use-layerwise-lr', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-csv', type=str,
                        default='/mnt/datasets/csv/multiturn_train.csv')
    args = parser.parse_args()

    if args.save_dir is None:
        lw_tag = 'lw' if args.use_layerwise_lr else 'unified'
        args.save_dir = f"/mnt/checkpoints/roberta_bilstm_v8_multiturn_L{args.num_fusion_layers}_d{int(args.dropout*10):02d}_{lw_tag}"
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"设备: {device}")
    print(f"配置: L{args.num_fusion_layers} d{args.dropout} {args.fusion_mode} {'layerwise' if args.use_layerwise_lr else 'unified'}")
    print(f"多轮训练集: {args.train_csv}")

    # 加载数据
    df = pd.read_csv(args.train_csv)
    print(f"加载 {len(df)} 条样本")
    print(f"上下文窗口样本: {(df['prev_text'] != '').sum()} 条 ({(df['prev_text'] != '').mean()*100:.1f}%)")

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    dataset = MultiTurnDataset(df, tokenizer, max_length=128)

    # 划分训练/验证集
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(args.seed))

    # 计算类别权重
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

    # WeightedRandomSampler
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 模型
    model = RoBERTaBiLSTMV8(
        num_classes=get_num_classes(), dropout_rate=args.dropout,
        lstm_hidden_size=256, num_fusion_layers=args.num_fusion_layers,
        fusion_mode=args.fusion_mode)
    model = model.to(device)
    print(f"总参数: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    if args.use_layerwise_lr:
        param_groups = model.get_param_groups(base_lr=args.lr)
        optimizer = AdamW(param_groups, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)

    # 训练
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='max')
    best_f1 = -1.0

    print(f"\n{'='*60}\n开始多轮对话模型训练\n{'='*60}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}\n{'-' * 60}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, detailed=True)
        scheduler.step()

        lw = model.get_layer_weights()
        if lw is not None:
            print(f"  层权重: {', '.join([f'L{i+1}={w:.3f}' for i, w in enumerate(lw)])}")
        print(f"  训练: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
        print(f"  验证: Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%, Macro F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': val_f1,
                'num_classes': get_num_classes(),
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ✓ 最佳模型已保存 (Macro F1={val_f1:.4f})")

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("\n早停触发")
            break

    print(f"\n训练完成! 最佳验证 Macro F1: {best_f1:.4f}")
    print(f"模型保存至: {args.save_dir}")


if __name__ == '__main__':
    main()
