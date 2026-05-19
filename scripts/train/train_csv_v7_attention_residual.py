"""
训练脚本 V7-Attention-Residual — RoBERTa-BiLSTM + Attention Pooling + Residual
用法:
    python scripts/train/train_csv_v7_attention_residual.py \
        --num-fusion-layers 4 --dropout 0.3 --use-layerwise-lr
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

sys.path.insert(0, '/mnt')
from models.model_v7_attention_residual import RoBERTaBiLSTMV7AttentionResidual
from data.dataset_csv_loader_v4 import (
    load_merged_train_test_v4, get_num_classes, get_subset_labels,
)


class FocalLoss(nn.Module):
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


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(train_loader, desc="Training"):
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
    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, criterion, device, detailed=False):
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

            if detailed:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    macro_f1 = 0.0
    if detailed and all_preds:
        macro_f1 = f1_score(np.array(all_labels), np.array(all_preds), average='macro', zero_division=0)

    return avg_loss, accuracy, macro_f1


def train(model, train_loader, val_loader, test_loader, device, epochs, optimizer,
          criterion, scheduler, save_dir, config_desc):
    os.makedirs(save_dir, exist_ok=True)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='max')

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_macro_f1': []
    }
    best_macro_f1 = -1.0

    print(f"\n{'='*60}")
    print(f"开始训练 {config_desc}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device, detailed=True)
        scheduler.step()

        lw = model.get_layer_weights()
        if lw is not None:
            lw_str = ", ".join([f"L{i+1}={w:.3f}" for i, w in enumerate(lw)])
            print(f"  层权重: {lw_str}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_macro_f1)

        print(f"  训练: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
        print(f"  验证: Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%, Macro F1={val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            ckpt = os.path.join(save_dir, 'best_model.pt')
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss, 'val_acc': val_acc,
                'val_macro_f1': val_macro_f1,
                'num_classes': get_num_classes(),
            }
            if lw is not None:
                save_dict['layer_weights'] = lw.tolist()
            torch.save(save_dict, ckpt)
            print(f"  ✓ 最佳模型已保存 (Macro F1={val_macro_f1:.4f})")

        early_stopping(val_macro_f1)
        if early_stopping.early_stop:
            print(f"\n⏹️ 早停触发")
            break

    print(f"\n{'='*60}")
    print("在测试集上评估最佳模型...")
    test_loss, test_acc, test_macro_f1 = evaluate(model, test_loader, criterion, device, detailed=True)
    print(f"测试集: Loss={test_loss:.4f}, Acc={test_acc*100:.2f}%, Macro F1={test_macro_f1:.4f}")
    print(f"{'='*60}\n")

    return history, best_macro_f1, test_macro_f1


class Tee:
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
    def isatty(self): return False
    def close(self):
        self.file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-fusion-layers', type=int, default=4, choices=[2, 3, 4])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--fusion-mode', type=str, default='weighted_sum', choices=['weighted_sum', 'concat'])
    parser.add_argument('--use-layerwise-lr', action='store_true', help='使用分层学习率')
    parser.add_argument('--roberta-lr-ratio', type=float, default=1.0)
    parser.add_argument('--bilstm-lr-ratio', type=float, default=5.0)
    parser.add_argument('--classifier-lr-ratio', type=float, default=5.0)
    parser.add_argument('--attention-lr-ratio', type=float, default=5.0)
    parser.add_argument('--residual-lr-ratio', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.save_dir is None:
        lw_tag = 'lw' if args.use_layerwise_lr else 'unified'
        args.save_dir = f"/mnt/checkpoints/roberta_bilstm_v7attnres_L{args.num_fusion_layers}_d{int(args.dropout*10):02d}_{lw_tag}"

    os.makedirs(args.save_dir, exist_ok=True)

    log_path = os.path.join(args.save_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    tee = Tee(log_path)
    sys.stdout = tee

    print(f"配置: fusion_layers={args.num_fusion_layers}, dropout={args.dropout}, "
          f"mode={args.fusion_mode}, layerwise_lr={args.use_layerwise_lr}")
    print(f"保存目录: {args.save_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_csv_paths = ['/mnt/datasets/csv/split_train_enhanced.csv']
    test_csv_path = '/mnt/datasets/csv/split_test.csv'

    (train_subset, val_subset, test_dataset,
     train_loader, val_loader, test_loader,
     tokenizer, class_weights, train_labels) = load_merged_train_test_v4(
        train_csv_paths=train_csv_paths,
        test_csv_path=test_csv_path,
        batch_size=args.batch_size,
        max_length=128,
        val_ratio=0.1,
        random_seed=args.seed,
    )

    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    num_classes = get_num_classes()
    model = RoBERTaBiLSTMV7AttentionResidual(
        num_classes=num_classes,
        dropout_rate=args.dropout,
        lstm_hidden_size=256,
        num_fusion_layers=args.num_fusion_layers,
        fusion_mode=args.fusion_mode,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}")

    if args.use_layerwise_lr:
        param_groups = model.get_param_groups(
            base_lr=args.lr,
            roberta_lr_ratio=args.roberta_lr_ratio,
            bilstm_lr_ratio=args.bilstm_lr_ratio,
            classifier_lr_ratio=args.classifier_lr_ratio,
            attention_lr_ratio=args.attention_lr_ratio,
            residual_lr_ratio=args.residual_lr_ratio,
        )
        optimizer = AdamW(param_groups, weight_decay=0.01)
        print(f"分层学习率: RoBERTa={args.lr*args.roberta_lr_ratio:.0e}, "
              f"BiLSTM={args.lr*args.bilstm_lr_ratio:.0e}, "
              f"Attention={args.lr*args.attention_lr_ratio:.0e}, "
              f"Residual={args.lr*args.residual_lr_ratio:.0e}, "
              f"Classifier={args.lr*args.classifier_lr_ratio:.0e}")
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print(f"统一学习率: {args.lr:.0e}")

    total_steps = len(train_loader) * args.epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    config_desc = (f"V7-Attention-Residual L{args.num_fusion_layers} d{args.dropout} "
                   f"{'layerwise' if args.use_layerwise_lr else 'unified'}")

    history, best_val_f1, test_f1 = train(
        model, train_loader, val_loader, test_loader,
        device, args.epochs, optimizer, criterion, scheduler,
        args.save_dir, config_desc
    )

    print(f"\n{'='*60}")
    print(f"实验完成!")
    print(f"最佳验证 Macro F1: {best_val_f1:.4f}")
    print(f"测试集 Macro F1: {test_f1:.4f}")
    print(f"{'='*60}")

    tee.close()


if __name__ == "__main__":
    main()
