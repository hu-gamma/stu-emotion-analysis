"""
评估旧模型（16类）在10类体系下的表现。
将16维输出按合并规则映射到10维后再计算指标。

合并规则（与 dataset_csv_loader_v2 一致）:
  悲伤 <- 悲伤(0) + 孤独(13) + 失望(6)
  快乐 <- 快乐(1) + 感激(12) + 自豪(15)
  愤怒 <- 愤怒(2)
  满足 <- 满足(3)
  焦虑 <- 焦虑(4) + 压力(14)
  中性 <- 中性(5)
  尴尬 <- 尴尬(7)
  厌恶 <- 厌恶(8)
  惊讶 <- 惊讶(9)
  好奇 <- 好奇(11) + 期待(10)
"""
import sys
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, '/mnt')
from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import ECSADataset, get_num_classes, get_label_name

# 16类 -> 10类 映射表（按logits相加）
# 目标10类顺序: 悲伤, 快乐, 愤怒, 满足, 焦虑, 中性, 尴尬, 厌恶, 惊讶, 好奇
OLD_TO_NEW_GROUPS = [
    [0, 13, 6],   # 悲伤
    [1, 12, 15],  # 快乐
    [2],          # 愤怒
    [3],          # 满足
    [4, 14],      # 焦虑
    [5],          # 中性
    [7],          # 尴尬
    [8],          # 厌恶
    [9],          # 惊讶
    [11, 10],     # 好奇
]


def map_logits(old_logits):
    """将16维logits映射到10维"""
    new_logits = np.zeros((old_logits.shape[0], 10), dtype=np.float32)
    for new_id, old_ids in enumerate(OLD_TO_NEW_GROUPS):
        new_logits[:, new_id] = old_logits[:, old_ids].sum(axis=1)
    return new_logits


def quick_eval_16as10(model_path, test_csv, batch_size=16, max_length=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"模型: {model_path}")
    print(f"测试集: {test_csv}")
    print("模式: 16类模型映射到10类评估\n")

    # 加载16类模型
    model = RoBERTaBiLSTM(num_classes=16, dropout_rate=0.1, lstm_hidden_size=256)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    test_dataset = ECSADataset(
        csv_path=test_csv,
        tokenizer=tokenizer,
        max_length=max_length,
        text_column='transformed_text'
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)  # [B, 16]

            # 映射到10维
            old_logits = outputs.cpu().numpy()
            new_logits = map_logits(old_logits)
            predicted = torch.tensor(new_logits.argmax(axis=1)).to(device)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = correct / total
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*50}")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"{'='*50}")
    print("\n分类报告:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(10)),
        target_names=[get_label_name(i) for i in range(10)],
        digits=4, zero_division=0
    ))
    return {'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--test-csv', '-t', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    quick_eval_16as10(args.model, args.test_csv, args.batch_size)
