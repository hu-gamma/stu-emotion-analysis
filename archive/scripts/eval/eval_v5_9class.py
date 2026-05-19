"""
评估 V5 模型 (9类: 快乐+满足合并)
"""
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, '/mnt')
from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v3 import ECSADataset, get_label_name


def evaluate(model_path, test_csv, name="V5"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"评估: {name} (9类)")
    print(f"{'='*60}")

    model = RoBERTaBiLSTM(num_classes=9, dropout_rate=0.1, lstm_hidden_size=256)
    cp = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    ds = ECSADataset(csv_path=test_csv, tokenizer=tokenizer, max_length=128, text_column='transformed_text')
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            _, pred = torch.max(outputs, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, labels=list(range(9)),
          target_names=[get_label_name(i) for i in range(9)], digits=4, zero_division=0))

    return {'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}


if __name__ == "__main__":
    evaluate('/mnt/checkpoints/roberta_bilstm_v5/best_model.pt',
             '/mnt/datasets/csv/split_test.csv', name="V5 (9类)")
