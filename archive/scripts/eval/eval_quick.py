"""快速评估脚本，不画图，只输出指标"""
import sys
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, '/mnt')
from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import ECSADataset, get_num_classes, get_label_name


def quick_eval(model_path, test_csv, batch_size=16, max_length=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"模型: {model_path}")
    print(f"测试集: {test_csv}")

    num_classes = get_num_classes()
    model = RoBERTaBiLSTM(num_classes=num_classes, dropout_rate=0.1, lstm_hidden_size=256)
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
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
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
        labels=list(range(num_classes)),
        target_names=[get_label_name(i) for i in range(num_classes)],
        digits=4, zero_division=0
    ))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--test-csv', '-t', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    quick_eval(args.model, args.test_csv, args.batch_size)
