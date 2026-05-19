"""
公平对比：V4（无数据泄露） vs 旧模型（16→10映射）
在同一测试集 split_test.csv 上评估
"""
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, '/mnt')
from models.model import RoBERTaBiLSTM
from data.dataset_csv_loader_v2 import ECSADataset, get_label_name


def evaluate(model_path, test_csv, num_classes=10, is_16class=False, name="模型"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"评估: {name}")
    print(f"{'='*60}")

    model = RoBERTaBiLSTM(num_classes=16 if is_16class else num_classes,
                          dropout_rate=0.1, lstm_hidden_size=256)
    cp = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    ds = ECSADataset(csv_path=test_csv, tokenizer=tokenizer, max_length=128, text_column='transformed_text')
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    old_to_new = [[0,13,6],[1,12,15],[2],[3],[4,14],[5],[7],[8],[9],[11,10]]

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)

            if is_16class:
                old_logits = outputs.cpu().numpy()
                new_logits = np.zeros((old_logits.shape[0], 10), dtype=np.float32)
                for nid, oids in enumerate(old_to_new):
                    new_logits[:, nid] = old_logits[:, oids].sum(axis=1)
                pred = torch.tensor(new_logits.argmax(axis=1)).to(device)
            else:
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
    print(classification_report(y_true, y_pred, labels=list(range(10)),
          target_names=[get_label_name(i) for i in range(10)], digits=4, zero_division=0))

    return {'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}


def main():
    test_csv = '/mnt/datasets/csv/split_test.csv'

    old = evaluate('/mnt/checkpoints/roberta_bilstm_full/best_model.pt',
                   test_csv, is_16class=True, name="旧模型 (原始数据 + 16→10映射)")

    v4 = evaluate('/mnt/checkpoints/roberta_bilstm_v4/best_model.pt',
                  test_csv, is_16class=False, name="V4模型 (GoEmotions + 分层划分 / 无数据泄露)")

    print(f"\n{'='*60}")
    print("公平对比汇总")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'旧模型':<18} {'V4模型':<18} {'变化':<10}")
    print("-" * 70)
    for k in ['accuracy', 'macro_f1', 'weighted_f1']:
        osv = old[k]
        nv = v4[k]
        diff = (nv - osv) / osv * 100 if osv != 0 else 0
        os_str = f"{osv*100:.2f}%" if k == 'accuracy' else f"{osv:.4f}"
        nv_str = f"{nv*100:.2f}%" if k == 'accuracy' else f"{nv:.4f}"
        print(f"{k:<20} {os_str:<18} {nv_str:<18} {diff:+.1f}%")


if __name__ == "__main__":
    main()
