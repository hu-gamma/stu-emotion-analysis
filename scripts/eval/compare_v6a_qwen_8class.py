"""
V6a (RoBERTa-BiLSTM) vs Qwen2.5-0.5B-Instruct (LoRA 8类) 对比评估
"""
import json
import os
import sys
import time
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, '/mnt')
from models.model import RoBERTaBiLSTM
from models.model_v6b import RoBERTaBiLSTMDropoutEnhanced
from data.dataset_csv_loader_v4 import ECSADataset, get_label_name

LABELS_8CLASS = ['悲伤', '快乐', '愤怒', '焦虑', '尴尬', '厌恶', '惊讶', '好奇']
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS_8CLASS)}


def eval_v6a(model_path, test_csv, device):
    """评估 V6a 模型"""
    print(f"\n{'='*60}")
    print("评估: V6a (RoBERTa-BiLSTM, 8类)")
    print(f"{'='*60}")

    if 'v6b' in model_path.lower():
        model = RoBERTaBiLSTMDropoutEnhanced(num_classes=8)
        print("✓ 使用 V6b 增强 Dropout 模型")
    else:
        model = RoBERTaBiLSTM(num_classes=8, dropout_rate=0.1, lstm_hidden_size=256)
        print("✓ 使用标准 RoBERTa-BiLSTM 模型")
    cp = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    ds = ECSADataset(csv_path=test_csv, tokenizer=tokenizer, max_length=128, text_column='transformed_text')
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            _, pred = torch.max(outputs, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    total_time = time.time() - start_time

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return compute_metrics(y_true, y_pred, total_time, len(y_true), "V6a")


def extract_label(text, labels):
    text = text.strip()
    for label in labels:
        if label in text:
            return label
    first = text.split("\n")[0].strip()
    return first if first else "UNKNOWN"


def eval_qwen(model_name, lora_path, test_jsonl, device):
    """评估 Qwen 模型"""
    print(f"\n{'='*60}")
    print("评估: Qwen2.5-3B-Instruct + LoRA (8类)")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # 加载测试数据
    test_data = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    preds, golds = [], []
    latencies = []

    for item in tqdm(test_data, desc="Qwen Predicting"):
        text = item["text"]
        gold = item["label"]

        label_str = ", ".join(LABELS_8CLASS)
        instruction = (
            f"请判断以下文本的情感类别。可选类别：{label_str}。\n\n"
            f"文本：{text}\n\n类别："
        )
        messages = [
            {"role": "system", "content": "你是一个文本分类专家，只输出类别标签，不输出其他内容。"},
            {"role": "user", "content": instruction},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        pred = extract_label(response, LABELS_8CLASS)

        preds.append(LABEL_TO_ID.get(pred, -1))
        golds.append(LABEL_TO_ID.get(gold, -1))

    total_time = sum(latencies) / 1000
    y_true = np.array(golds)
    y_pred = np.array(preds)
    return compute_metrics(y_true, y_pred, total_time, len(y_true), "Qwen")


def compute_metrics(y_true, y_pred, total_time, num_samples, name):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    avg_latency = total_time / num_samples * 1000

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(8)), zero_division=0
    )

    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"平均延迟:    {avg_latency:.2f} ms/sample")
    print(f"总延迟:      {total_time:.2f} s")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, labels=list(range(8)),
          target_names=LABELS_8CLASS, digits=4, zero_division=0))

    return {
        'name': name,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'avg_latency_ms': avg_latency,
        'total_time_s': total_time,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def generate_report(v6a_result, qwen_result, output_path):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# V6a vs Qwen2.5-3B-Instruct (8类) 对比报告",
        f"",
        f"**生成时间**: {now}",
        f"",
        f"## 模型配置",
        f"",
        f"| 属性 | V6a | Qwen",
        f"|------|-----|------|",
        f"| 基座模型 | Chinese-RoBERTa-wwm-ext (110M) | Qwen2.5-3B-Instruct (3B) |",
        f"| 上层结构 | BiLSTM + Dense | LoRA (r=16) |",
        f"| 训练数据 | split_train.csv (8类, 去掉中性) | 同上 |",
        f"| 测试数据 | split_test.csv (9951条) | 同上 |",
        f"| Epochs | 10 (Focal Loss) | 3 |",
        f"",
        f"## 整体指标对比",
        f"",
        f"| 指标 | V6a | Qwen |",
        f"|------|-------|-------|",
    ]

    def fmt(v6a_v, qwen_v, higher_is_better=True):
        v6a_s = f"{v6a_v:.4f}"
        qwen_s = f"{qwen_v:.4f}"
        if higher_is_better:
            if v6a_v > qwen_v:
                return f"**{v6a_s}**", qwen_s
            elif qwen_v > v6a_v:
                return v6a_s, f"**{qwen_s}**"
        else:
            if v6a_v < qwen_v:
                return f"**{v6a_s}**", qwen_s
            elif qwen_v < v6a_v:
                return v6a_s, f"**{qwen_s}**"
        return v6a_s, qwen_s

    v6a_acc, qwen_acc = fmt(v6a_result['accuracy'], qwen_result['accuracy'])
    v6a_macro, qwen_macro = fmt(v6a_result['macro_f1'], qwen_result['macro_f1'])
    v6a_wf1, qwen_wf1 = fmt(v6a_result['weighted_f1'], qwen_result['weighted_f1'])
    v6a_lat, qwen_lat = fmt(v6a_result['avg_latency_ms'], qwen_result['avg_latency_ms'], higher_is_better=False)

    lines.append(f"| Accuracy | {v6a_acc} | {qwen_acc} |")
    lines.append(f"| Macro F1 | {v6a_macro} | {qwen_macro} |")
    lines.append(f"| Weighted F1 | {v6a_wf1} | {qwen_wf1} |")
    lines.append(f"| 平均延迟 (ms) | {v6a_lat} | {qwen_lat} |")
    lines.append("")

    lines.append("## 各类别 F1-Score 对比")
    lines.append("")
    lines.append("| 情感 | V6a | Qwen |")
    lines.append("|------|-------|-------|")
    for i, name in enumerate(LABELS_8CLASS):
        v6a_f1, qwen_f1 = fmt(v6a_result['f1'][i], qwen_result['f1'][i])
        lines.append(f"| {name} | {v6a_f1} | {qwen_f1} |")

    lines.append("")
    lines.append("## 各类别 Precision 对比")
    lines.append("")
    lines.append("| 情感 | V6a | Qwen |")
    lines.append("|------|-------|-------|")
    for i, name in enumerate(LABELS_8CLASS):
        v6a_p, qwen_p = fmt(v6a_result['precision'][i], qwen_result['precision'][i])
        lines.append(f"| {name} | {v6a_p} | {qwen_p} |")

    lines.append("")
    lines.append("## 各类别 Recall 对比")
    lines.append("")
    lines.append("| 情感 | V6a | Qwen |")
    lines.append("|------|-------|-------|")
    for i, name in enumerate(LABELS_8CLASS):
        v6a_r, qwen_r = fmt(v6a_result['recall'][i], qwen_result['recall'][i])
        lines.append(f"| {name} | {v6a_r} | {qwen_r} |")

    lines.append("")
    lines.append("## 结论")
    lines.append("")

    winner = "V6a" if v6a_result['macro_f1'] > qwen_result['macro_f1'] else "Qwen"
    lines.append(f"- **Macro F1 更高**: {winner} (V6a={v6a_result['macro_f1']:.4f}, Qwen={qwen_result['macro_f1']:.4f})")

    winner_acc = "V6a" if v6a_result['accuracy'] > qwen_result['accuracy'] else "Qwen"
    lines.append(f"- **Accuracy 更高**: {winner_acc} (V6a={v6a_result['accuracy']*100:.2f}%, Qwen={qwen_result['accuracy']*100:.2f}%)")

    winner_speed = "V6a" if v6a_result['avg_latency_ms'] < qwen_result['avg_latency_ms'] else "Qwen"
    lines.append(f"- **推理速度更快**: {winner_speed} (V6a={v6a_result['avg_latency_ms']:.2f}ms, Qwen={qwen_result['avg_latency_ms']:.2f}ms)")

    lines.append("")
    lines.append("### 小类表现")
    lines.append("")
    for i, name in enumerate(['尴尬', '惊讶', '厌恶', '焦虑']):
        cid = LABELS_8CLASS.index(name)
        v6a_f1 = v6a_result['f1'][cid]
        qwen_f1 = qwen_result['f1'][cid]
        winner_cls = "V6a" if v6a_f1 > qwen_f1 else "Qwen"
        lines.append(f"- **{name}**: {winner_cls} 更优 (V6a={v6a_f1:.4f}, Qwen={qwen_f1:.4f})")
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n对比报告已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V6b vs Qwen 对比评估')
    parser.add_argument('--model', '-m', type=str, default='/mnt/checkpoints/roberta_bilstm_v6b/best_model.pt',
                        help='RoBERTa-BiLSTM 模型路径 (支持 V6a/V6b)')
    parser.add_argument('--test-csv', '-t', type=str, default='/mnt/datasets/csv/split_test.csv',
                        help='测试集 CSV')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 评估 RoBERTa-BiLSTM (自动检测 V6b)
    v6a_result = eval_v6a(args.model, args.test_csv, device)

    # 评估 Qwen
    qwen_result = eval_qwen(
        'Qwen/Qwen2.5-3B-Instruct',
        '/mnt/checkpoints/qwen_lora_8class/best',
        '/mnt/data_processed/qwen_jsonl_8class/test.jsonl',
        device
    )

    # 生成报告
    os.makedirs('/mnt/results/compare', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = f'/mnt/results/compare/v6a_vs_qwen_8class_{timestamp}.md'
    generate_report(v6a_result, qwen_result, report_path)


if __name__ == "__main__":
    main()
