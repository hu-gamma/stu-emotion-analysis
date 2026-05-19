import json
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA 微调后模型目录")
    parser.add_argument("--test_file", type=str, required=True, help="测试集 jsonl，需与 RoBERTa 完全一致")
    parser.add_argument("--labels", type=str, required=True, help="逗号分隔的标签列表")
    parser.add_argument("--output_pred", type=str, default="qwen_predictions.jsonl",
                        help="预测结果保存路径，用于后续统一对比")
    parser.add_argument("--batch_size", type=int, default=1, help="目前只支持 bs=1，用于精确测延迟")
    return parser.parse_args()


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_label(text, labels):
    """从模型输出中提取标签"""
    text = text.strip()
    for label in labels:
        if label in text:
            return label
    # 兜底：返回最可能的前几个字
    first = text.split("\n")[0].strip()
    return first if first else "UNKNOWN"


def predict(text, model, tokenizer, labels, device):
    label_str = ", ".join(labels)
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
    latency = (time.time() - start) * 1000  # ms

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    pred = extract_label(response, labels)
    return pred, latency


def main():
    args = parse_args()
    labels = [l.strip() for l in args.labels.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    device = next(model.parameters()).device

    test_data = load_jsonl(args.test_file)

    preds, golds, latencies = [], [], []
    results = []

    for item in tqdm(test_data, desc="Predicting"):
        pred, latency = predict(item["text"], model, tokenizer, labels, device)
        gold = item["label"]
        preds.append(pred)
        golds.append(gold)
        latencies.append(latency)
        results.append({
            "text": item["text"],
            "gold": gold,
            "pred": pred,
            "latency_ms": round(latency, 2),
        })

    # 保存预测结果
    with open(args.output_pred, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = accuracy_score(golds, preds)
    f1_macro = f1_score(golds, preds, average="macro")
    f1_weighted = f1_score(golds, preds, average="weighted")
    avg_latency = sum(latencies) / len(latencies)

    print("=" * 50)
    print(f"模型: {args.model_name} + LoRA ({args.lora_path})")
    print(f"测试样本数: {len(golds)}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"F1 (macro):    {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"平均延迟:      {avg_latency:.2f} ms/sample")
    print("=" * 50)
    print(classification_report(golds, preds))


if __name__ == "__main__":
    main()
