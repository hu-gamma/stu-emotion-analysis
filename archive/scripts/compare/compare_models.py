import json
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一对比两个模型的预测结果（如 RoBERTa-BiLSTM vs Qwen）"
    )
    parser.add_argument("--model_a_pred", type=str, required=True,
                        help="模型A预测结果 jsonl，每行包含 gold, pred")
    parser.add_argument("--model_b_pred", type=str, required=True,
                        help="模型B预测结果 jsonl")
    parser.add_argument("--model_a_name", type=str, default="RoBERTa-BiLSTM")
    parser.add_argument("--model_b_name", type=str, default="Qwen-LoRA")
    parser.add_argument("--output_report", type=str, default="comparison_report.txt")
    return parser.parse_args()


def load_preds(path):
    """加载预测结果，返回 golds, preds, latencies"""
    golds, preds, latencies = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            golds.append(item["gold"])
            preds.append(item["pred"])
            latencies.append(item.get("latency_ms", 0))
    return golds, preds, latencies


def compute_metrics(golds, preds, latencies):
    return {
        "accuracy": accuracy_score(golds, preds),
        "f1_macro": f1_score(golds, preds, average="macro"),
        "f1_weighted": f1_score(golds, preds, average="weighted"),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def main():
    args = parse_args()

    golds_a, preds_a, lat_a = load_preds(args.model_a_pred)
    golds_b, preds_b, lat_b = load_preds(args.model_b_pred)

    assert golds_a == golds_b, "两套预测结果的 gold label 不一致，请检查数据划分"

    m_a = compute_metrics(golds_a, preds_a, lat_a)
    m_b = compute_metrics(golds_b, preds_b, lat_b)

    lines = []
    lines.append("=" * 60)
    lines.append("模型对比报告")
    lines.append("=" * 60)
    lines.append(f"{'指标':<20} {args.model_a_name:<18} {args.model_b_name:<18}")
    lines.append("-" * 60)
    lines.append(f"{'Accuracy':<20} {m_a['accuracy']:<18.4f} {m_b['accuracy']:<18.4f}")
    lines.append(f"{'F1 (macro)':<20} {m_a['f1_macro']:<18.4f} {m_b['f1_macro']:<18.4f}")
    lines.append(f"{'F1 (weighted)':<20} {m_a['f1_weighted']:<18.4f} {m_b['f1_weighted']:<18.4f}")
    lines.append(f"{'Avg Latency (ms)':<20} {m_a['avg_latency_ms']:<18.2f} {m_b['avg_latency_ms']:<18.2f}")
    lines.append("=" * 60)

    lines.append(f"\n【{args.model_a_name} 详细报告】")
    lines.append(classification_report(golds_a, preds_a))

    lines.append(f"\n【{args.model_b_name} 详细报告】")
    lines.append(classification_report(golds_b, preds_b))

    # 找出两个模型预测不一致的样本
    diffs = []
    for i, (g, pa, pb) in enumerate(zip(golds_a, preds_a, preds_b)):
        if pa != pb:
            diffs.append((i, g, pa, pb))

    lines.append(f"\n预测不一致样本数: {len(diffs)} / {len(golds_a)}")
    for idx, g, pa, pb in diffs[:20]:  # 只展示前20个
        lines.append(f"  idx={idx} | gold={g} | {args.model_a_name}={pa} | {args.model_b_name}={pb}")

    report = "\n".join(lines)
    print(report)

    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存至: {args.output_report}")


if __name__ == "__main__":
    main()
