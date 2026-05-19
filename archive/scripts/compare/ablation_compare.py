"""
消融实验对比工具
支持多个模型预测结果的统一对比，输出综合报告和 CSV 表格。
用法：
    python ablation_compare.py \
        --pred_files "roberta_full.jsonl,roberta_no_dreaddit.jsonl" \
        --model_names "Full(OCEMOTION+dreaddit+ECSA),NoDreaddit(OCEMOTION+ECSA)" \
        --output_dir /mnt/results/compare
"""
import json
import argparse
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd


def load_preds(path):
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
        "f1_macro": f1_score(golds, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(golds, preds, average="weighted", zero_division=0),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_files", type=str, required=True,
                        help="逗号分隔的预测文件路径列表（在 results/compare 目录下）")
    parser.add_argument("--model_names", type=str, required=True,
                        help="逗号分隔的模型名称")
    parser.add_argument("--output_dir", type=str, default="/mnt/results/compare")
    parser.add_argument("--report_name", type=str, default="ablation_report.txt")
    parser.add_argument("--csv_name", type=str, default="ablation_summary.csv")
    args = parser.parse_args()

    pred_files = [p.strip() for p in args.pred_files.split(",")]
    model_names = [n.strip() for n in args.model_names.split(",")]
    assert len(pred_files) == len(model_names), "pred_files 和 model_names 数量必须一致"

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载所有预测
    all_golds = None
    all_results = []
    for pf, mn in zip(pred_files, model_names):
        path = os.path.join(args.output_dir, pf) if not os.path.isabs(pf) else pf
        golds, preds, lats = load_preds(path)
        if all_golds is None:
            all_golds = golds
        else:
            assert golds == all_golds, f"{mn} 的 gold label 与其他模型不一致"

        metrics = compute_metrics(golds, preds, lats)
        all_results.append((mn, metrics, golds, preds))

    # 生成报告
    lines = []
    lines.append("=" * 70)
    lines.append("消融实验对比报告")
    lines.append(f"测试样本数: {len(all_golds)}")
    lines.append("=" * 70)

    # 综合指标表格
    lines.append("\n【综合指标对比】\n")
    header = f"{'模型':<35} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<15} {'Avg Latency(ms)':<18}"
    lines.append(header)
    lines.append("-" * 70)
    for mn, metrics, _, _ in all_results:
        lines.append(
            f"{mn:<35} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f} "
            f"{metrics['f1_weighted']:<15.4f} {metrics['avg_latency_ms']:<18.2f}"
        )

    # 各类别 F1 对比
    lines.append("\n\n【各类别 F1-Score 对比】\n")
    labels = sorted(set(all_golds))
    header = f"{'类别':<10}"
    for mn, _, _, _ in all_results:
        header += f" {mn:<20}"
    lines.append(header)
    lines.append("-" * (10 + 22 * len(all_results)))

    # 计算每个模型每个类别的 F1
    per_class_f1 = defaultdict(dict)
    for mn, _, golds, preds in all_results:
        report = classification_report(golds, preds, output_dict=True, zero_division=0)
        for label in labels:
            f1 = report.get(label, {}).get("f1-score", 0.0)
            per_class_f1[label][mn] = f1

    for label in labels:
        row = f"{label:<10}"
        for mn, _, _, _ in all_results:
            row += f" {per_class_f1[label][mn]:<20.4f}"
        lines.append(row)

    # 预测不一致分析（两两对比）
    if len(all_results) == 2:
        mn_a, _, g_a, p_a = all_results[0]
        mn_b, _, g_b, p_b = all_results[1]
        diff_correct = 0
        same_correct = 0
        diff_total = 0
        a_better = 0
        b_better = 0
        for i, (gold, pa, pb) in enumerate(zip(all_golds, p_a, p_b)):
            if pa != pb:
                diff_total += 1
                if pa == gold:
                    a_better += 1
                if pb == gold:
                    b_better += 1
            else:
                if pa == gold:
                    same_correct += 1

        lines.append(f"\n\n【{mn_a} vs {mn_b} 预测差异分析】\n")
        lines.append(f"预测不一致样本: {diff_total} / {len(all_golds)} ({diff_total/len(all_golds)*100:.1f}%)")
        lines.append(f"不一致样本中 {mn_a} 正确: {a_better} 条")
        lines.append(f"不一致样本中 {mn_b} 正确: {b_better} 条")
        lines.append(f"预测一致且都正确: {same_correct} 条")
        lines.append(f"预测一致但都错误: {len(all_golds) - diff_total - same_correct} 条")

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(args.output_dir, args.report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存: {report_path}")

    # 保存 CSV 表格（方便后续累积对比）
    csv_data = []
    for mn, metrics, _, _ in all_results:
        row = {"model": mn}
        row.update(metrics)
        for label in labels:
            row[f"f1_{label}"] = per_class_f1[label][mn]
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(args.output_dir, args.csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
