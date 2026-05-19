"""
GoEmotions 数据处理全流程脚本

阶段1: 合并现有数据集标签（压力->焦虑，孤独->悲伤）
阶段2: 从 GoEmotions 抽样非中性样本（14类体系）
阶段3: 调用 Kimi API 批量翻译为中文
阶段4: 合并输出统一格式 CSV

使用方式:
    python process_goemotions.py

输出:
    - datasets/csv/OCEMOTION_processed_v2.csv    (标签合并后)
    - datasets/csv/dreaddit_train_processed_v2.csv (标签合并后)
    - datasets/csv/dreaddit_test_processed_v2.csv  (标签合并后)
    - datasets/csv/Dataset_ECSA_processed_v2.csv   (标签合并后)
    - datasets/csv/goemotions_translated.csv       (翻译后的GoEmotions)
    - datasets/csv/all_merged_v2.csv               (全部合并)
"""
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# ============================================================================
# 配置区
# ============================================================================
API_KEY = "sk-RB2oozKyx5bKJgVxZsx7W1RZcB0cBQ7iCRSR626oPQJqfvTi"
BASE_URL = "https://api.moonshot.cn/v1"
MODEL = "moonshot-v1-8k"

# 限速配置: 500 RPM, 并发10
MAX_WORKERS = 10
REQUEST_INTERVAL = 60.0 / 500  # 0.12 秒
MAX_RETRIES = 5

# GoEmotions 标签定义
GO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# GoEmotions -> 14类中文映射
GO_TO_CN = {
    "admiration":    "感激",
    "amusement":     "快乐",
    "anger":         "愤怒",
    "annoyance":     "愤怒",
    "approval":      "满足",
    "caring":        "满足",
    "confusion":     "好奇",
    "curiosity":     "好奇",
    "desire":        "期待",
    "disappointment":"失望",
    "disapproval":   "厌恶",
    "disgust":       "厌恶",
    "embarrassment": "尴尬",
    "excitement":    "快乐",
    "fear":          "焦虑",
    "gratitude":     "感激",
    "grief":         "悲伤",
    "joy":           "快乐",
    "love":          "快乐",
    "nervousness":   "焦虑",
    "optimism":      "期待",
    "pride":         "自豪",
    "realization":   "好奇",
    "relief":        "满足",
    "remorse":       "悲伤",
    "sadness":       "悲伤",
    "surprise":      "惊讶",
    "neutral":       "中性",  # 将被丢弃
}

# 14类最终标签列表
FINAL_LABELS = [
    "悲伤", "快乐", "愤怒", "满足", "焦虑", "失望", "厌恶",
    "惊讶", "好奇", "感激", "尴尬", "期待", "自豪", "中性"
]

# 稀缺度排序（越前面越稀缺，优先保留）
SCARCITY_ORDER = [
    "自豪", "感激", "期待", "好奇", "惊讶", "失望",
    "厌恶", "尴尬", "焦虑", "悲伤", "快乐", "愤怒", "满足", "中性"
]

# GoEmotions 各标签抽取上限 (去中性后)
SAMPLING_LIMITS = {
    "自豪":   99999,  # 全部
    "感激":    3000,
    "好奇":    2000,
    "期待":    1500,
    "厌恶":    1500,
    "失望":     800,
    "惊讶":     600,
    "焦虑":     500,
    "尴尬":     300,
    "悲伤":     500,
    "快乐":     500,
    "愤怒":     500,
    "满足":     300,
    "中性":       0,  # 丢弃
}

# 路径配置
DATASETS_DIR = "/mnt/datasets/csv"
EXTERNAL_DIR = "/mnt/external_datasets"
OUTPUT_DIR = DATASETS_DIR


# ============================================================================
# 阶段1: 合并现有数据集标签
# ============================================================================
def merge_existing_labels():
    """将现有数据集中的'压力'->'焦虑', '孤独'->'悲伤'"""
    print("=" * 60)
    print("阶段1: 合并现有数据集标签")
    print("=" * 60)

    files = [
        "OCEMOTION_processed.csv",
        "dreaddit_train_processed.csv",
        "dreaddit_test_processed.csv",
        "Dataset_ECSA_processed.csv",
        "augmented_negative_from_satisfied_merged.csv",
    ]

    merge_map = {"压力": "焦虑", "孤独": "悲伤"}
    total_merged = 0

    for fname in files:
        fpath = os.path.join(DATASETS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [跳过] {fname} 不存在")
            continue

        df = pd.read_csv(fpath)
        if "fine_grained_emotion" not in df.columns:
            print(f"  [跳过] {fname} 无 fine_grained_emotion 列")
            continue

        before = df["fine_grained_emotion"].value_counts().to_dict()
        df["fine_grained_emotion"] = df["fine_grained_emotion"].replace(merge_map)
        after = df["fine_grained_emotion"].value_counts().to_dict()

        merged = sum(1 for k in merge_map if k in before)
        total_merged += merged

        # 保存为新版本
        out_name = fname.replace(".csv", "_v2.csv")
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [完成] {fname} -> {out_name}")
        for old_lab, new_lab in merge_map.items():
            if old_lab in before:
                print(f"         '{old_lab}'({before.get(old_lab,0)}) -> '{new_lab}'")

    print(f"\n标签合并完成，共处理 {total_merged} 个文件")
    return True


# ============================================================================
# 阶段2: GoEmotions 抽样
# ============================================================================
def load_and_sample_goemotions():
    """从 GoEmotions 中按策略抽样非中性样本"""
    print("\n" + "=" * 60)
    print("阶段2: GoEmotions 抽样")
    print("=" * 60)

    all_samples = []
    for split in ["train", "validation", "test"]:
        path = os.path.join(EXTERNAL_DIR, f"goemotions_{split}.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                go_names = [GO_LABELS[l] for l in d["labels"]]
                # 丢弃纯中性
                if set(go_names) == {"neutral"}:
                    continue
                # 映射到中文标签
                cn_labels = list(dict.fromkeys([GO_TO_CN[n] for n in go_names if n in GO_TO_CN]))
                # 丢弃只剩中性的
                cn_labels = [c for c in cn_labels if c != "中性"]
                if not cn_labels:
                    continue
                # 按稀缺度选主标签
                primary = None
                for emo in SCARCITY_ORDER:
                    if emo in cn_labels:
                        primary = emo
                        break
                if primary is None:
                    continue
                all_samples.append({
                    "text": d["text"],
                    "primary": primary,
                    "cn_labels": cn_labels,
                    "go_labels": go_names,
                })

    # 按主标签分组并抽样
    grouped = {}
    for s in all_samples:
        grouped.setdefault(s["primary"], []).append(s)

    sampled = []
    stats = []
    for emo in SCARCITY_ORDER:
        if emo == "中性":
            continue
        pool = grouped.get(emo, [])
        limit = SAMPLING_LIMITS.get(emo, 0)
        if limit <= 0:
            continue
        n_take = min(len(pool), limit)
        chosen = random.sample(pool, n_take) if n_take < len(pool) else pool
        sampled.extend(chosen)
        stats.append((emo, len(pool), n_take))

    print(f"总候选样本: {len(all_samples)}")
    print(f"抽样后样本: {len(sampled)}")
    print(f"\n{'标签':<8} {'候选':>8} {'抽取':>8}")
    print("-" * 28)
    for emo, pool, take in stats:
        print(f"{emo:<8} {pool:>8} {take:>8}")

    return sampled


# ============================================================================
# 阶段3: Kimi API 翻译
# ============================================================================
def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def translate_text(client, text: str, retry: int = 0) -> Optional[str]:
    """调用 Kimi API 将英文翻译为中文"""
    system_prompt = (
        "你是一位专业的翻译助手。请将下面的英文句子翻译成自然、通顺的中文，"
        "保持原文的情感色彩和语气。只输出翻译后的中文句子，不要添加任何解释。"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        is_rate_limit = "429" in error_msg or "rate_limit" in error_msg.lower()
        if retry < MAX_RETRIES:
            sleep_time = (2 + retry * 2) if is_rate_limit else (1 * (retry + 1))
            time.sleep(sleep_time)
            return translate_text(client, text, retry + 1)
        else:
            return None


def batch_translate(samples: List[Dict]) -> pd.DataFrame:
    """批量翻译样本"""
    print("\n" + "=" * 60)
    print("阶段3: Kimi API 批量翻译")
    print("=" * 60)
    print(f"总样本数: {len(samples)}")
    print(f"并发数: {MAX_WORKERS}, 请求间隔: {REQUEST_INTERVAL:.3f}s (约 {60/REQUEST_INTERVAL:.0f} RPM)")
    print(f"预计耗时: ~{len(samples) * REQUEST_INTERVAL / 60:.1f} 分钟\n")

    client = get_openai_client()
    rate_limit_lock = Lock()
    last_request_time = 0.0

    results = []

    def task_wrapper(sample: Dict):
        nonlocal last_request_time
        with rate_limit_lock:
            elapsed = time.time() - last_request_time
            if elapsed < REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL - elapsed)
            last_request_time = time.time()
        translated = translate_text(client, sample["text"])
        return {
            "original_text": sample["text"],
            "transformed_text": translated,
            "fine_grained_emotion": sample["primary"],
            "go_labels": ",".join(sample["go_labels"]),
            "cn_labels": ",".join(sample["cn_labels"]),
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(task_wrapper, s): s for s in samples}
        for future in tqdm(as_completed(futures), total=len(samples), desc="翻译中"):
            result = future.result()
            if result["transformed_text"] is not None:
                results.append(result)

    df = pd.DataFrame(results)
    print(f"\n翻译成功: {len(df)} / {len(samples)}")
    print(f"翻译失败: {len(samples) - len(df)}")

    # 去掉辅助列，保持与现有CSV一致
    df_out = df[["original_text", "transformed_text", "fine_grained_emotion"]].copy()
    return df_out


# ============================================================================
# 阶段4: 合并输出
# ============================================================================
def merge_all(df_go: pd.DataFrame):
    """将翻译后的GoEmotions与现有v2数据合并"""
    print("\n" + "=" * 60)
    print("阶段4: 合并所有数据")
    print("=" * 60)

    # 读取现有v2数据（取transformed_text和fine_grained_emotion）
    existing_files = [
        "OCEMOTION_processed_v2.csv",
        "dreaddit_train_processed_v2.csv",
        "dreaddit_test_processed_v2.csv",
        "Dataset_ECSA_processed_v2.csv",
        "augmented_negative_from_satisfied_merged_v2.csv",
    ]

    all_dfs = [df_go]
    for fname in existing_files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            # 尝试不带_v2的版本
            fpath = os.path.join(DATASETS_DIR, fname.replace("_v2.csv", ".csv"))
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
        else:
            df = pd.read_csv(fpath)

        if "transformed_text" in df.columns and "fine_grained_emotion" in df.columns:
            keep = df[["transformed_text", "fine_grained_emotion"]].copy()
            keep.columns = ["transformed_text", "fine_grained_emotion"]
            # 合并压力->焦虑, 孤独->悲伤
            keep["fine_grained_emotion"] = keep["fine_grained_emotion"].replace({"压力": "焦虑", "孤独": "悲伤"})
            all_dfs.append(keep)
            print(f"  [加载] {fname}: {len(keep)} 条")

    # 合并
    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged[["transformed_text", "fine_grained_emotion"]]

    # 过滤只保留14类
    valid_labels = set(FINAL_LABELS)
    merged = merged[merged["fine_grained_emotion"].isin(valid_labels)]

    # 保存
    out_path = os.path.join(OUTPUT_DIR, "all_merged_v2.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n合并完成: {out_path}")
    print(f"总样本数: {len(merged)}")
    print(f"\n标签分布:")
    print("-" * 30)
    for emo in FINAL_LABELS:
        count = (merged["fine_grained_emotion"] == emo).sum()
        pct = count / len(merged) * 100
        print(f"  {emo:<8} {count:>8} ({pct:>5.2f}%)")

    # 同时保存GoEmotions单独文件
    go_path = os.path.join(OUTPUT_DIR, "goemotions_translated.csv")
    df_go.to_csv(go_path, index=False, encoding="utf-8-sig")
    print(f"\nGoEmotions单独保存: {go_path} ({len(df_go)} 条)")

    return merged


# ============================================================================
# 主函数
# ============================================================================
def main():
    random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 阶段1
    merge_existing_labels()

    # 阶段2
    samples = load_and_sample_goemotions()
    if not samples:
        print("没有可抽样的GoEmotions样本，退出")
        return

    # 阶段3
    df_go = batch_translate(samples)
    if df_go.empty:
        print("翻译失败，没有生成数据")
        return

    # 阶段4
    merge_all(df_go)

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
