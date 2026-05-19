"""
LLM 数据增强脚本
将 ECSA 数据集中的 "满足" 类样本改写成指定的负面情绪样本，
并保存为与原数据集格式兼容的 CSV。

使用方式:
1. 填入下方的 API_KEY、BASE_URL、MODEL
2. 运行: python augment_llm.py
3. 生成的文件默认保存至 ./datasets/csv/augmented_negative_from_satisfied.csv

依赖安装:
    pip install openai pandas tqdm
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 用户配置区 - 请在此处填入你的 API 信息
# ---------------------------------------------------------------------------
API_KEY = os.getenv("KIMI_API_KEY", "")  # 从环境变量读取 Kimi API Key
BASE_URL = "https://api.moonshot.cn/v1"           # Kimi (Moonshot) API Base URL
MODEL = "moonshot-v1-8k"                          # 可选: moonshot-v1-8k / 32k / 128k

# 每个满足样本要生成的目标负面情绪列表
# 这些类别在原数据集中样本稀少，适合通过增强补充
TARGET_EMOTIONS = ["沮丧", "愤怒", "失望", "焦虑", "压力", "孤独"]

# 每个情绪从多少条原始样本中生成
# 受限于 Kimi 20 RPM 限速，全部 217 条约需 70 分钟。
# 此处设为 100 以在合理时间内完成增强（约 30 分钟）。
MAX_SOURCE_SAMPLES = 100

# 输出路径
OUTPUT_CSV = "./datasets/csv/augmented_negative_from_satisfied.csv"

# 并发与限速
# Kimi 个人用户限速：组织级 20 RPM。使用锁严格控速。
MAX_WORKERS = 3             # 并发线程数（仅控制连接数，实际请求被锁串行）
REQUEST_INTERVAL = 3.3      # 每两个请求间隔 3.3 秒 → 约 18 RPM，留有余量
MAX_RETRIES = 5             # 单条请求最大重试次数

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一位专业的中文情感分析数据增强助手。
你的任务是将用户提供的"满足/满意"情绪的句子，改写为表达特定负面情感的句子。
要求：
1. 保持原始场景、主语和事件背景不变，只改变情感倾向。
2. 改写后的句子要自然、通顺，符合中文口语或书面语习惯。
3. 必须准确表达指定的负面情绪，不能产生歧义。
4. 直接输出要求的 JSON 格式，不要有多余解释。"""

USER_PROMPT_TEMPLATE = """原始句子：{text}
目标情感：{emotion}

请返回以下 JSON 格式（不要包含 markdown 代码块标记）：
{{
  "rewritten_text": "改写后的句子",
  "emotion": "{emotion}"
}}"""


def get_openai_client():
    """初始化 OpenAI 兼容客户端"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请先安装 openai 库: pip install openai")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return client


def call_llm(client, text: str, emotion: str, retry: int = 0) -> Optional[Dict[str, str]]:
    """调用 LLM 生成单条增强样本，带重试机制"""
    user_prompt = USER_PROMPT_TEMPLATE.format(text=text, emotion=emotion)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()

        # 去除可能的 markdown 代码块
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        data = json.loads(content)
        return {
            "original_text": text,
            "transformed_text": data["rewritten_text"],
            "fine_grained_emotion": data["emotion"],
        }
    except Exception as e:
        error_msg = str(e)
        is_rate_limit = "429" in error_msg or "rate_limit" in error_msg.lower()
        if retry < MAX_RETRIES:
            # 遇到限速额外多等几秒
            sleep_time = (2 + retry * 2) if is_rate_limit else (1 * (retry + 1))
            time.sleep(sleep_time)
            return call_llm(client, text, emotion, retry + 1)
        else:
            print(f"\n[Error] 生成失败: text='{text[:30]}...' emotion={emotion}, 错误: {e}")
            return None


def augment_satisfied_samples(
    df: pd.DataFrame,
    target_emotions: List[str],
    max_source_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    对满足类样本调用 LLM 进行情感改写增强
    """
    client = get_openai_client()

    # 筛选满足类样本
    satisfied_df = df[df["fine_grained_emotion"] == "满足"].copy()
    if max_source_samples is not None:
        satisfied_df = satisfied_df.head(max_source_samples)

    source_texts = satisfied_df["original_text"].tolist()
    total_tasks = len(source_texts) * len(target_emotions)

    print(f"原始满足样本数: {len(source_texts)}")
    print(f"目标情绪: {target_emotions}")
    print(f"总任务数: {total_tasks}")
    print(f"并发数: {MAX_WORKERS}, 模型: {MODEL}\n")

    results: List[Dict[str, str]] = []
    rate_limit_lock = Lock()
    last_request_time = 0.0

    def task_wrapper(text: str, emotion: str):
        nonlocal last_request_time
        with rate_limit_lock:
            elapsed = time.time() - last_request_time
            if elapsed < REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL - elapsed)
            last_request_time = time.time()
        return call_llm(client, text, emotion)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(task_wrapper, text, emotion): (text, emotion)
            for text in source_texts
            for emotion in target_emotions
        }

        for future in tqdm(as_completed(futures), total=total_tasks, desc="LLM 增强中"):
            result = future.result()
            if result is not None:
                results.append(result)

    print(f"\n成功生成 {len(results)} / {total_tasks} 条样本")
    return pd.DataFrame(results)


def main():
    input_csv = "/mnt/datasets/csv/Dataset_ECSA_processed.csv"
    print(f"读取数据集: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"总样本数: {len(df)}")

    # 执行增强
    aug_df = augment_satisfied_samples(
        df,
        target_emotions=TARGET_EMOTIONS,
        max_source_samples=MAX_SOURCE_SAMPLES,
    )

    if aug_df.empty:
        print("没有生成任何新样本，请检查 API 配置和网络连接。")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 保存增强数据
    aug_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"增强数据已保存: {OUTPUT_CSV}")

    # 打印增强后的标签分布
    print("\n增强样本标签分布:")
    for emotion, count in aug_df["fine_grained_emotion"].value_counts().items():
        print(f"  {emotion}: {count}")

    # 合并原数据 + 增强数据并保存（可选）
    merged_path = OUTPUT_CSV.replace(".csv", "_merged.csv")
    merged_df = pd.concat([df, aug_df], ignore_index=True)
    merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"\n合并数据集已保存: {merged_path}")
    print(f"合并后总样本数: {len(merged_df)}")

    print("\n增强样本示例:")
    for _, row in aug_df.head(6).iterrows():
        print(f"[{row['fine_grained_emotion']}] {row['transformed_text']}")


if __name__ == "__main__":
    main()
