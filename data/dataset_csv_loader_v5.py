"""
数据集加载器 V5 - 8类情感分类（去掉中性）
支持单轮和多轮对话数据
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# 8类情感标签
EMOTION_LABELS = ['悲伤', '快乐', '愤怒', '焦虑', '厌恶', '惊讶', '好奇']
EMOTION_MAP = {label: i for i, label in enumerate(EMOTION_LABELS)}


def get_num_classes():
    return len(EMOTION_LABELS)


def get_label_name(idx):
    if 0 <= idx < len(EMOTION_LABELS):
        return EMOTION_LABELS[idx]
    return '未知'


class ECSADataset(Dataset):
    """
    单轮情感分析数据集
    """
    def __init__(self, csv_path, tokenizer, max_length=128, text_column='transformed_text', label_column='fine_grained_emotion'):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(csv_path)
        self.samples = []

        for _, row in df.iterrows():
            text = str(row[text_column]) if pd.notna(row[text_column]) else ''
            emotion = str(row[label_column]) if pd.notna(row[label_column]) else ''

            if not text or emotion not in EMOTION_MAP:
                continue

            label = EMOTION_MAP[emotion]

            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
            )

            self.samples.append({
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long),
                'text': text,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'input_ids': s['input_ids'],
            'attention_mask': s['attention_mask'],
            'labels': s['label'],
        }


class MultiTurnDataset(Dataset):
    """
    多轮对话数据集
    将 prev_text + curr_text 拼接为 [CLS] prev [SEP] curr [SEP]
    """
    def __init__(self, df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = []
        for _, row in df.iterrows():
            emotion = row['emotion']
            if emotion not in EMOTION_MAP:
                continue
            label = EMOTION_MAP[emotion]

            prev = str(row['prev_text']) if pd.notna(row['prev_text']) and row['prev_text'] else ''
            curr = str(row['curr_text'])

            # 拼接: [CLS] prev [SEP] curr [SEP]
            if prev:
                input_text = prev + tokenizer.sep_token + curr
            else:
                input_text = curr

            encoding = tokenizer(
                input_text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
            )

            self.samples.append({
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long),
                'has_context': 1 if prev else 0,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'input_ids': s['input_ids'],
            'attention_mask': s['attention_mask'],
            'labels': s['label'],
        }
