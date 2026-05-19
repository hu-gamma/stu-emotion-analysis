"""
V7 模型推理脚本（8类，去掉中性）
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/mnt')
from models.model_v7 import RoBERTaBiLSTMV7
from data.dataset_csv_loader_v4 import ECSADataset, get_num_classes, get_label_name


EMOTION_NAMES = {
    0: '悲伤', 1: '快乐', 2: '愤怒', 3: '焦虑',
    4: '尴尬', 5: '厌恶', 6: '惊讶', 7: '好奇'
}


class V7Predictor:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"使用设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.num_classes = get_num_classes()

        self.model = RoBERTaBiLSTMV7(num_classes=self.num_classes)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ V7 模型加载成功: {model_path}")

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

        pred_id = predicted.item()
        return {
            'text': text,
            'predicted_class': pred_id,
            'sentiment': EMOTION_NAMES[pred_id],
            'confidence': confidence.item(),
            'probabilities': {EMOTION_NAMES[i]: p.item() for i, p in enumerate(probs[0])}
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probs, dim=1)

        results = []
        for i, text in enumerate(texts):
            pred_id = predicted[i].item()
            results.append({
                'text': text,
                'predicted_class': pred_id,
                'sentiment': EMOTION_NAMES[pred_id],
                'confidence': confidences[i].item(),
                'probabilities': {EMOTION_NAMES[j]: probs[i][j].item() for j in range(self.num_classes)}
            })
        return results


def plot_bar(result: Dict, save_path: str):
    emotions = list(result['probabilities'].keys())
    probs = [result['probabilities'][e] * 100 for e in emotions]
    sorted_pairs = sorted(zip(emotions, probs), key=lambda x: x[1], reverse=True)
    emotions, probs = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#ff6b6b' if e == result['sentiment'] else '#4ecdc4' for e in emotions]
    ax.barh(emotions, probs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Probability (%)')
    ax.set_title(f"Predicted: {result['sentiment']} ({result['confidence']*100:.1f}%)")
    ax.set_xlim(0, 100)
    for bar, prob in zip(ax.patches, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{prob:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='V7 情感分析推理')
    parser.add_argument('--model', '-m', type=str,
                        default='/mnt/checkpoints/roberta_bilstm_v7_L4_d03_layerwise/best_model.pt',
                        help='V7 模型路径')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='输入文本文件，每行一个样本')
    parser.add_argument('--output', '-o', type=str, default='/mnt/results/v7_predict',
                        help='结果输出目录')
    parser.add_argument('--text', type=str, default=None,
                        help='单条文本直接预测')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    predictor = V7Predictor(args.model_path)

    if args.text:
        results = [predictor.predict(args.text)]
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        results = predictor.predict_batch(texts)
    else:
        texts = [
            "今天的印度菜太好吃了！",
            "食堂的服务态度太差了，我很生气。",
            "最近感觉很沮丧，什么都不想做。",
            "考试压力好大，我感到焦虑。",
            "这个消息太令人震惊了！"
        ]
        results = predictor.predict_batch(texts)

    for i, r in enumerate(results):
        print(f"\n【样本 {i+1}】{r['text']}")
        print(f"  → {r['sentiment']} (置信度: {r['confidence']*100:.1f}%)")
        top3 = sorted(r['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for j, (emotion, prob) in enumerate(top3, 1):
            marker = " ✓" if emotion == r['sentiment'] else ""
            print(f"    {j}. {emotion}: {prob*100:.1f}%{marker}")

        plot_bar(r, os.path.join(args.output, f"sample_{i+1:02d}_bar.png"))

    # 保存 JSON
    json_path = os.path.join(args.output, 'predictions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结果已保存: {json_path}")


if __name__ == "__main__":
    main()
