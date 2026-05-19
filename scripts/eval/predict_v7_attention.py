"""
V7-Attention 模型推理脚本（8类，去掉中性）
支持注意力权重可视化
"""
import os
import sys
import json
import argparse
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/mnt')
from models.model_v7_attention import RoBERTaBiLSTMV7Attention
from data.dataset_csv_loader_v4 import get_num_classes


EMOTION_NAMES = {
    0: '悲伤', 1: '快乐', 2: '愤怒', 3: '焦虑',
    4: '尴尬', 5: '厌恶', 6: '惊讶', 7: '好奇'
}


class V7AttentionPredictor:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"使用设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.num_classes = get_num_classes()

        self.model = RoBERTaBiLSTMV7Attention(num_classes=self.num_classes)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ V7-Attention 模型加载成功: {model_path}")

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
    def predict_with_attention(self, text: str) -> Dict:
        """预测并返回注意力权重，用于可视化"""
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

        # 预测
        outputs = self.model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        pred_id = predicted.item()

        # 提取注意力权重
        attn_weights = self.model.get_attention_weights(input_ids, attention_mask)
        # 只保留有效长度（非 padding 部分）
        valid_len = attention_mask.sum(dim=1).item()
        attn_valid = attn_weights[0, :valid_len].cpu().numpy()

        # 获取 token 列表（有效部分）
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :valid_len].cpu().tolist())

        return {
            'text': text,
            'predicted_class': pred_id,
            'sentiment': EMOTION_NAMES[pred_id],
            'confidence': confidence.item(),
            'probabilities': {EMOTION_NAMES[i]: probs[0][i].item() for i in range(self.num_classes)},
            'tokens': tokens,
            'attention_weights': attn_valid.tolist(),
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


def plot_probability_bar(result: Dict, save_path: str):
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


def plot_attention(result: Dict, save_path: str):
    """绘制 BiLSTM-Attention 权重热力图"""
    tokens = result['tokens']
    weights = np.array(result['attention_weights'])

    # 过滤特殊标记以提升可读性（可选）
    display_tokens = []
    display_weights = []
    for t, w in zip(tokens, weights):
        if t in ['[PAD]']:
            break
        display_tokens.append(t)
        display_weights.append(w)

    if len(display_tokens) == 0:
        display_tokens = tokens
        display_weights = weights

    fig, ax = plt.subplots(figsize=(max(10, len(display_tokens) * 0.5), 4))

    # 绘制水平热力条
    cmap = plt.cm.Reds
    norm = plt.Normalize(vmin=0, vmax=max(display_weights) * 1.2)
    for i, (token, weight) in enumerate(zip(display_tokens, display_weights)):
        color = cmap(norm(weight))
        ax.barh(0, 1, left=i, height=0.8, color=color, edgecolor='white', linewidth=0.5)
        ax.text(i + 0.5, 0, token, ha='center', va='center', fontsize=8, rotation=45)

    ax.set_xlim(0, len(display_tokens))
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"BiLSTM-Attention Weights\nPredicted: {result['sentiment']} ({result['confidence']*100:.1f}%)", fontsize=12)

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Attention Weight', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  注意力权重图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='V7-Attention 情感分析推理')
    parser.add_argument('--model', '-m', type=str,
                        default='/mnt/checkpoints/roberta_bilstm_v7attn_L4_d03_layerwise/best_model.pt',
                        help='V7-Attention 模型路径')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='输入文本文件，每行一个样本')
    parser.add_argument('--output', '-o', type=str, default='/mnt/results/v7attn_predict',
                        help='结果输出目录')
    parser.add_argument('--text', type=str, default=None,
                        help='单条文本直接预测')
    parser.add_argument('--no-attention-viz', action='store_true',
                        help='不生成注意力权重可视化')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    predictor = V7AttentionPredictor(args.model)

    if args.text:
        texts = [args.text]
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [
            "今天的印度菜太好吃了！",
            "食堂的服务态度太差了，我很生气。",
            "最近感觉很沮丧，什么都不想做。",
            "考试压力好大，我感到焦虑。",
            "这个消息太令人震惊了！",
            "他居然在公共场合做出这种事，真让人厌恶。",
            "第一次上台演讲，太尴尬了。",
            "这本书的结局让我很好奇后续发展。"
        ]

    all_results = []
    for i, text in enumerate(texts):
        if not args.no_attention_viz:
            result = predictor.predict_with_attention(text)
        else:
            result = predictor.predict(text)

        all_results.append(result)

        print(f"\n【样本 {i+1}】{result['text']}")
        print(f"  → {result['sentiment']} (置信度: {result['confidence']*100:.1f}%)")
        top3 = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for j, (emotion, prob) in enumerate(top3, 1):
            marker = " ✓" if emotion == result['sentiment'] else ""
            print(f"    {j}. {emotion}: {prob*100:.1f}%{marker}")

        plot_probability_bar(result, os.path.join(args.output, f"sample_{i+1:02d}_bar.png"))

        if not args.no_attention_viz and 'attention_weights' in result:
            plot_attention(result, os.path.join(args.output, f"sample_{i+1:02d}_attention.png"))

    # 保存 JSON
    json_path = os.path.join(args.output, 'predictions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结果已保存: {json_path}")


if __name__ == "__main__":
    main()
