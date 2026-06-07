"""
RoBERTa-BiLSTM V8 MultiTurn - 多轮对话情感分析模型
继承V8，添加多轮对话预测能力
"""
import re
import torch
import torch.nn.functional as F

from .model_v8 import RoBERTaBiLSTMV8


class RoBERTaBiLSTMV8MultiTurn(RoBERTaBiLSTMV8):
    """
    多轮对话情感分析模型
    将长文本按句子分割，逐句预测，利用上下文窗口
    """

    def split_sentences(self, text):
        """将文本分割为句子"""
        # 按中文标点分割
        sentences = re.split(r'[。！？\n；]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [text]

    @torch.no_grad()
    def predict_multiturn(self, text, device, context_window=1):
        """
        多轮对话情感预测
        Args:
            text: 输入文本（可能包含多个句子）
            device: torch device
            context_window: 上下文窗口大小（考虑前面几句话）
        Returns:
            list of dict, 每个句子一个预测结果
        """
        self.eval()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

        sentences = self.split_sentences(text)
        results = []

        for i, sentence in enumerate(sentences):
            # 构建上下文
            prev_context = ""
            if context_window > 0 and i > 0:
                start_idx = max(0, i - context_window)
                prev_sentences = sentences[start_idx:i]
                prev_context = tokenizer.sep_token.join(prev_sentences)

            # 拼接: [CLS] prev [SEP] curr [SEP]
            if prev_context:
                input_text = prev_context + tokenizer.sep_token + sentence
            else:
                input_text = sentence

            encoding = tokenizer(
                input_text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            outputs = self(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

            emotion = self.EMOTION_LABELS[predicted.item()]

            results.append({
                'sentence': sentence,
                'emotion': emotion,
                'confidence': confidence.item(),
                'probabilities': {
                    label: probs[0][j].item()
                    for j, label in enumerate(self.EMOTION_LABELS)
                },
                'prev_context': prev_context,
            })

        return results

    @torch.no_grad()
    def predict_multiturn_html(self, text, device, context_window=1):
        """
        多轮对话预测，返回HTML可视化结果
        """
        results = self.predict_multiturn(text, device, context_window)

        # 构建HTML
        colors = {
            '悲伤': '#5B8FF9',
            '快乐': '#F6BD16',
            '愤怒': '#E8684A',
            '焦虑': '#9266F9',
            '厌恶': '#6DC8EC',
            '惊讶': '#FF9D4D',
            '好奇': '#5D7092',
        }

        html_parts = []
        html_parts.append('<div style="font-family: sans-serif; padding: 20px;">')
        html_parts.append('<h3>多轮对话情感分析结果</h3>')

        emotion_sequence = []
        for r in results:
            emotion_sequence.append(r['emotion'])
            color = colors.get(r['emotion'], '#999')
            bar_width = int(r['confidence'] * 100)

            html_parts.append(f'<div style="margin: 10px 0; padding: 10px; border-left: 4px solid {color}; background: #f8f9fa;">')
            html_parts.append(f'<div style="color: {color}; font-weight: bold;">{r["emotion"]} ({r["confidence"]:.1%})</div>')
            html_parts.append(f'<div style="margin: 5px 0;">{r["sentence"]}</div>')
            html_parts.append(f'<div style="background: #e9ecef; height: 8px; border-radius: 4px;">')
            html_parts.append(f'<div style="background: {color}; width: {bar_width}%; height: 100%; border-radius: 4px;"></div>')
            html_parts.append('</div>')
            html_parts.append('</div>')

        html_parts.append(f'<div style="margin-top: 15px; color: #666;">')
        html_parts.append(f'情感序列: {" → ".join(emotion_sequence)}')
        html_parts.append('</div>')
        html_parts.append('</div>')

        html = '\n'.join(html_parts)
        return html, results
