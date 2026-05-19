"""
ECSA Dataset 推理脚本
支持12类中文情感分析预测，包含可视化结果保存
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

# 可视化相关
import matplotlib
matplotlib.use('Agg')  # 无界面环境使用
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体 - 尝试多种方式
import matplotlib.font_manager as fm

# 查找可用中文字体
chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
available_fonts = [f.name for f in fm.fontManager.ttflist]
selected_font = None
for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
else:
    # 使用备用方案：使用默认字体但情感用拼音表示
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

# 情感标签拼音映射（用于图表显示）
EMOTION_PINYIN = {
    '悲伤': 'Sadness',
    '快乐': 'Happiness',
    '愤怒': 'Anger',
    '满足': 'Satisfaction',
    '焦虑': 'Anxiety',
    '中性': 'Neutral',
    '失望': 'Disappointment',
    '尴尬': 'Embarrassment',
    '厌恶': 'Disgust',
    '惊讶': 'Surprise',
    '期待': 'Expectation',
    '好奇': 'Curiosity',
    '感激': 'Gratitude',
    '孤独': 'Loneliness',
    '压力': 'Stress',
    '自豪': 'Pride'
}

from models.model import RoBERTaBiLSTM
from models.model_v6b import RoBERTaBiLSTMDropoutEnhanced
from data.dataset_csv_loader_v2 import ChineseTextPreprocessor, get_num_classes, get_label_name, ECSADataset


class ECSAPredictor:
    """
    ECSA情感分析预测器 (12分类)
    """

    def __init__(
        self,
        model_path: str = './checkpoints_csv/best_model.pt',
        max_length: int = 128,
        device: str = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        self.preprocessor = ChineseTextPreprocessor()
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.max_length = max_length
        self.num_classes = get_num_classes()

        if 'v6b' in model_path.lower():
            self.model = RoBERTaBiLSTMDropoutEnhanced(num_classes=self.num_classes)
            print("✓ 使用 V6b 增强 Dropout 模型")
        else:
            self.model = RoBERTaBiLSTM(
                num_classes=self.num_classes,
                dropout_rate=0.1,
                lstm_hidden_size=256
            )
            print("✓ 使用标准 RoBERTa-BiLSTM 模型")

        self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str):
        """加载模型权重"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 模型加载成功: {model_path}")
            if 'epoch' in checkpoint:
                print(f"  训练轮数: {checkpoint['epoch']}")
            if 'val_acc' in checkpoint:
                print(f"  验证准确率: {checkpoint['val_acc']*100:.2f}%")
        except FileNotFoundError:
            print(f"⚠️ 警告: 未找到模型文件 {model_path}")
        except Exception as e:
            print(f"⚠️ 警告: 加载模型时出错: {e}")

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """预测单条文本的情感"""
        processed_text = self.preprocessor.preprocess(text)

        encoding = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask)
        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)
        top3_probs, top3_classes = torch.topk(probabilities, k=3, dim=1)

        return {
            'text': text,
            'processed_text': processed_text,
            'predicted_class': predicted_class.item(),
            'sentiment': get_label_name(predicted_class.item()),
            'confidence': confidence.item(),
            'top3': [
                {
                    'emotion': get_label_name(top3_classes[0][i].item()),
                    'probability': top3_probs[0][i].item()
                }
                for i in range(3)
            ],
            'probabilities': {
                get_label_name(i): prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[dict]:
        """批量预测"""
        processed_texts = [self.preprocessor.preprocess(t) for t in texts]

        encodings = self.tokenizer(
            processed_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask)
        probabilities = F.softmax(outputs, dim=1)

        confidences, predicted_classes = torch.max(probabilities, dim=1)

        results = []
        for i, text in enumerate(texts):
            pred_class = predicted_classes[i].item()
            results.append({
                'text': text,
                'predicted_class': pred_class,
                'sentiment': get_label_name(pred_class),
                'confidence': confidences[i].item(),
                'probabilities': {
                    get_label_name(j): prob.item()
                    for j, prob in enumerate(probabilities[i])
                }
            })

        return results

    @torch.no_grad()
    def evaluate(self, texts: List[str], labels: List[int]) -> dict:
        """
        在带标签数据上评估模型，返回预测结果和评估指标
        """
        processed_texts = [self.preprocessor.preprocess(t) for t in texts]

        encodings = self.tokenizer(
            processed_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted_classes = torch.max(probabilities, dim=1)

        y_pred = predicted_classes.cpu().numpy()
        y_prob = probabilities.cpu().numpy()
        y_true = np.array(labels)

        from sklearn.metrics import classification_report, f1_score, accuracy_score

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        results = []
        for i, text in enumerate(texts):
            pred_class = predicted_classes[i].item()
            results.append({
                'text': text,
                'predicted_class': pred_class,
                'sentiment': get_label_name(pred_class),
                'confidence': confidences[i].item(),
                'probabilities': {
                    get_label_name(j): prob.item()
                    for j, prob in enumerate(probabilities[i])
                }
            })

        return {
            'results': results,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'classification_report': classification_report(
                y_true, y_pred,
                target_names=[get_label_name(i) for i in range(self.num_classes)],
                digits=4,
                zero_division=0
            )
        }


class PredictionVisualizer:
    """预测结果可视化器"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _get_emotion_label(self, emotion_cn: str) -> str:
        """获取英文情感标签"""
        return EMOTION_PINYIN.get(emotion_cn, emotion_cn)

    def plot_probability_bar(self, result: dict, filename: str, sample_id: int = None):
        """绘制概率分布柱状图"""
        emotions = list(result['probabilities'].keys())
        probs = [result['probabilities'][e] * 100 for e in emotions]

        # 按概率排序
        sorted_pairs = sorted(zip(emotions, probs), key=lambda x: x[1], reverse=True)
        emotions, probs = zip(*sorted_pairs)

        # 转换为英文标签
        emotions_en = [self._get_emotion_label(e) for e in emotions]
        predicted_en = self._get_emotion_label(result['sentiment'])

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#ff6b6b' if e == result['sentiment'] else '#4ecdc4' for e in emotions]
        bars = ax.barh(emotions_en, probs, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Probability (%)', fontsize=12)
        title = f'Sample {sample_id}' if sample_id else 'Emotion Prediction'
        ax.set_title(f'Emotion Prediction Distribution\n{title}', fontsize=14)
        ax.set_xlim(0, 100)

        # 添加数值标签
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{prob:.1f}%', va='center', fontsize=10)

        ax.axvline(x=result['confidence']*100, color='red', linestyle='--',
                  label=f'Predicted: {predicted_en} ({result["confidence"]*100:.1f}%)')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_top3_pie(self, result: dict, filename: str, sample_id: int = None):
        """绘制Top3概率饼图"""
        fig, ax = plt.subplots(figsize=(8, 8))

        top3 = result['top3']
        labels = [f"{self._get_emotion_label(item['emotion'])}\n{item['probability']*100:.1f}%" for item in top3]
        sizes = [item['probability'] for item in top3]
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        explode = (0.05, 0, 0)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        title = f'Sample {sample_id}' if sample_id else 'Top 3 Predictions'
        ax.set_title(f'Top 3 Emotion Predictions\n{title}', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_batch_comparison(self, results: List[dict], filename: str):
        """批量预测对比图"""
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 6))

        if n == 1:
            axes = [axes]

        for idx, (ax, result) in enumerate(zip(axes, results)):
            emotions = list(result['probabilities'].keys())
            probs = [result['probabilities'][e] * 100 for e in emotions]

            sorted_pairs = sorted(zip(emotions, probs), key=lambda x: x[1], reverse=True)[:5]
            emotions, probs = zip(*sorted_pairs)
            emotions_en = [self._get_emotion_label(e) for e in emotions]

            colors = ['#ff6b6b' if e == result['sentiment'] else '#95a5a6' for e in emotions]
            ax.barh(emotions_en, probs, color=colors)
            ax.set_xlabel('Probability (%)')
            ax.set_title(f'Sample {idx+1}\n→ {self._get_emotion_label(result["sentiment"])}', fontsize=10)
            ax.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_confidence_distribution(self, results: List[dict], filename: str):
        """绘制置信度分布图"""
        confidences = [r['confidence'] * 100 for r in results]
        sentiments = [r['sentiment'] for r in results]
        sentiments_en = [self._get_emotion_label(s) for s in sentiments]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 柱状图
        x = range(len(confidences))
        colors = ['#2ecc71' if c > 50 else '#e74c3c' if c < 30 else '#f39c12' for c in confidences]
        ax1.bar(x, confidences, color=colors, edgecolor='black')
        ax1.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='High (>50%)')
        ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Low (<30%)')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Confidence (%)')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.legend()
        ax1.set_ylim(0, 100)

        # 添加标签
        for i, (conf, sent) in enumerate(zip(confidences, sentiments_en)):
            ax1.text(i, conf + 2, f'{sent[:6]}\n{conf:.0f}%', ha='center', fontsize=8)

        # 饼图 - 情感分布
        sentiment_counts = {}
        for s in sentiments_en:
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

        ax2.pie(sentiment_counts.values(), labels=sentiment_counts.keys(),
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Predicted Sentiment Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_probability_heatmap(self, results: List[dict], filename: str = 'probability_heatmap.png'):
        """
        绘制概率分布热力矩阵
        展示所有样本的12类情感概率分布 (样本×类别矩阵)
        """
        import seaborn as sns

        # 构建概率矩阵
        emotion_order = ['悲伤', '快乐', '愤怒', '满足', '焦虑', '中性',
                        '失望', '尴尬', '厌恶', '惊讶', '期待', '好奇',
                        '感激', '孤独', '压力', '自豪']
        emotion_labels_en = [self._get_emotion_label(e) for e in emotion_order]

        n_samples = len(results)
        prob_matrix = np.zeros((n_samples, self.num_classes))

        for i, result in enumerate(results):
            for j, emotion in enumerate(emotion_order):
                prob_matrix[i, j] = result['probabilities'][emotion] * 100

        # 创建图形
        fig, ax = plt.subplots(figsize=(14, max(6, n_samples * 0.8)))

        # 绘制热力图
        sns.heatmap(prob_matrix,
                    xticklabels=emotion_labels_en,
                    yticklabels=[f"Sample {i+1}" for i in range(n_samples)],
                    annot=True,
                    fmt='.1f',
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Probability (%)'},
                    ax=ax,
                    linewidths=0.5,
                    linecolor='white')

        ax.set_xlabel('Emotion Classes', fontsize=12)
        ax.set_ylabel('Test Samples', fontsize=12)
        ax.set_title('Emotion Probability Heatmap (Sample × Class)', fontsize=14, pad=20)

        # 标注每行的最大概率
        for i in range(n_samples):
            max_idx = np.argmax(prob_matrix[i])
            ax.add_patch(plt.Rectangle((max_idx, i), 1, 1, fill=False,
                                       edgecolor='blue', linewidth=3))

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 概率热力矩阵已保存: {filename}")

    def plot_confusion_matrix_heatmap(self, y_true: List[int], y_pred: List[int],
                                      filename: str = 'confusion_matrix.png'):
        """
        绘制混淆矩阵热力图 (需要真实标签)
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        emotion_order = ['悲伤', '快乐', '愤怒', '满足', '焦虑', '中性',
                        '失望', '尴尬', '厌恶', '惊讶', '期待', '好奇',
                        '感激', '孤独', '压力', '自豪']
        emotion_labels_en = [self._get_emotion_label(e) for e in emotion_order]

        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(cm,
                    xticklabels=emotion_labels_en,
                    yticklabels=emotion_labels_en,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    cbar_kws={'label': 'Count'},
                    ax=ax,
                    linewidths=0.5)

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix Heatmap', fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 混淆矩阵热力图已保存: {filename}")

    def plot_roc_curves(self, y_true: List[int], y_prob: np.ndarray, filename: str = 'roc_curve.png'):
        """
        绘制多分类ROC曲线 (One-vs-Rest)
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_classes = y_prob.shape[1]
        y_true_arr = np.array(y_true)
        y_true_bin = label_binarize(y_true_arr, classes=list(range(n_classes)))

        # 计算每个类别的 ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC
        fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        # 绘制
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr['micro'], tpr['micro'],
                label=f"Micro-average ROC (AUC = {roc_auc['micro']:.2f})",
                color='deeppink', linestyle=':', linewidth=2)
        ax.plot(fpr['macro'], tpr['macro'],
                label=f"Macro-average ROC (AUC = {roc_auc['macro']:.2f})",
                color='navy', linestyle=':', linewidth=2)

        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            emotion_en = EMOTION_PINYIN.get(get_label_name(i), get_label_name(i))
            ax.plot(fpr[i], tpr[i], color=color, lw=1.5,
                    label=f"{emotion_en} (AUC = {roc_auc[i]:.2f})")

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Multi-class ROC Curve (One-vs-Rest)', fontsize=14)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ ROC 曲线已保存: {filename}")

    def save_results_json(self, results: List[dict], filename: str = 'results.json'):
        """保存结果为JSON"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存: {filepath}")

    def save_summary_txt(self, results: List[dict], filename: str = 'summary.txt'):
        """保存文本摘要"""
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ECSA 情感分析预测结果汇总\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"\n【样本 {i}】\n")
                f.write(f"原文: {result['text']}\n")
                f.write(f"预处理: {result['processed_text']}\n")
                f.write(f"预测情感: {result['sentiment']} (ID: {result['predicted_class']})\n")
                f.write(f"置信度: {result['confidence']*100:.2f}%\n")
                f.write("Top-3:\n")
                for j, item in enumerate(result['top3'], 1):
                    marker = " ✓" if j == 1 else ""
                    f.write(f"  {j}. {item['emotion']}: {item['probability']*100:.2f}%{marker}\n")
                f.write("-"*40 + "\n")

            # 统计信息
            sentiments = [r['sentiment'] for r in results]
            from collections import Counter
            counts = Counter(sentiments)

            f.write("\n" + "="*60 + "\n")
            f.write("统计汇总\n")
            f.write("="*60 + "\n")
            for sent, count in counts.most_common():
                f.write(f"  {sent}: {count} ({count/len(results)*100:.1f}%)\n")

            avg_conf = sum(r['confidence'] for r in results) / len(results)
            f.write(f"\n平均置信度: {avg_conf*100:.2f}%\n")

        print(f"✓ 摘要已保存: {filepath}")


def create_result_subfolder(base_dir: str = './result') -> str:
    """创建带时间戳的子文件夹"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    subfolder = os.path.join(base_dir, f"test_{timestamp}")
    os.makedirs(subfolder, exist_ok=True)
    return subfolder


def load_test_samples(file_path: str = './datasets/test/test_samples.txt') -> List[str]:
    """
    从外部文件加载测试样例

    支持格式:
    - .txt: 每行一个样本
    - .json: [{"text": "..."}, ...] 或 ["...", ...]
    - .csv: 包含 'text' 列

    Args:
        file_path: 测试样例文件路径

    Returns:
        测试文本列表
    """
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        print("使用默认测试样例...")
        return get_default_samples()

    ext = os.path.splitext(file_path)[1].lower()
    samples = []

    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    if len(data) > 0:
                        if isinstance(data[0], dict):
                            samples = [item.get('text', item.get('content', '')) for item in data]
                        else:
                            samples = data
                elif isinstance(data, dict):
                    samples = data.get('samples', data.get('texts', []))

        elif ext == '.csv':
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get('text', row.get('content', row.get('original_text', '')))
                    if text:
                        samples.append(text)

        else:  # .txt 或其他格式
            with open(file_path, 'r', encoding='utf-8') as f:
                samples = [line.strip() for line in f if line.strip()]

        if not samples:
            print(f"⚠️ 未从 {file_path} 读取到有效样本")
            return get_default_samples()

        print(f"✓ 从 {file_path} 加载了 {len(samples)} 个测试样例")
        return samples

    except Exception as e:
        print(f"⚠️ 读取文件出错: {e}")
        return get_default_samples()


def load_test_samples_with_labels(file_path: str = './datasets/test/test_samples.txt') -> Tuple[List[str], List[int]]:
    """
    从外部文件加载测试样例及其标签（如果存在）

    支持格式:
    - .txt: 每行一个样本（无标签）
    - .json: [{"text": "...", "label": "..."}, ...]
    - .csv: 包含 'text' 列和 'label'/'fine_grained_emotion'/'emotion' 列

    Returns:
        (测试文本列表, 标签列表或None)
    """
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        print("使用默认测试样例...")
        return get_default_samples(), None

    ext = os.path.splitext(file_path)[1].lower()
    texts = []
    labels = []
    has_labels = False

    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    for item in data:
                        text = item.get('text', item.get('content', ''))
                        if text:
                            texts.append(text)
                            label = item.get('label', item.get('emotion', item.get('fine_grained_emotion', None)))
                            if label is not None:
                                has_labels = True
                            labels.append(label)

        elif ext == '.csv':
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                label_cols = ['fine_grained_emotion', 'label', 'emotion', 'category']
                label_col = None
                # 尝试找到标签列
                if reader.fieldnames:
                    for col in label_cols:
                        if col in reader.fieldnames:
                            label_col = col
                            break
                for row in reader:
                    text = row.get('text', row.get('content', row.get('original_text', '')))
                    if text:
                        texts.append(text)
                        if label_col and row.get(label_col):
                            has_labels = True
                            labels.append(row[label_col])
                        else:
                            labels.append(None)

        else:  # .txt 或其他格式，认为无标签
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            return texts, None

        if not texts:
            print(f"⚠️ 未从 {file_path} 读取到有效样本")
            return get_default_samples(), None

        # 转换标签为数字 ID
        if has_labels:
            numeric_labels = []
            for lbl in labels:
                if lbl is None:
                    numeric_labels.append(-1)
                elif isinstance(lbl, int) or (isinstance(lbl, str) and lbl.isdigit()):
                    numeric_labels.append(int(lbl))
                elif isinstance(lbl, str):
                    # 使用 ECSADataset 的映射
                    if lbl in ECSADataset.EMOTION_MAP:
                        numeric_labels.append(ECSADataset.EMOTION_MAP[lbl])
                    else:
                        print(f"⚠️ 未知标签 '{lbl}'，将忽略标签列")
                        return texts, None
                else:
                    numeric_labels.append(-1)
            print(f"✓ 从 {file_path} 加载了 {len(texts)} 个测试样例（含标签）")
            return texts, numeric_labels

        print(f"✓ 从 {file_path} 加载了 {len(texts)} 个测试样例（无标签）")
        return texts, None

    except Exception as e:
        print(f"⚠️ 读取文件出错: {e}")
        return get_default_samples(), None


def get_default_samples() -> List[str]:
    """获取默认测试样例"""
    return [
        "今天的印度菜太好吃了！",
        "食堂的服务态度太差了，我很生气。",
        "今天天气一般，没什么特别的感觉。",
        "工作人员服务很有礼貌，我很感激。",
        "那个汉堡太棒了！我非常欣喜！",
        "最近感觉很沮丧，什么都不想做。",
        "考试压力好大，我感到焦虑。"
    ]


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='ECSA 12类情感分析预测器')
    parser.add_argument('--input', '-i', type=str, default='./datasets/test/test_samples.txt',
                        help='测试样例文件路径 (.txt/.json/.csv)，默认: ./datasets/test/test_samples.txt')
    parser.add_argument('--output', '-o', type=str, default='./result',
                        help='结果输出目录，默认: ./result')
    parser.add_argument('--model', '-m', type=str, default='./checkpoints_csv/best_model.pt',
                        help='模型文件路径，默认: ./checkpoints_csv/best_model.pt')

    args = parser.parse_args()

    print("="*60)
    print("ECSA 12类情感分析预测器 (带可视化)")
    print("="*60)

    # 创建结果子文件夹
    result_dir = create_result_subfolder(args.output)
    print(f"\n结果将保存至: {result_dir}")

    # 初始化预测器和可视化器
    predictor = ECSAPredictor(
        model_path=args.model,
        max_length=128
    )
    visualizer = PredictionVisualizer(result_dir)

    # 从外部文件加载测试样本（尝试读取标签）
    test_texts, test_labels = load_test_samples_with_labels(args.input)

    # 如果有标签，进行评估并计算 F1 Score / ROC
    if test_labels is not None:
        print("\n" + "="*60)
        print("带标签评估模式")
        print("="*60)

        eval_info = predictor.evaluate(test_texts, test_labels)
        print(f"\n准确率: {eval_info['accuracy']*100:.2f}%")
        print(f"Macro F1: {eval_info['macro_f1']:.4f}")
        print(f"Weighted F1: {eval_info['weighted_f1']:.4f}")
        print("\n详细分类报告:")
        print(eval_info['classification_report'])

        # 绘制混淆矩阵和 ROC 曲线
        visualizer.plot_confusion_matrix_heatmap(
            eval_info['y_true'].tolist(),
            eval_info['y_pred'].tolist(),
            filename='confusion_matrix.png'
        )
        visualizer.plot_roc_curves(
            eval_info['y_true'].tolist(),
            eval_info['y_prob'],
            filename='roc_curve.png'
        )

        all_results = eval_info['results']
    else:
        # 单条预测并可视化
        print("\n" + "="*60)
        print("单条预测示例")
        print("="*60)

        all_results = []
        for i, text in enumerate(test_texts):
            result = predictor.predict(text)
            all_results.append(result)

            # 打印结果
            print(f"\n【样本 {i+1}】")
            print(f"原文: {text}")
            print(f"预测: {result['sentiment']} | 置信度: {result['confidence']*100:.2f}%")

            # 生成可视化
            visualizer.plot_probability_bar(result, f"sample_{i+1:02d}_bar.png", sample_id=i+1)
            visualizer.plot_top3_pie(result, f"sample_{i+1:02d}_pie.png", sample_id=i+1)

        # 批量对比图
        print("\n" + "="*60)
        print("批量预测可视化")
        print("="*60)

        visualizer.plot_batch_comparison(all_results, "batch_comparison.png")
        visualizer.plot_confidence_distribution(all_results, "confidence_distribution.png")

        # 生成热力矩阵
        print("\n" + "="*60)
        print("生成热力矩阵")
        print("="*60)
        visualizer.plot_probability_heatmap(all_results, "probability_heatmap.png")

    # 保存结果
    visualizer.save_results_json(all_results)
    if test_labels is None:
        visualizer.save_summary_txt(all_results)

    # 列出保存的文件
    print("\n" + "="*60)
    print("保存的文件:")
    print("="*60)
    for f in sorted(os.listdir(result_dir)):
        size = os.path.getsize(os.path.join(result_dir, f))
        print(f"  {f} ({size:,} bytes)")

    print(f"\n✓ 所有结果已保存到: {result_dir}")


if __name__ == "__main__":
    main()
