"""
学生情感分析对话系统 - Web交互界面
使用Gradio构建，支持单轮/多轮情感分析 + 情绪趋势可视化
"""
import os
import re
import json
import random
import tempfile
import io
from datetime import datetime
from collections import defaultdict, Counter

import gradio as gr
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties

# 强制刷新字体缓存以识别新安装的中文字体
fm._load_fontmanager(try_read_cache=False)

# 找到中文字体文件路径
CHINESE_FONT_PATH = None
for font in fm.fontManager.ttflist:
    if 'WenQuanYi' in font.name and 'Zen' in font.name:
        CHINESE_FONT_PATH = font.fname
        break
    elif 'Noto Sans CJK' in font.name:
        CHINESE_FONT_PATH = font.fname
        break

if CHINESE_FONT_PATH:
    print(f"[Font] Found Chinese font: {CHINESE_FONT_PATH}")
    CHINESE_FP = FontProperties(fname=CHINESE_FONT_PATH)
else:
    print("[Font] WARNING: No Chinese font found!")
    CHINESE_FP = None

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 情感分析引擎（演示版本 - 基于规则）
# 后续可替换为真实模型推理
# ============================================================================

EMOTION_LABELS = ['悲伤', '快乐', '愤怒', '焦虑', '厌恶', '惊讶', '好奇']
EMOTION_COLORS = {
    '悲伤': '#5B8FF9',
    '快乐': '#F6BD16',
    '愤怒': '#E8684A',
    '焦虑': '#9266F9',
    '厌恶': '#6DC8EC',
    '惊讶': '#FF9D4D',
    '好奇': '#5D7092',
}
EMOTION_ICONS = {
    '悲伤': '😢',
    '快乐': '😊',
    '愤怒': '😠',
    '焦虑': '😰',
    '厌恶': '🤢',
    '惊讶': '😲',
    '好奇': '🤔',
}

# 关键词映射（用于演示）
KEYWORD_MAP = {
    '悲伤': ['难过', '伤心', '哭', '失落', '沮丧', '痛苦', '绝望', '孤单', '遗憾', '悲伤', '消沉', ' melancholy', 'depressed', 'sad'],
    '快乐': ['开心', '高兴', '快乐', '幸福', '棒', '赞', '喜欢', '兴奋', '激动', '满足', '愉悦', '欣喜', 'awesome', 'great', 'happy'],
    '愤怒': ['生气', '愤怒', '火大', '讨厌', '恨', '恼火', '气愤', '暴怒', '不爽', '烦死了', '可恶', 'angry', 'mad', 'furious'],
    '焦虑': ['紧张', '焦虑', '担心', '害怕', '压力', '不安', '忐忑', '慌', '忧虑', '恐慌', '焦急', 'anxious', 'worried', 'nervous'],
    '厌恶': ['恶心', '反感', '嫌弃', '讨厌', '受不了', '厌恶', '鄙视', '唾弃', '憎恶', '作呕', 'disgust', 'gross'],
    '惊讶': ['惊讶', '意外', '震惊', '居然', '没想到', '天哪', '哇', '吃惊', '诧异', '惊呆', 'surprised', 'shocked', 'amazing'],
    '好奇': ['好奇', '想知道', '为什么', '怎么回事', '疑问', '疑惑', '感兴趣', '探索', '纳闷', '不解', 'curious', 'wonder'],
}


def rule_based_predict(text):
    """基于关键词规则的情感预测（演示用）"""
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        # 考虑否定词翻转
        if emotion == '快乐' and any(n in text_lower for n in ['不', '没', '别']):
            for neg in ['不', '没', '别']:
                for kw in keywords[:5]:
                    if neg + kw in text_lower or kw + neg in text_lower:
                        score -= 2
        scores[emotion] = max(0, score)

    # 如果没有匹配到任何关键词，随机给一个低置信度的结果
    if sum(scores.values()) == 0:
        emotion = random.choice(EMOTION_LABELS)
        confidence = random.uniform(0.15, 0.35)
        probs = {e: random.uniform(0.05, 0.15) for e in EMOTION_LABELS}
        probs[emotion] = confidence
        # 归一化
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        confidence = probs[emotion]
    else:
        # 归一化
        total = sum(scores.values())
        probs = {k: v / total for k, v in scores.items()}
        emotion = max(probs, key=probs.get)
        confidence = probs[emotion]

    return {
        'emotion': emotion,
        'confidence': min(confidence, 0.95),
        'probabilities': probs,
    }


def split_sentences(text):
    """将文本分割为句子"""
    sentences = re.split(r'[。！？\n；]+', text)
    return [s.strip() for s in sentences if s.strip()]


def multiturn_predict(text, history, context_window=1):
    """多轮对话情感预测"""
    sentences = split_sentences(text)
    results = []

    # 合并历史对话，强制全部转为字符串，防止 Gradio 传递嵌套结构
    all_utterances = [str(h) for h in history if h is not None] + [str(s) for s in sentences if s]

    for i, sentence in enumerate(sentences):
        global_idx = len(history) + i

        # 获取上下文
        start_idx = max(0, global_idx - context_window)
        context = ' '.join(all_utterances[start_idx:global_idx])

        # 结合上下文进行预测（简化版：拼接上下文和当前句子）
        if context:
            combined = context + ' ' + sentence
        else:
            combined = sentence

        pred = rule_based_predict(combined)
        results.append({
            'sentence': sentence,
            'emotion': pred['emotion'],
            'confidence': pred['confidence'],
            'probabilities': pred['probabilities'],
        })

    return results


# ============================================================================
# 全局对话历史存储
# ============================================================================
class ConversationStore:
    def __init__(self):
        self.sessions = defaultdict(list)
        self.session_counter = 0

    def new_session(self):
        self.session_counter += 1
        return f"session_{self.session_counter}"

    def add(self, session_id, user_text, single_result, multi_results):
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_text': user_text,
            'single_result': single_result,
            'multi_results': multi_results,
        }
        self.sessions[session_id].append(entry)

    def get_history(self, session_id):
        return self.sessions.get(session_id, [])

    def get_emotion_timeline(self, session_id):
        """获取情绪时间线，用于趋势图"""
        history = self.get_history(session_id)
        timeline = []
        for entry in history:
            # 使用单轮结果作为总体情绪
            single = entry['single_result']
            timeline.append({
                'timestamp': entry['timestamp'],
                'emotion': single['emotion'],
                'confidence': single['confidence'],
                'text': entry['user_text'][:30] + '...' if len(entry['user_text']) > 30 else entry['user_text'],
            })
        return timeline


store = ConversationStore()


# ============================================================================
# 可视化函数
# ============================================================================
def fig_to_pil(fig):
    """将 matplotlib Figure 转换为 PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def create_emotion_trend_chart(timeline):
    """创建情绪趋势图，返回 PIL Image"""
    if not timeline:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, '暂无对话数据', ha='center', va='center', fontsize=16, color='gray',
                fontproperties=CHINESE_FP)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig_to_pil(fig)

    emotions = [t['emotion'] for t in timeline]
    confidences = [t['confidence'] for t in timeline]
    indices = list(range(len(timeline)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})

    # --- 上图：情绪变化折线图 ---
    emotion_to_y = {e: i for i, e in enumerate(EMOTION_LABELS)}
    y_values = [emotion_to_y[e] for e in emotions]

    # 绘制折线
    for i in range(len(indices) - 1):
        color = EMOTION_COLORS[emotions[i]]
        ax1.plot(indices[i:i+2], y_values[i:i+2], color=color, linewidth=2.5, alpha=0.7)

    # 绘制数据点
    for i, (idx, y, conf, emotion) in enumerate(zip(indices, y_values, confidences, emotions)):
        color = EMOTION_COLORS[emotion]
        size = 100 + conf * 300
        ax1.scatter(idx, y, s=size, c=color, edgecolors='white', linewidths=2, zorder=5)
        ax1.annotate(f"{emotion}", (idx, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10, fontproperties=CHINESE_FP)

    ax1.set_yticks(range(len(EMOTION_LABELS)))
    ax1.set_yticklabels(EMOTION_LABELS, fontproperties=CHINESE_FP)
    ax1.set_xlabel('对话轮次', fontsize=12, fontproperties=CHINESE_FP)
    ax1.set_ylabel('情感类别', fontsize=12, fontproperties=CHINESE_FP)
    ax1.set_title('情绪变化趋势图', fontsize=16, fontweight='bold', fontproperties=CHINESE_FP)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xlim(-0.5, len(indices) - 0.5)

    # --- 下图：各类别出现频次 ---
    emotion_counts = Counter(emotions)
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    colors = [EMOTION_COLORS[l] for l in labels]

    sorted_pairs = sorted(zip(values, labels, colors), reverse=True)
    values_sorted = [v for v, _, _ in sorted_pairs]
    labels_sorted = [l for _, l, _ in sorted_pairs]
    colors_sorted = [c for _, _, c in sorted_pairs]

    bars = ax2.barh(labels_sorted, values_sorted,
                    color=colors_sorted, edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('出现次数', fontsize=12, fontproperties=CHINESE_FP)
    ax2.set_title('各情感类别出现频次', fontsize=14, fontweight='bold', fontproperties=CHINESE_FP)
    ax2.invert_yaxis()

    for bar, val in zip(bars, values_sorted):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig_to_pil(fig)


def create_probability_radar(probabilities):
    """创建概率雷达图，返回 PIL Image"""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    categories = EMOTION_LABELS
    values = [probabilities.get(c, 0) for c in categories]
    values += values[:1]  # 闭合

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax.plot(angles, values, 'o-', linewidth=2, color='#5B8FF9')
    ax.fill(angles, values, alpha=0.25, color='#5B8FF9')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10,
                       fontproperties=CHINESE_FP)
    ax.set_ylim(0, 1)
    ax.set_title('情感概率分布', fontsize=14, fontweight='bold', pad=20, fontproperties=CHINESE_FP)
    ax.grid(True)

    plt.tight_layout()
    return fig_to_pil(fig)


# ============================================================================
# Gradio 界面回调函数
# ============================================================================
def process_message(message, chat_history, session_id, mode):
    """处理用户消息"""
    if not message.strip():
        return chat_history, session_id, None, None, ""

    # 调试: 打印 chat_history 结构
    if chat_history:
        for i, turn in enumerate(chat_history[:2]):

    # 单轮分析
    single_result = rule_based_predict(message)

    # 多轮分析
    # 从 chat_history 提取之前的用户消息作为历史
    # 兼容 Gradio 6.x 多种数据格式：字典 / ChatMessage / 元组
    history_texts = []
    for turn in chat_history:
        if turn is None:
            continue
        # 格式1: dict {'role': 'user', 'content': '...'}
        if isinstance(turn, dict):
            if turn.get('role') == 'user':
                content = turn.get('content', '')
                if isinstance(content, str):
                    history_texts.append(content)
                elif isinstance(content, list):
                    history_texts.append(str(content))
        # 格式2: ChatMessage 对象 (Gradio 6.x)
        elif hasattr(turn, 'role') and hasattr(turn, 'content'):
            if turn.role == 'user':
                content = turn.content
                if isinstance(content, str):
                    history_texts.append(content)
                elif isinstance(content, list):
                    history_texts.append(str(content))
        # 格式3: tuple/list (user_msg, bot_msg)
        elif isinstance(turn, (list, tuple)) and len(turn) >= 1:
            if turn[0]:
                history_texts.append(str(turn[0]))

    # 确保所有元素都是字符串
    history_texts = [str(h) for h in history_texts]
    multi_results = multiturn_predict(message, history_texts, context_window=1)

    # 保存到对话存储
    store.add(session_id, message, single_result, multi_results)

    # 构建系统回复
    emotion = single_result['emotion']
    icon = EMOTION_ICONS[emotion]
    confidence = single_result['confidence']
    color = EMOTION_COLORS[emotion]

    # 多轮详细结果
    multi_detail = ""
    if len(multi_results) > 1:
        multi_detail = "\n\n**分句分析：**\n"
        for i, r in enumerate(multi_results, 1):
            multi_detail += f"\n{i}. {r['sentence'][:40]}{'...' if len(r['sentence']) > 40 else ''}\n"
            multi_detail += f"   → {EMOTION_ICONS[r['emotion']]} **{r['emotion']}** ({r['confidence']:.1%})\n"

    response = f"""### {icon} 情感分析结果

**主导情感：** <span style="color:{color};font-size:1.2em;font-weight:bold;">{emotion}</span>

**置信度：** {confidence:.1%}

**7类概率分布：**
"""
    for emo in EMOTION_LABELS:
        prob = single_result['probabilities'].get(emo, 0)
        bar = '█' * int(prob * 20)
        response += f"\n- {EMOTION_ICONS[emo]} {emo}: {bar} {prob:.1%}"

    response += multi_detail

    # 更新聊天历史
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    # 生成可视化图表
    timeline = store.get_emotion_timeline(session_id)
    trend_fig = create_emotion_trend_chart(timeline)
    radar_fig = create_probability_radar(single_result['probabilities'])

    return chat_history, session_id, trend_fig, radar_fig, ""


def new_conversation():
    """开始新对话"""
    session_id = store.new_session()
    return [], session_id, None, None, f"新对话已创建 (ID: {session_id})"


def export_conversation(session_id):
    """导出对话记录"""
    history = store.get_history(session_id)
    if not history:
        return None, "暂无对话记录可导出"

    # 创建临时文件
    fd, path = tempfile.mkstemp(suffix='.json', prefix=f'conversation_{session_id}_')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return path, f"对话记录已导出: {path}"


# ============================================================================
# Gradio 界面构建
# ============================================================================
CSS = """
.main-container {
    max-width: 1400px;
    margin: 0 auto;
}
.chatbot {
    height: 500px;
}
.title {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #7f8c8d;
    margin-bottom: 20px;
}
.emotion-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    color: white;
    font-weight: bold;
    margin: 2px;
}
"""

with gr.Blocks(title="学生情感分析对话系统") as demo:

    gr.Markdown("""
    <div class="title">
        <h1>🎓 学生情感分析对话系统</h1>
    </div>
    <div class="subtitle">
        <p>基于 RoBERTa-BiLSTM 的多轮对话情感分析 | 实时情绪追踪与可视化</p>
    </div>
    """)

    session_state = gr.State(value=store.new_session())

    with gr.Row():
        # --- 左侧：对话区域 ---
        with gr.Column(scale=1):
            gr.Markdown("### 💬 对话区域")

            chatbot = gr.Chatbot(
                label="",
                height=500,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="请输入你想说的话...",
                    show_label=False,
                    scale=4,
                    lines=1
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            with gr.Row():
                new_chat_btn = gr.Button("🆕 新对话", variant="secondary")
                export_btn = gr.Button("📥 导出记录", variant="secondary")

            status_text = gr.Textbox(label="状态", interactive=False, value="")

        # --- 右侧：分析结果区域 ---
        with gr.Column(scale=1):
            gr.Markdown("### 📊 情绪分析可视化")

            with gr.Tab("趋势图"):
                trend_plot = gr.Image(label="情绪变化趋势", type="pil")

            with gr.Tab("概率雷达"):
                radar_plot = gr.Image(label="情感概率分布", type="pil")

            with gr.Tab("对话统计"):
                stats_text = gr.Markdown("""
                **使用说明：**
                - 在左侧输入文本，系统会实时分析情感
                - 支持单句和多句输入
                - 情绪趋势图会随着对话轮次自动更新
                - 点击"导出记录"可保存对话历史
                """)

    # --- 事件绑定 ---
    send_btn.click(
        fn=process_message,
        inputs=[msg_input, chatbot, session_state, gr.State("multiturn")],
        outputs=[chatbot, session_state, trend_plot, radar_plot, msg_input]
    )

    msg_input.submit(
        fn=process_message,
        inputs=[msg_input, chatbot, session_state, gr.State("multiturn")],
        outputs=[chatbot, session_state, trend_plot, radar_plot, msg_input]
    )

    new_chat_btn.click(
        fn=new_conversation,
        inputs=[],
        outputs=[chatbot, session_state, trend_plot, radar_plot, status_text]
    )

    export_btn.click(
        fn=export_conversation,
        inputs=[session_state],
        outputs=[gr.File(label="下载"), status_text]
    )

    # 页脚
    gr.Markdown("""
    <div style="text-align: center; margin-top: 20px; color: #95a5a6; font-size: 0.9em;">
        <p>当前为演示版本，使用基于规则的情感分析引擎 | 后续可替换为 RoBERTa-BiLSTM 真实模型</p>
    </div>
    """)


if __name__ == '__main__':
    demo.launch(
        server_name='0.0.0.0',
        server_port=3389,
        share=False,
        show_error=True,
        css=CSS,
    )
