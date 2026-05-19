import sys
sys.path.insert(0, '/mnt')
import torch
import numpy as np
from transformers import AutoTokenizer
from models.model_v8 import RoBERTaBiLSTMV8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

model = RoBERTaBiLSTMV8(num_classes=7, dropout_rate=0.3,
                         num_fusion_layers=4, fusion_mode='weighted_sum')
ckpt = torch.load('/mnt/checkpoints/roberta_bilstm_v8_7class_L4_d03_lw/best_model.pt',
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()

text = "这次考试又挂了，我真的受够了这种无力感"
encoding = tokenizer(text, add_special_tokens=True, max_length=128,
                     padding='max_length', truncation=True, return_tensors='pt')
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

attn_weights = model.get_attention_weights(input_ids, attention_mask)
attn_weights = attn_weights.squeeze(0).cpu().numpy()

tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
valid_len = attention_mask.sum().item()
tokens = tokens[:valid_len]
attn_weights = attn_weights[:valid_len]

# 手工合并 char-level tokens 为有意义的词组
# tokenizer 的输出: 这/次/考/试/又/挂/了/，/我/真/的/受/够/了/这/种/无/力/感/[SEP]
char_groups = [
    (0, 4, "这次考试"),     # 这+次+考+试
    (4, 5, "又"),
    (5, 7, "挂了"),         # 挂+了
    (7, 8, "，"),
    (8, 9, "我"),
    (9, 11, "真的"),        # 真+的
    (11, 14, "受够了"),     # 受+够+了
    (14, 16, "这种"),       # 这+种
    (16, 19, "无力感"),     # 无+力+感
]
# Remaining tokens: [SEP] at index 19

words = []
word_weights = []
for start, end, label in char_groups:
    words.append(label)
    word_weights.append(attn_weights[start:end].sum())

# Print
print("Word → Attention Weight:")
for w, weight in zip(words, word_weights):
    marker = " ◀◀◀" if weight > 0.1 else ""
    print(f"  {w:8s}  {weight:.4f}{marker}")

# ---- 可视化：水平条形图 ----
fig, ax = plt.subplots(figsize=(12, 6))

# 情感关键词：受够了、无力感、挂了
emotion_kw = {'受够了', '无力感', '挂了', '真的'}
colors = ['#e74c3c' if w in emotion_kw else '#5d8aa8' for w in words]

y_pos = range(len(words))
bars = ax.barh(y_pos, word_weights, color=colors, edgecolor='white', linewidth=0.8, height=0.65)

for i, (w, weight) in enumerate(zip(words, word_weights)):
    ax.text(weight + 0.005, i, f'{weight:.3f}', va='center', fontsize=11,
            fontweight='bold' if w in emotion_kw else 'normal',
            color='#c0392b' if w in emotion_kw else '#333333')

ax.set_yticks(y_pos)
ax.set_yticklabels(words, fontsize=13)
ax.set_xlabel('Attention Weight', fontsize=14)
ax.set_title('V8 Attention Pooling — 注意力权重分布\n"这次考试又挂了，我真的受够了这种无力感"',
             fontsize=15, fontweight='bold', pad=18)
ax.invert_yaxis()
ax.set_xlim(0, max(word_weights) * 1.25)
ax.grid(axis='x', linestyle='--', alpha=0.35)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='情感关键词 (受够了/无力感/挂了)'),
    Patch(facecolor='#5d8aa8', label='功能词 / 一般词汇'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='lower right',
          framealpha=0.9, edgecolor='#dddddd')

plt.tight_layout()
plt.savefig('/mnt/results/2026-05-11/attention_visualization.png', dpi=200,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("\nSaved: /mnt/results/2026-05-11/attention_visualization.png")

# 同时生成一个竖版（更适合 Word 纵向排版）
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = range(len(words))
colors_v = ['#e74c3c' if w in emotion_kw else '#5d8aa8' for w in words]
bars = ax.bar(x_pos, word_weights, color=colors_v, edgecolor='white', linewidth=0.8, width=0.7)

for i, (w, weight) in enumerate(zip(words, word_weights)):
    ax.text(i, weight + 0.005, f'{weight:.3f}', ha='center', va='bottom', fontsize=11,
            fontweight='bold' if w in emotion_kw else 'normal')

ax.set_xticks(x_pos)
ax.set_xticklabels(words, fontsize=12)
ax.set_ylabel('Attention Weight', fontsize=14)
ax.set_title('V8 Attention Pooling 权重分布\n"这次考试又挂了，我真的受够了这种无力感"',
             fontsize=15, fontweight='bold', pad=18)
ax.set_ylim(0, max(word_weights) * 1.25)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.legend(handles=legend_elements, fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('/mnt/results/2026-05-11/attention_visualization_v.png', dpi=200,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved: /mnt/results/2026-05-11/attention_visualization_v.png")
