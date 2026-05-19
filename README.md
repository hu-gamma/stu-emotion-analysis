# RoBERTa-BiLSTM 情感分析模型

基于论文《RoBERTa-BiLSTM: A Context-Aware Hybrid Model for Sentiment Analysis》的 PyTorch 实现。

## 当前最佳模型（V6c-grid L4-d03-layerwise）

基于 `checkpoints/roberta_bilstm_v6c_L4_d03_layerwise/best_model.pt`，通过参数搜索找到的最佳配置。

| 指标 | 数值 |
|------|------|
| Accuracy | **67.96%** |
| Macro F1 | **0.6101** |
| Weighted F1 | **0.6741** |

**配置**: 4层 RoBERTa 融合 + dropout=0.3 + 分层学习率（RoBERTa 1x, BiLSTM/Classifier 5x）

### 各类别 F1

| 情感 | F1-Score | 相比 V6b |
|------|----------|---------|
| 悲伤 | 0.6943 | +0.027 |
| 快乐 | 0.7718 | +0.004 |
| 愤怒 | 0.4032 | -0.076 |
| 焦虑 | 0.5861 | -0.003 |
| 尴尬 | 0.4207 | -0.024 |
| 厌恶 | 0.6983 | +0.003 |
| **惊讶** | **0.6844** | **+0.109** |
| 好奇 | 0.6224 | -0.005 |

## 历史基线（V6b）

基于 `checkpoints/roberta_bilstm_v6b/best_model.pt`。

| 指标 | 数值 |
|------|------|
| Accuracy | 66.52% |
| Macro F1 | 0.6058 |
| Weighted F1 | 0.6711 |

## V6c-grid 参数搜索实验

通过 `scripts/train/train_csv_v6c_grid.py` 系统性搜索多层融合的最佳配置。

### 搜索空间与结果

| 配置 | 融合层数 | Dropout | 学习率 | 验证 Macro F1 | 测试 Macro F1 |
|------|---------|---------|--------|--------------|--------------|
| L2_d02_unified | 2 | 0.2 | 统一 1e-5 | 0.6544 | 0.5993 |
| L2_d03_unified | 2 | 0.3 | 统一 1e-5 | 0.6511 | 0.5895 |
| **L4_d03_layerwise** | **4** | **0.3** | **分层** | **0.6581** | **0.6101** |

**结论**: 4层融合 + 强 dropout (0.3) + 分层学习率 的组合在测试集上超越 V6b。

### 层权重分析（L4_d03_layerwise 最终权重）

| 层 | 位置 | 权重 |
|----|------|------|
| L1 | 倒数第 4 层 | 0.237 |
| L2 | 倒数第 3 层 | 0.241 |
| L3 | 倒数第 2 层 | 0.258 |
| L4 | 最后 1 层 | **0.264** |

深层（最后两层）权重更高，说明高层语义对情感分类更重要。

## 模型架构

### V6b 标准架构

```
输入文本
    ↓
RoBERTa-base (12层, 768隐藏状态) - 特征提取
    ↓
Dropout (rate=0.1) - 正则化
    ↓
BiLSTM (256隐藏单元) - 序列特征提取
    ↓
Flatten + Dense + Softmax - 分类输出
```

### V6c-grid 多层融合架构

```
输入文本
    ↓
RoBERTa-base (输出最后 N 层隐藏状态)
    ↓
可学习权重加权融合 / 拼接投影
    ↓
Dropout (rate=0.3)
    ↓
BiLSTM (256隐藏单元)
    ↓
Flatten + Dense + Softmax
```

## 项目结构

```
.
├── archive/                  # 废弃代码归档（保留文字记录）
│   └── docs/version_history.md  # 完整版本演进记录
├── data/                     # 当前数据加载器
│   └── dataset_csv_loader_v4.py  # 8类加载器（去掉中性）
├── datasets/                 # 数据集
│   ├── csv/                  # 处理后的CSV数据集
│   │   ├── split_train_enhanced.csv  # 增强训练集（当前默认）
│   │   ├── split_test.csv    # 测试集
│   │   └── ...
│   └── data_processing.md    # 数据处理流程文档
├── models/                   # 模型定义
│   ├── model.py              # RoBERTa-BiLSTM (标准版, V6b)
│   ├── model_v6b.py          # RoBERTa-BiLSTM (增强 Dropout)
│   ├── model_v6c_multilayer.py  # 多层融合 (第一版, 已废弃)
│   └── model_v6c_configurable.py  # 可配置多层融合 (V6c-grid)
├── checkpoints/              # 模型检查点
│   ├── roberta_bilstm_v6c_L4_d03_layerwise/  # 当前最佳 (Macro F1=0.6101)
│   ├── roberta_bilstm_v6b/   # 历史最佳 (Macro F1=0.6058)
│   ├── roberta_bilstm_v6c/   # 多层融合第一版 (Macro F1=0.6032)
│   ├── roberta_bilstm_full/  # 16类历史基线
│   ├── roberta_bilstm_split/ # 数据划分实验
│   ├── roberta_bilstm_no_dreaddit/  # 消融实验
│   └── qwen_lora_8class/     # Qwen2.5-3B 对比模型
├── results/                  # 实验结果
│   ├── roberta_bilstm/       # 8类测试结果
│   └── compare/              # 模型对比报告
├── scripts/
│   ├── train/
│   │   ├── train_csv_v6a.py  # V6b 训练脚本
│   │   ├── train_csv_v6c.py  # V6c 第一版训练脚本
│   │   └── train_csv_v6c_grid.py  # V6c-grid 参数搜索脚本
│   ├── eval/
│   │   ├── test_v6.py        # 测试脚本
│   │   ├── predict_csv.py    # 推理脚本
│   │   └── compare_v6a_qwen_8class.py  # 对比脚本
│   └── data/
├── docs/
│   ├── version_history.md    # 版本演进记录
│   └── thesis_outline.md     # 论文大纲
├── README.md
├── requirements.txt
└── process_goemotions.py
```

## 环境安装

```bash
pip install -r requirements.txt
```

首次运行时会自动下载 RoBERTa 预训练权重。

## 训练

### V6b（历史基线）

```bash
python scripts/train/train_csv_v6a.py
```

### V6c-grid 参数搜索

```bash
# 最佳配置复现
python scripts/train/train_csv_v6c_grid.py \
    --num-fusion-layers 4 \
    --dropout 0.3 \
    --use-layerwise-lr \
    --lr 1e-5

# 自定义配置
python scripts/train/train_csv_v6c_grid.py \
    --num-fusion-layers 2 \
    --dropout 0.2 \
    --fusion-mode concat
```

训练配置：
- 训练集：`split_train_enhanced.csv`（清洗后 + GoEmotions 厌恶/惊讶补充，43,358条）
- 测试集：`split_test.csv`（去掉中性后 9,951条）
- 类别数：8 分类
- 损失函数：Focal Loss (gamma=2.0) + 类别权重 + WeightedRandomSampler
- 优化器：AdamW
- 早停：基于 Macro F1，Patience=5

## 测试

```bash
# 测试当前最佳模型
python scripts/eval/test_v6.py -m checkpoints/roberta_bilstm_v6c_L4_d03_layerwise/best_model.pt

# 测试 V6b 基线
python scripts/eval/test_v6.py -m checkpoints/roberta_bilstm_v6b/best_model.pt

# 指定测试集
python scripts/eval/test_v6.py -m checkpoints/roberta_bilstm_v6c_L4_d03_layerwise/best_model.pt -t datasets/csv/split_test.csv
```

## 推理预测

```bash
# 使用默认测试样本进行预测和可视化
python scripts/eval/predict_csv.py

# 指定测试样本文件
python scripts/eval/predict_csv.py -i datasets/test/test_samples.txt
```

## 模型细节

### RoBERTa-BiLSTM 类

```python
# V6b 标准版
model = RoBERTaBiLSTM(
    num_classes=8,
    dropout_rate=0.1,
    lstm_hidden_size=256
)

# V6c-grid 可配置版
model = RoBERTaBiLSTMConfigurable(
    num_classes=8,
    dropout_rate=0.3,
    lstm_hidden_size=256,
    num_fusion_layers=4,
    fusion_mode='weighted_sum'
)
```

### 前向传播流程（V6c-grid）

1. RoBERTa 编码（`output_hidden_states=True`）: 输出 13 层隐藏状态
2. 选取最后 N 层并加权融合: `[batch_size, seq_len, 768]`
3. Dropout 正则化
4. BiLSTM: `[batch_size, seq_len, 512]`
5. 拼接前向/后向隐藏状态: `[batch_size, 512]`
6. 全连接层 + Softmax: `[batch_size, num_classes]`

## 8类情感标签

| ID | 情感 | ID | 情感 |
|----|------|----|------|
| 0 | 悲伤 | 4 | 尴尬 |
| 1 | 快乐 | 5 | 厌恶 |
| 2 | 愤怒 | 6 | 惊讶 |
| 3 | 焦虑 | 7 | 好奇 |

## 模型版本对比（8类）

| 指标 | V6c-grid L4-d03 | V6b | V6c (第一版) | Qwen2.5-3B |
|------|----------------|-----|-------------|-----------|
| Accuracy | **67.96%** | 66.52% | 66.89% | **68.28%** |
| **Macro F1** | **0.6101** | 0.6058 | 0.6032 | 0.5883 |
| Weighted F1 | **0.6741** | 0.6711 | 0.6715 | **0.6854** |
| 验证最佳 F1 | **0.6581** | ~0.62 | 0.6533 | — |
| 推理延迟 | ~1.7 ms | **1.67 ms** | ~1.7 ms | 734 ms |

**注**: V6c-grid 在 Macro F1 上超越 V6b (+0.0043) 和 Qwen2.5-3B (+0.0218)，成为当前最佳模型。详见 `docs/version_history.md`。

## 版本演进

完整版本历史见 `docs/version_history.md`。主要演进路线：

- **V1**: 16分类基线
- **V2~V3**: 数据加载器升级 → 初步类别合并
- **V4**: 数据划分策略优化
- **V5**: 去掉中性类（9类）
- **V6**: 正式定为 8 类
- **V6a**: 8类 + Focal Loss → Macro F1=0.5703
- **V6b**: V6a + 数据清洗 + GoEmotions 补充 → Macro F1=0.6058
- **V6c**: 多层融合第一版（dropout=0.1）→ 测试集 0.6032，未超越 V6b
- **V6c-grid**: 参数搜索（层数/dropout/学习率）→ **L4+d0.3+layerwise 达到 0.6101，超越 V6b**
- **V7**: 6类实验 → 已废弃

## 依赖版本

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
