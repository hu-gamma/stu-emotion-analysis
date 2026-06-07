"""
RoBERTa-BiLSTM V8 - 可配置多层融合模型
支持分层学习率和多种融合模式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class RoBERTaBiLSTMV8(nn.Module):
    """
    RoBERTa + BiLSTM with configurable multi-layer fusion
    """
    EMOTION_LABELS = ['悲伤', '快乐', '愤怒', '焦虑', '厌恶', '惊讶', '好奇']

    def __init__(
        self,
        num_classes: int = 7,
        dropout_rate: float = 0.3,
        lstm_hidden_size: int = 256,
        num_fusion_layers: int = 4,
        fusion_mode: str = 'weighted_sum',
        freeze_roberta: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.lstm_hidden_size = lstm_hidden_size
        self.num_fusion_layers = num_fusion_layers
        self.fusion_mode = fusion_mode

        # RoBERTa encoder
        self.roberta = AutoModel.from_pretrained(
            'hfl/chinese-roberta-wwm-ext',
            output_hidden_states=True,
        )
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.hidden_size = self.roberta.config.hidden_size  # 768

        # Multi-layer fusion
        if fusion_mode == 'weighted_sum':
            self.layer_weights = nn.Parameter(
                torch.ones(num_fusion_layers) / num_fusion_layers
            )
            fusion_output_dim = self.hidden_size
        elif fusion_mode == 'concat':
            fusion_output_dim = self.hidden_size * num_fusion_layers
            self.layer_weights = None
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=fusion_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier
        classifier_input = lstm_hidden_size * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, classifier_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(classifier_input // 2, num_classes),
        )

    def fuse_layers(self, hidden_states):
        """
        Fuse the last N hidden states from RoBERTa
        hidden_states: list of [batch, seq_len, hidden] tensors (13 layers including embedding)
        Returns: [batch, seq_len, fusion_output_dim]
        """
        # Take last N layers
        selected = hidden_states[-self.num_fusion_layers:]

        if self.fusion_mode == 'weighted_sum':
            # Normalize weights with softmax
            weights = F.softmax(self.layer_weights, dim=0)
            fused = sum(w * h for w, h in zip(weights, selected))
            return fused
        elif self.fusion_mode == 'concat':
            return torch.cat(selected, dim=-1)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            logits: [batch, num_classes]
        """
        # RoBERTa encoding
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # tuple of 13 tensors

        # Multi-layer fusion
        fused = self.fuse_layers(hidden_states)  # [batch, seq_len, fusion_dim]

        # Dropout
        fused = self.dropout(fused)

        # BiLSTM
        lstm_out, _ = self.bilstm(fused)  # [batch, seq_len, lstm_hidden*2]

        # Pooling: take the [CLS] token representation (first token)
        # Or use mean pooling
        # Let's use [CLS] (index 0) as it's the most common approach
        pooled = lstm_out[:, 0, :]  # [batch, lstm_hidden*2]

        # Classifier
        logits = self.classifier(pooled)  # [batch, num_classes]
        return logits

    def get_param_groups(self, base_lr: float = 1e-5):
        """
        Get parameter groups for layer-wise learning rate
        RoBERTa: base_lr, BiLSTM + Classifier: base_lr * 5
        """
        roberta_params = list(self.roberta.parameters())
        other_params = (
            list(self.bilstm.parameters()) +
            list(self.classifier.parameters())
        )
        if self.layer_weights is not None:
            other_params.append(self.layer_weights)

        return [
            {'params': roberta_params, 'lr': base_lr},
            {'params': other_params, 'lr': base_lr * 5},
        ]

    def get_layer_weights(self):
        """Return normalized layer weights"""
        if self.layer_weights is not None:
            return F.softmax(self.layer_weights, dim=0).detach().cpu().numpy()
        return None

    @torch.no_grad()
    def predict(self, text, tokenizer, device):
        """Predict single text"""
        self.eval()
        encoding = tokenizer(
            text,
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

        return {
            'emotion': self.EMOTION_LABELS[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                label: probs[0][i].item()
                for i, label in enumerate(self.EMOTION_LABELS)
            }
        }
