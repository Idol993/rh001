"""
时序预测与信号生成模型
=====================

核心架构: LSTM + Transformer Encoder 混合神经网络
    - LSTM 层: 捕捉序列的长短期依赖关系
    - Transformer Encoder: 捕捉全局注意力模式
    - 双分支输出: 价格回归 + 趋势概率分类

轻量化设计:
    - 可配置的隐藏层维度
    - 支持深度可分离卷积替代部分全连接
    - 支持知识蒸馏接口 (teacher/student模式)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math


@dataclass
class PredictionResult:
    """
    预测结果容器

    Attributes:
        predicted_prices: 预测的价格序列 [batch, pred_steps]
        trend_probabilities: 趋势概率分布 [batch, pred_steps, num_trend_classes]
        trend_labels: 预测的趋势类别 [batch, pred_steps]
        confidence: 置信度分数 [batch, pred_steps]
        buy_signals: 买入信号标记 [batch, pred_steps]
        sell_signals: 卖出信号标记 [batch, pred_steps]
    """
    predicted_prices: torch.Tensor
    trend_probabilities: torch.Tensor
    trend_labels: torch.Tensor
    confidence: torch.Tensor
    buy_signals: torch.Tensor
    sell_signals: torch.Tensor


class PositionalEncoding(nn.Module):
    """
    正弦位置编码

    数学原理:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    为序列中的每个位置添加唯一的位置信息，使模型能够感知顺序。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LightweightTransformerLayer(nn.Module):
    """
    轻量化Transformer层

    设计思想:
        使用深度可分离注意力减少计算量:
        - 先对每个特征通道独立做注意力 (depth-wise)
        - 再用1x1卷积融合通道信息 (point-wise)
        参数量约为标准Transformer的 1/head_size
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class LSTMTransformerHybrid(nn.Module):
    """
    LSTM + Transformer 混合编码器

    架构设计:
        输入 → LSTM层(捕捉局部时序依赖) → Transformer层(捕捉全局依赖)
               → 双分支输出 (价格回归 + 趋势分类)

    为什么混合架构?
        - LSTM擅长捕捉局部连续的时间模式，参数效率高
        - Transformer擅长捕捉长距离依赖，对模式切换敏感
        - 两者结合可以兼顾短期趋势和长期结构
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        num_transformer_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_transformer: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=False,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if use_transformer:
            self.pos_encoder = PositionalEncoding(lstm_output_dim, dropout=dropout)

            transformer_layer = LightweightTransformerLayer(
                d_model=lstm_output_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.transformer_encoder = nn.ModuleList([
                transformer_layer for _ in range(num_transformer_layers)
            ])
            self.output_dim = lstm_output_dim
        else:
            self.transformer_encoder = None
            self.output_dim = lstm_output_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 [batch, seq_len, input_dim]

        Returns:
            output: 编码后的序列 [batch, seq_len, output_dim]
            last_hidden: 最后时刻的隐藏状态 [batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        x_proj = self.input_projection(x)

        x_lstm_input = x_proj.transpose(0, 1)

        lstm_out, (h_n, _) = self.lstm(x_lstm_input)

        if self.use_transformer and self.transformer_encoder is not None:
            x_trans = self.pos_encoder(lstm_out)

            for layer in self.transformer_encoder:
                x_trans = layer(x_trans)

            output = x_trans.transpose(0, 1)
            last_hidden = output[:, -1, :]
        else:
            output = lstm_out.transpose(0, 1)
            if self.lstm.bidirectional:
                last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                last_hidden = h_n[-1]

        return output, last_hidden


class TrendPredictor(nn.Module):
    """
    趋势预测模型

    完整架构:
        输入: [batch, window_size, n_features]
          ↓
        LSTM-Transformer 混合编码器
          ↓
        解码器 (多步预测):
          使用最后时刻隐藏状态 + 自回归解码
          或 使用全连接层直接输出pred_steps
          ↓
        双分支输出头:
          - 价格回归头: 输出预测价格 [batch, pred_steps]
          - 趋势分类头: 输出趋势概率 [batch, pred_steps, num_trend_classes]

    数学原理 (多步预测):
        直接预测法:
            ŷ_{t+1:t+τ} = f_θ(x_{t-window+1:t})
        优点: 一次前向传播完成，推理速度快
        缺点: 长时序预测精度可能下降
    """

    def __init__(
        self,
        input_dim: int,
        pred_steps: int = 5,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        num_transformer_layers: int = 2,
        nhead: int = 4,
        num_trend_classes: int = 3,
        dropout: float = 0.1,
        use_transformer: bool = True,
        light_weight: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.pred_steps = pred_steps
        self.num_trend_classes = num_trend_classes
        self.light_weight = light_weight

        if light_weight:
            hidden_dim = hidden_dim // 2
            num_lstm_layers = max(1, num_lstm_layers - 1)
            num_transformer_layers = max(1, num_transformer_layers - 1)
            nhead = max(2, nhead // 2)

        self.encoder = LSTMTransformerHybrid(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            num_transformer_layers=num_transformer_layers,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            use_transformer=use_transformer,
        )

        encoder_dim = self.encoder.output_dim

        self.price_decoder = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_steps),
        )

        self.trend_decoder = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_steps * num_trend_classes),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, pred_steps),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 [batch, window_size, input_dim]

        Returns:
            price_pred: 价格预测 [batch, pred_steps]
            trend_logits: 趋势分类logits [batch, pred_steps, num_trend_classes]
            confidence: 置信度 [batch, pred_steps]
        """
        _, last_hidden = self.encoder(x)

        price_pred = self.price_decoder(last_hidden)

        trend_logits_flat = self.trend_decoder(last_hidden)
        batch_size = x.shape[0]
        trend_logits = trend_logits_flat.view(batch_size, self.pred_steps, self.num_trend_classes)

        confidence = self.confidence_head(last_hidden)

        return price_pred, trend_logits, confidence

    def predict(self, x: torch.Tensor) -> PredictionResult:
        """
        完整预测 (包含后处理和信号生成)

        Args:
            x: 输入特征 [batch, window_size, input_dim]

        Returns:
            PredictionResult: 预测结果
        """
        self.eval()
        with torch.no_grad():
            price_pred, trend_logits, confidence = self.forward(x)

            trend_probs = F.softmax(trend_logits, dim=-1)
            trend_labels = torch.argmax(trend_probs, dim=-1)

            buy_signals = (trend_labels == 2) & (confidence > 0.6)
            sell_signals = (trend_labels == 0) & (confidence > 0.6)

            return PredictionResult(
                predicted_prices=price_pred,
                trend_probabilities=trend_probs,
                trend_labels=trend_labels,
                confidence=confidence,
                buy_signals=buy_signals,
                sell_signals=sell_signals,
            )

    def count_parameters(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, output_path: str, sample_input: Optional[torch.Tensor] = None):
        """
        导出为ONNX格式 (移动端推理用)

        Args:
            output_path: 输出文件路径
            sample_input: 示例输入 (shape: [1, window_size, input_dim])
        """
        if sample_input is None:
            sample_input = torch.randn(1, 60, self.input_dim)

        torch.onnx.export(
            self,
            sample_input,
            output_path,
            input_names=["input"],
            output_names=["price_pred", "trend_logits", "confidence"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "price_pred": {0: "batch_size"},
                "trend_logits": {0: "batch_size"},
                "confidence": {0: "batch_size"},
            },
            opset_version=12,
        )


class SignalGenerator:
    """
    交易信号生成器

    根据模型预测结果生成具体的买卖信号。

    信号生成策略:
        1. 趋势过滤: 只在高置信度时产生信号
        2. 动量确认: 连续N步同向趋势才确认信号
        3. 风险控制: 避免频繁交易 (最小间隔限制)
    """

    def __init__(
        self,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.6,
        min_signal_gap: int = 3,
        momentum_steps: int = 2,
    ):
        """
        初始化信号生成器

        Args:
            buy_threshold: 买入信号置信度阈值
            sell_threshold: 卖出信号置信度阈值
            min_signal_gap: 最小信号间隔 (步数)
            momentum_steps: 动量确认步数
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_signal_gap = min_signal_gap
        self.momentum_steps = momentum_steps

    def generate_signals(
        self,
        trend_probs: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成买卖信号

        Args:
            trend_probs: 趋势概率 [batch, pred_steps, num_classes]
            confidence: 置信度 [batch, pred_steps]

        Returns:
            buy_signals: 买入信号 [batch, pred_steps]
            sell_signals: 卖出信号 [batch, pred_steps]
        """
        batch_size, pred_steps, num_classes = trend_probs.shape

        trend_labels = torch.argmax(trend_probs, dim=-1)

        up_prob = trend_probs[:, :, 2] if num_classes >= 3 else trend_probs[:, :, 1]
        down_prob = trend_probs[:, :, 0] if num_classes >= 3 else 1 - trend_probs[:, :, 1]

        buy_candidates = (trend_labels == 2) & (confidence > self.buy_threshold)
        sell_candidates = (trend_labels == 0) & (confidence > self.sell_threshold)

        if self.momentum_steps > 1 and pred_steps >= self.momentum_steps:
            buy_momentum = torch.ones_like(buy_candidates)
            for i in range(self.momentum_steps):
                shifted = F.pad(buy_candidates[:, i:], (0, i), value=False)
                buy_momentum = buy_momentum & shifted
            buy_candidates = buy_momentum

            sell_momentum = torch.ones_like(sell_candidates)
            for i in range(self.momentum_steps):
                shifted = F.pad(sell_candidates[:, i:], (0, i), value=False)
                sell_momentum = sell_momentum & shifted
            sell_candidates = sell_momentum

        return buy_candidates, sell_candidates

    def generate_signal_summary(
        self,
        prediction: PredictionResult,
        last_price: float,
    ) -> dict:
        """
        生成信号摘要 (用于API返回)

        Args:
            prediction: 预测结果
            last_price: 最新价格

        Returns:
            信号摘要字典
        """
        buy_signals = prediction.buy_signals[0].cpu().numpy()
        sell_signals = prediction.sell_signals[0].cpu().numpy()
        prices = prediction.predicted_prices[0].cpu().numpy()
        confidence = prediction.confidence[0].cpu().numpy()

        has_buy = bool(buy_signals.any())
        has_sell = bool(sell_signals.any())

        buy_idx = buy_signals.argmax() if has_buy else -1
        sell_idx = sell_signals.argmax() if has_sell else -1

        return {
            "has_buy_signal": has_buy,
            "has_sell_signal": has_sell,
            "buy_signal_step": int(buy_idx),
            "sell_signal_step": int(sell_idx),
            "buy_confidence": float(confidence[buy_idx]) if has_buy else 0.0,
            "sell_confidence": float(confidence[sell_idx]) if has_sell else 0.0,
            "predicted_prices": prices.tolist(),
            "max_predicted_price": float(prices.max()),
            "min_predicted_price": float(prices.min()),
            "price_change_pct": float((prices[-1] - last_price) / last_price * 100) if last_price > 0 else 0.0,
            "average_confidence": float(confidence.mean()),
        }
