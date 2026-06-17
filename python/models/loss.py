"""
自定义损失函数
============

核心功能:
1. 趋势反转点加权损失 - 对趋势反转点的预测误差施加更高惩罚
2. 趋势概率损失 - 结合概率分类和回归的混合损失

数学原理:
    趋势反转点检测:
        设价格序列为 p₁, p₂, ..., p_T
        定义趋势方向: d_t = sign(p_t - p_{t-1})
        反转点满足: d_t * d_{t-1} < 0 (方向发生变化)

    反转加权损失:
        L_total = Σ w_t * L(p̂_t, p_t)
        其中 w_t = α if t是反转点 else 1
        α > 1 为反转惩罚系数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ReversalWeightedLoss(nn.Module):
    """
    趋势反转点加权损失函数

    数学原理:
        金融时间序列中，趋势反转点是最重要的交易信号。
        普通MSE损失对所有时间步一视同仁，无法有效学习反转模式。
        本损失通过对反转点施加更高权重，提升模型对趋势变化的敏感度。

    损失公式:
        L = (1/N) * Σ [w_t * (ŷ_t - y_t)²]
        其中 w_t = 1 + α * r_t
        r_t ∈ {0, 1} 表示t时刻是否为反转点
        α 为反转惩罚系数 (默认2.0)

    反转点检测:
        使用二阶差分符号变化检测反转:
        Δy_t = y_t - y_{t-1}
        r_t = 1 if sign(Δy_t) ≠ sign(Δy_{t-1}) else 0
    """

    def __init__(
        self,
        alpha: float = 2.0,
        reversal_threshold: float = 0.0,
        base_loss: str = "mse",
    ):
        """
        初始化反转加权损失

        Args:
            alpha: 反转惩罚系数 (越大越关注反转点)
            reversal_threshold: 反转检测阈值 (过滤小幅波动)
            base_loss: 基础损失类型 'mse' | 'mae' | 'huber'
        """
        super().__init__()
        self.alpha = alpha
        self.reversal_threshold = reversal_threshold
        self.base_loss = base_loss

    def _detect_reversals(self, targets: torch.Tensor) -> torch.Tensor:
        """
        检测目标序列中的趋势反转点

        Args:
            targets: 目标序列 [batch, pred_steps] 或 [batch, pred_steps, features]

        Returns:
            reversal_mask: 反转点掩码 [batch, pred_steps]
        """
        if targets.dim() == 3:
            targets = targets[:, :, 0]

        batch_size, pred_steps = targets.shape

        diff = torch.diff(targets, dim=1, prepend=targets[:, 0:1])

        if self.reversal_threshold > 0:
            diff = torch.where(
                torch.abs(diff) < self.reversal_threshold,
                torch.zeros_like(diff),
                diff,
            )

        sign_diff = torch.sign(diff)

        if pred_steps > 2:
            sign_change = sign_diff[:, 1:] * sign_diff[:, :-1]
            reversal = (sign_change < 0).float()

            reversal_mask = F.pad(reversal, (1, 0), mode="constant", value=0.0)
        else:
            reversal_mask = torch.zeros_like(targets)

        return reversal_mask

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reversal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算损失

        Args:
            predictions: 预测值 [batch, pred_steps] 或 [batch, pred_steps, features]
            targets: 目标值 [batch, pred_steps] 或 [batch, pred_steps, features]
            reversal_mask: 可选的预计算反转掩码

        Returns:
            loss: 标量损失值
        """
        if predictions.dim() == 3 and predictions.shape[-1] > 1:
            pred_prices = predictions[:, :, 0]
        else:
            pred_prices = predictions.squeeze(-1) if predictions.dim() == 3 else predictions

        if targets.dim() == 3 and targets.shape[-1] > 1:
            target_prices = targets[:, :, 0]
        else:
            target_prices = targets.squeeze(-1) if targets.dim() == 3 else targets

        if reversal_mask is None:
            reversal_mask = self._detect_reversals(target_prices)

        weights = 1.0 + self.alpha * reversal_mask

        if self.base_loss == "mse":
            element_loss = (pred_prices - target_prices) ** 2
        elif self.base_loss == "mae":
            element_loss = torch.abs(pred_prices - target_prices)
        elif self.base_loss == "huber":
            element_loss = F.smooth_l1_loss(pred_prices, target_prices, reduction="none")
        else:
            raise ValueError(f"未知损失类型: {self.base_loss}")

        weighted_loss = weights * element_loss
        loss = weighted_loss.mean()

        return loss


class TrendProbabilityLoss(nn.Module):
    """
    趋势概率损失函数 (混合分类+回归损失)

    数学原理:
        将预测分解为两部分:
        1. 趋势分类: 上涨/下跌/震荡 的概率分布
        2. 价格回归: 具体的价格预测值

    总损失:
        L_total = λ_cls * L_classification + λ_reg * L_regression

    趋势定义:
        设 Δp = p_{t+τ} - p_{t}
        - 上涨: Δp > +threshold
        - 下跌: Δp < -threshold
        - 震荡: |Δp| ≤ threshold
    """

    def __init__(
        self,
        num_trend_classes: int = 3,
        threshold: float = 0.01,
        lambda_cls: float = 0.5,
        lambda_reg: float = 1.0,
        reversal_alpha: float = 0.0,
    ):
        """
        初始化趋势概率损失

        Args:
            num_trend_classes: 趋势类别数 (3: 下跌/震荡/上涨)
            threshold: 趋势判断阈值 (归一化后的价格变化幅度)
            lambda_cls: 分类损失权重
            lambda_reg: 回归损失权重
            reversal_alpha: 反转点加权系数 (0表示不使用)
        """
        super().__init__()
        self.num_trend_classes = num_trend_classes
        self.threshold = threshold
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg

        if reversal_alpha > 0:
            self.reversal_loss = ReversalWeightedLoss(alpha=reversal_alpha)
        else:
            self.reversal_loss = None

    def _get_trend_labels(self, target_prices: torch.Tensor) -> torch.Tensor:
        """
        根据价格变化获取趋势标签

        Args:
            target_prices: 目标价格 [batch, pred_steps]

        Returns:
            labels: 趋势类别标签 [batch, pred_steps]
        """
        if target_prices.dim() == 3:
            target_prices = target_prices[:, :, 0]

        batch_size, pred_steps = target_prices.shape

        prev_prices = target_prices[:, 0:1]
        price_changes = target_prices - prev_prices

        if self.num_trend_classes == 3:
            labels = torch.zeros_like(target_prices, dtype=torch.long)
            labels[price_changes > self.threshold] = 2
            labels[price_changes < -self.threshold] = 0
            labels[
                (price_changes >= -self.threshold) & (price_changes <= self.threshold)
            ] = 1
        else:
            labels = (price_changes > 0).long()

        return labels

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trend_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算混合损失

        Args:
            predictions: 价格预测 [batch, pred_steps] 或 [batch, pred_steps, 1]
            targets: 目标价格 [batch, pred_steps] 或 [batch, pred_steps, 1]
            trend_logits: 趋势分类logits [batch, pred_steps, num_classes]

        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量的字典
        """
        loss_dict = {}

        if predictions.dim() == 3 and predictions.shape[-1] == 1:
            pred_prices = predictions.squeeze(-1)
        else:
            pred_prices = predictions

        if targets.dim() == 3 and targets.shape[-1] == 1:
            target_prices = targets.squeeze(-1)
        else:
            target_prices = targets

        if self.reversal_loss is not None:
            reg_loss = self.reversal_loss(pred_prices, target_prices)
        else:
            reg_loss = F.mse_loss(pred_prices, target_prices)

        loss_dict["regression_loss"] = reg_loss.item()

        cls_loss = torch.tensor(0.0, device=predictions.device)
        if trend_logits is not None:
            trend_labels = self._get_trend_labels(target_prices)

            logits_flat = trend_logits.reshape(-1, self.num_trend_classes)
            labels_flat = trend_labels.reshape(-1)

            cls_loss = F.cross_entropy(logits_flat, labels_flat)
            loss_dict["classification_loss"] = cls_loss.item()

        total_loss = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss
        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


class DirectionalAccuracyLoss(nn.Module):
    """
    方向准确率损失

    数学原理:
        在金融预测中，方向的正确性往往比精确的价格更重要。
        该损失惩罚预测方向与实际方向不一致的情况。

    损失公式:
        L_dir = -sign(Δp) * sign(Δp̂)
        当方向一致时损失为-1（最小化目标），不一致时为+1

    与MSE结合:
        L_total = λ * L_mse + (1-λ) * L_dir
    """

    def __init__(self, lambda_mse: float = 0.7):
        super().__init__()
        self.lambda_mse = lambda_mse

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if predictions.dim() == 3 and predictions.shape[-1] == 1:
            pred = predictions.squeeze(-1)
        else:
            pred = predictions

        if targets.dim() == 3 and targets.shape[-1] == 1:
            tgt = targets.squeeze(-1)
        else:
            tgt = targets

        mse_loss = F.mse_loss(pred, tgt)

        target_dir = torch.sign(tgt[:, 1:] - tgt[:, :-1])
        pred_dir = torch.sign(pred[:, 1:] - pred[:, :-1])

        dir_agreement = (target_dir * pred_dir)
        dir_loss = -torch.mean(dir_agreement)

        total_loss = self.lambda_mse * mse_loss + (1 - self.lambda_mse) * dir_loss
        return total_loss
