"""
多因子特征工程管道
================
输入: 原始OHLCV数据 (Open, High, Low, Close, Volume)
输出: 归一化的张量序列 (用于模型训练/推理)

核心功能:
1. 技术指标计算 (基于TA-Lib: MACD, RSI, Bollinger Bands, KDJ, OBV等)
2. 时间滑窗序列切片
3. 特征归一化 (Z-Score / Min-Max)
4. 张量格式输出
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

try:
    import talib
except ImportError:
    talib = None


@dataclass
class MarketData:
    """
    市场行情数据容器

    Attributes:
        timestamps: 时间戳序列
        open: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价
        volume: 成交量
    """
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "MarketData":
        """从DataFrame构建MarketData"""
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame缺少必要列: {col}")

        return cls(
            timestamps=df.index.values if hasattr(df.index, "values") else np.arange(len(df)),
            open=df["open"].values.astype(np.float32),
            high=df["high"].values.astype(np.float32),
            low=df["low"].values.astype(np.float32),
            close=df["close"].values.astype(np.float32),
            volume=df["volume"].values.astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.close)


class FeatureEngineeringPipeline:
    """
    多因子特征工程管道

    数学原理:
        技术指标基于价格/成交量的统计变换，将原始OHLCV数据映射到
        更高维的特征空间，使模型能够捕捉趋势、动量、波动率等市场模式。

    特征维度:
        - 基础价格特征: 5维 (OHLCV的归一化形式)
        - 趋势指标: MACD(3) + MA(3) = 6维
        - 动量指标: RSI(1) + KDJ(3) + MOM(1) = 5维
        - 波动率指标: Bollinger Bands(3) + ATR(1) = 4维
        - 成交量指标: OBV(1) + ADOSC(1) = 2维
        总计: ~22维特征
    """

    def __init__(
        self,
        window_size: int = 60,
        pred_steps: int = 5,
        norm_method: str = "zscore",
        use_volume_features: bool = True,
        light_weight: bool = False,
    ):
        """
        初始化特征工程管道

        Args:
            window_size: 时间滑窗大小 (序列长度)
            pred_steps: 预测未来时间步数
            norm_method: 归一化方法 'zscore' | 'minmax'
            use_volume_features: 是否使用成交量特征
            light_weight: 轻量化模式 (减少特征数量，适配移动端推理)
        """
        self.window_size = window_size
        self.pred_steps = pred_steps
        self.norm_method = norm_method
        self.use_volume_features = use_volume_features
        self.light_weight = light_weight

        self._norm_params = {}
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        """获取特征名称列表"""
        if not self._feature_names:
            self._feature_names = self._build_feature_names()
        return self._feature_names

    @property
    def n_features(self) -> int:
        """特征维度"""
        return len(self.feature_names)

    def _build_feature_names(self) -> List[str]:
        """构建特征名称列表"""
        names = [
            "open_norm", "high_norm", "low_norm", "close_norm",
            "macd", "macd_signal", "macd_hist",
            "rsi",
            "bb_upper", "bb_middle", "bb_lower",
            "atr",
        ]

        if not self.light_weight:
            names.extend([
                "ma5", "ma10", "ma20",
                "kdj_k", "kdj_d", "kdj_j",
                "mom",
            ])

        if self.use_volume_features:
            names.append("volume_norm")
            if not self.light_weight:
                names.extend(["obv", "adosc"])

        return names

    def compute_technical_indicators(self, data: MarketData) -> pd.DataFrame:
        """
        计算技术指标

        数学原理:
            MACD: 移动平均收敛发散指标
                MACD = EMA(close, 12) - EMA(close, 26)
                Signal = EMA(MACD, 9)
                Histogram = MACD - Signal

            RSI: 相对强弱指标
                RSI = 100 - 100 / (1 + RS)
                RS = 平均涨幅 / 平均跌幅

            Bollinger Bands: 布林带
                Middle = SMA(close, 20)
                Upper = Middle + 2 * STD(close, 20)
                Lower = Middle - 2 * STD(close, 20)

            KDJ: 随机指标
                RSV = (close - LLV(low, 9)) / (HHV(high, 9) - LLV(low, 9)) * 100
                K = SMA(RSV, 3)
                D = SMA(K, 3)
                J = 3*K - 2*D
        """
        close = data.close
        high = data.high
        low = data.low
        volume = data.volume

        df = pd.DataFrame({
            "open_norm": data.open,
            "high_norm": data.high,
            "low_norm": data.low,
            "close_norm": close,
            "volume_norm": volume,
        })

        if talib is not None:
            df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )

            df["rsi"] = talib.RSI(close, timeperiod=14)

            df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )

            df["atr"] = talib.ATR(high, low, close, timeperiod=14)

            if not self.light_weight:
                df["ma5"] = talib.MA(close, timeperiod=5)
                df["ma10"] = talib.MA(close, timeperiod=10)
                df["ma20"] = talib.MA(close, timeperiod=20)

                df["kdj_k"], df["kdj_d"] = talib.STOCH(
                    high, low, close,
                    fastk_period=9, slowk_period=3, slowk_matype=0,
                    slowd_period=3, slowd_matype=0,
                )
                df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

                df["mom"] = talib.MOM(close, timeperiod=10)

            if self.use_volume_features and not self.light_weight:
                df["obv"] = talib.OBV(close, volume)
                df["adosc"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        else:
            df = self._compute_indicators_native(df, close, high, low, volume)

        return df

    def _compute_indicators_native(
        self, df: pd.DataFrame, close: np.ndarray,
        high: np.ndarray, low: np.ndarray, volume: np.ndarray
    ) -> pd.DataFrame:
        """纯Python实现的技术指标 (TA-Lib不可用时的备选方案)"""

        def sma(data, period):
            return pd.Series(data).rolling(window=period).mean().values

        def ema(data, period):
            return pd.Series(data).ewm(span=period, adjust=False).mean().values

        ema12 = ema(close, 12)
        ema26 = ema(close, 26)
        df["macd"] = ema12 - ema26
        df["macd_signal"] = ema(df["macd"].values, 9)
        df["macd_hist"] = df["macd"].values - df["macd_signal"].values

        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = sma(gain, 14)
        avg_loss = sma(loss, 14)
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
        df["rsi"] = 100 - 100 / (1 + rs)

        sma20 = sma(close, 20)
        std20 = pd.Series(close).rolling(window=20).std().values
        df["bb_middle"] = sma20
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        df["atr"] = sma(tr, 14)

        if not self.light_weight:
            df["ma5"] = sma(close, 5)
            df["ma10"] = sma(close, 10)
            df["ma20"] = sma(close, 20)

            low_min = pd.Series(low).rolling(window=9).min().values
            high_max = pd.Series(high).rolling(window=9).max().values
            rsv = (close - low_min) / (high_max - low_min) * 100
            rsv = np.nan_to_num(rsv, nan=50)
            df["kdj_k"] = sma(rsv, 3)
            df["kdj_d"] = sma(df["kdj_k"].values, 3)
            df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

            df["mom"] = close - np.roll(close, 10)
            df["mom"][:10] = np.nan

        if self.use_volume_features and not self.light_weight:
            obv = np.zeros_like(volume)
            for i in range(1, len(volume)):
                if close[i] > close[i - 1]:
                    obv[i] = obv[i - 1] + volume[i]
                elif close[i] < close[i - 1]:
                    obv[i] = obv[i - 1] - volume[i]
                else:
                    obv[i] = obv[i - 1]
            df["obv"] = obv

            ad = np.zeros_like(close)
            for i in range(len(close)):
                hl_range = high[i] - low[i]
                if hl_range > 0:
                    ad[i] = ad[i - 1] + (close[i] - low[i] - (high[i] - close[i])) / hl_range * volume[i]
                else:
                    ad[i] = ad[i - 1] if i > 0 else 0
            df["adosc"] = pd.Series(ad).ewm(span=3, adjust=False).mean().values - \
                         pd.Series(ad).ewm(span=10, adjust=False).mean().values

        return df

    def _normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        特征归一化

        Z-Score归一化: x_norm = (x - μ) / σ
            优点: 对异常值相对不敏感，适合金融数据
        Min-Max归一化: x_norm = (x - min) / (max - min)
            优点: 输出范围固定[0,1]，适合需要固定输入范围的模型
        """
        df_norm = df.copy()

        for col in self.feature_names:
            if col not in df_norm.columns:
                continue

            values = df_norm[col].values
            valid_mask = ~np.isnan(values)
            valid_values = values[valid_mask]

            if len(valid_values) == 0:
                continue

            if fit:
                if self.norm_method == "zscore":
                    mean = np.mean(valid_values)
                    std = np.std(valid_values) + 1e-8
                    self._norm_params[col] = {"mean": mean, "std": std}
                else:
                    vmin = np.min(valid_values)
                    vmax = np.max(valid_values)
                    vmax = vmax if vmax != vmin else vmin + 1e-8
                    self._norm_params[col] = {"min": vmin, "max": vmax}

            if col in self._norm_params:
                params = self._norm_params[col]
                if self.norm_method == "zscore":
                    df_norm[col] = (values - params["mean"]) / params["std"]
                else:
                    df_norm[col] = (values - params["min"]) / (params["max"] - params["min"])

        return df_norm

    def create_sliding_windows(
        self, features: np.ndarray, targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        时间滑窗序列切片

        数学原理:
            将长度为T的时间序列 x₁, x₂, ..., x_T
            转换为多个滑窗样本:
                X[i] = [x_{i-window_size+1}, ..., x_i]
                y[i] = [x_{i+1}, ..., x_{i+pred_steps}]  (若提供targets)

        Args:
            features: 特征矩阵 [T, F]
            targets: 目标值 [T, ...]

        Returns:
            X: 滑窗特征 [n_samples, window_size, F]
            y: 滑窗目标 [n_samples, pred_steps, ...] 或 None
        """
        n_samples = len(features) - self.window_size - self.pred_steps + 1
        if targets is not None:
            n_samples = min(n_samples, len(targets) - self.window_size - self.pred_steps + 1)

        if n_samples <= 0:
            raise ValueError(
                f"数据长度不足. 需要至少 {self.window_size + self.pred_steps} 个时间点, "
                f"当前只有 {len(features)} 个"
            )

        X = np.zeros((n_samples, self.window_size, features.shape[1]), dtype=np.float32)

        for i in range(n_samples):
            X[i] = features[i:i + self.window_size]

        y = None
        if targets is not None:
            if targets.ndim == 1:
                y = np.zeros((n_samples, self.pred_steps), dtype=np.float32)
            else:
                y = np.zeros((n_samples, self.pred_steps, targets.shape[1]), dtype=np.float32)

            for i in range(n_samples):
                y[i] = targets[i + self.window_size:i + self.window_size + self.pred_steps]

        return X, y

    def fit_transform(self, data: MarketData, target_col: str = "close_norm") -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合并转换数据 (训练模式)

        Args:
            data: 原始行情数据
            target_col: 目标列名

        Returns:
            X: 训练特征 [n_samples, window_size, n_features]
            y: 训练目标 [n_samples, pred_steps]
        """
        indicators_df = self.compute_technical_indicators(data)
        norm_df = self._normalize_features(indicators_df, fit=True)

        feature_matrix = norm_df[self.feature_names].fillna(0).values.astype(np.float32)
        targets = norm_df[target_col].fillna(0).values.astype(np.float32)

        X, y = self.create_sliding_windows(feature_matrix, targets)
        return X, y

    def transform(self, data: MarketData) -> np.ndarray:
        """
        转换数据 (推理模式)

        Args:
            data: 原始行情数据

        Returns:
            X: 特征张量 [1, window_size, n_features]
        """
        indicators_df = self.compute_technical_indicators(data)
        norm_df = self._normalize_features(indicators_df, fit=False)

        feature_matrix = norm_df[self.feature_names].fillna(0).values.astype(np.float32)

        if len(feature_matrix) < self.window_size:
            raise ValueError(f"推理数据长度不足. 需要 {self.window_size} 个时间点")

        X = feature_matrix[-self.window_size:].reshape(1, self.window_size, -1)
        return X

    def inverse_transform_price(self, norm_value: float) -> float:
        """反归一化价格 (用于将模型输出转换回原始价格尺度)"""
        if "close_norm" not in self._norm_params:
            return norm_value

        params = self._norm_params["close_norm"]
        if self.norm_method == "zscore":
            return norm_value * params["std"] + params["mean"]
        else:
            return norm_value * (params["max"] - params["min"]) + params["min"]


def generate_sample_data(n_days: int = 500, seed: int = 42) -> MarketData:
    """生成模拟行情数据 (用于测试)"""
    np.random.seed(seed)

    t = np.arange(n_days)
    base_price = 100

    trend = 0.02 * t
    seasonality = 5 * np.sin(2 * np.pi * t / 60)
    noise = np.random.normal(0, 1.5, n_days).cumsum()
    close = base_price + trend + seasonality + noise

    high = close + np.abs(np.random.normal(0, 1.0, n_days))
    low = close - np.abs(np.random.normal(0, 1.0, n_days))
    open_ = close + np.random.normal(0, 0.5, n_days)

    for i in range(n_days):
        high[i] = max(high[i], open_[i], close[i])
        low[i] = min(low[i], open_[i], close[i])

    volume = 100000 + 50000 * np.random.randn(n_days) + 20000 * np.abs(np.sin(2 * np.pi * t / 30))
    volume = np.abs(volume)

    return MarketData(
        timestamps=np.arange(n_days),
        open=open_.astype(np.float32),
        high=high.astype(np.float32),
        low=low.astype(np.float32),
        close=close.astype(np.float32),
        volume=volume.astype(np.float32),
    )
