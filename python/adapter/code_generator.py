"""
跨端代码生成适配器
=================

核心功能:
    将模型预测结果 (JSON格式) 动态映射为 Flutter 前端图表配置代码。
    实现从后端推理到前端渲染的完整数据链路。

数据链路:
    后端模型推理 → JSON预测结果 → 代码生成器 → Flutter图表配置 → 前端渲染

支持的图表组件:
    - K线图 (蜡烛图)
    - 预测趋势覆盖层
    - 买卖点标记
    - 技术指标线 (MA, MACD, RSI等)
    - 置信度带
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class TrendType(str, Enum):
    """趋势类型"""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class SignalType(str, Enum):
    """交易信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class CandleData:
    """K线数据点"""
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TradeSignalMarker:
    """
    交易信号标记

    Attributes:
        signal_type: 信号类型 (buy/sell)
        price: 信号价格
        time_index: 时间索引
        confidence: 置信度 (0-1)
        label: 显示标签
    """
    signal_type: SignalType
    price: float
    time_index: int
    confidence: float = 1.0
    label: str = ""

    def to_flutter_map(self) -> Dict[str, Any]:
        """转换为Flutter端可用的Map格式"""
        return {
            "type": self.signal_type.value,
            "price": self.price,
            "timeIndex": self.time_index,
            "confidence": self.confidence,
            "label": self.label,
            "color": "0xFF4CAF50" if self.signal_type == SignalType.BUY else "0xFFF44336",
            "icon": "Icons.arrow_upward" if self.signal_type == SignalType.BUY else "Icons.arrow_downward",
        }


@dataclass
class PredictionOverlay:
    """
    预测覆盖层配置

    Attributes:
        predicted_prices: 预测价格序列
        confidence_upper: 置信带上界
        confidence_lower: 置信带下界
        start_index: 预测起始索引
        trend: 整体趋势方向
    """
    predicted_prices: List[float]
    confidence_upper: List[float]
    confidence_lower: List[float]
    start_index: int
    trend: TrendType

    def to_flutter_map(self) -> Dict[str, Any]:
        """转换为Flutter端可用的Map格式"""
        return {
            "predictedPrices": self.predicted_prices,
            "confidenceUpper": self.confidence_upper,
            "confidenceLower": self.confidence_lower,
            "startIndex": self.start_index,
            "trend": self.trend.value,
            "lineColor": "0xFF2196F3",
            "fillColor": "0x332196F3",
            "dashPattern": [5, 3],
        }


@dataclass
class IndicatorLine:
    """技术指标线配置"""
    name: str
    values: List[float]
    color: str
    stroke_width: float = 1.0

    def to_flutter_map(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "values": self.values,
            "color": self.color,
            "strokeWidth": self.stroke_width,
        }


@dataclass
class ChartConfig:
    """
    图表完整配置

    包含K线数据、技术指标、预测覆盖层、交易信号等所有图表元素。
    可直接序列化为JSON传递给Flutter端。
    """
    candles: List[CandleData] = field(default_factory=list)
    indicators: List[IndicatorLine] = field(default_factory=list)
    prediction_overlay: Optional[PredictionOverlay] = None
    trade_signals: List[TradeSignalMarker] = field(default_factory=list)
    chart_title: str = "K线图"
    show_volume: bool = True
    show_macd: bool = True
    show_rsi: bool = False

    def to_json(self, indent: Optional[int] = 2) -> str:
        """序列化为JSON字符串"""
        data = {
            "chartTitle": self.chart_title,
            "showVolume": self.show_volume,
            "showMacd": self.show_macd,
            "showRsi": self.show_rsi,
            "candles": [
                {
                    "time": c.time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in self.candles
            ],
            "indicators": [ind.to_flutter_map() for ind in self.indicators],
            "predictionOverlay": (
                self.prediction_overlay.to_flutter_map()
                if self.prediction_overlay
                else None
            ),
            "tradeSignals": [sig.to_flutter_map() for sig in self.trade_signals],
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ChartConfig":
        """从JSON字符串反序列化"""
        data = json.loads(json_str)

        candles = [
            CandleData(
                time=c["time"],
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c["volume"],
            )
            for c in data.get("candles", [])
        ]

        indicators = [
            IndicatorLine(
                name=ind["name"],
                values=ind["values"],
                color=ind["color"],
                stroke_width=ind.get("strokeWidth", 1.0),
            )
            for ind in data.get("indicators", [])
        ]

        prediction = None
        if data.get("predictionOverlay"):
            po = data["predictionOverlay"]
            prediction = PredictionOverlay(
                predicted_prices=po["predictedPrices"],
                confidence_upper=po["confidenceUpper"],
                confidence_lower=po["confidenceLower"],
                start_index=po["startIndex"],
                trend=TrendType(po["trend"]),
            )

        trade_signals = [
            TradeSignalMarker(
                signal_type=SignalType(sig["type"]),
                price=sig["price"],
                time_index=sig["timeIndex"],
                confidence=sig.get("confidence", 1.0),
                label=sig.get("label", ""),
            )
            for sig in data.get("tradeSignals", [])
        ]

        return cls(
            candles=candles,
            indicators=indicators,
            prediction_overlay=prediction,
            trade_signals=trade_signals,
            chart_title=data.get("chartTitle", "K线图"),
            show_volume=data.get("showVolume", True),
            show_macd=data.get("showMacd", True),
            show_rsi=data.get("showRsi", False),
        )


class FlutterCodeGenerator:
    """
    Flutter代码生成器

    将后端预测结果转换为Flutter图表配置代码。

    设计模式: 适配器模式
        - 后端数据模型 → 前端图表配置模型
        - 支持动态生成Dart代码片段
        - 支持生成完整的Widget代码
    """

    def __init__(self, chart_widget_name: str = "TradingChart"):
        self.chart_widget_name = chart_widget_name
        self._imports = [
            "import 'package:flutter/material.dart';",
            "import 'package:k_chart/flutter_k_chart.dart';",
        ]

    def generate_chart_config_code(self, config: ChartConfig) -> str:
        """
        生成图表配置的Dart代码

        Args:
            config: 图表配置

        Returns:
            Dart代码字符串 (ChartConfig对象定义)
        """
        code_lines = []
        code_lines.append("/// 自动生成的图表配置")
        code_lines.append("/// 由后端预测结果动态生成")
        code_lines.append("final ChartConfig chartConfig = ChartConfig(")

        code_lines.append(f"  chartTitle: '{config.chart_title}',")
        code_lines.append(f"  showVolume: {str(config.show_volume).lower()},")
        code_lines.append(f"  showMacd: {str(config.show_macd).lower()},")
        code_lines.append(f"  showRsi: {str(config.show_rsi).lower()},")

        code_lines.append("  candles: [")
        for candle in config.candles[-50:]:
            code_lines.append(
                f"    KLineEntity("
                f"time: {candle.time}, "
                f"open: {candle.open:.2f}, "
                f"high: {candle.high:.2f}, "
                f"low: {candle.low:.2f}, "
                f"close: {candle.close:.2f}, "
                f"vol: {candle.volume:.0f}"
                f"),"
            )
        code_lines.append("  ],")

        if config.prediction_overlay:
            code_lines.append("  predictionOverlay: PredictionOverlay(")
            code_lines.append(
                f"    predictedPrices: {config.prediction_overlay.predicted_prices},"
            )
            code_lines.append(
                f"    confidenceUpper: {config.prediction_overlay.confidence_upper},"
            )
            code_lines.append(
                f"    confidenceLower: {config.prediction_overlay.confidence_lower},"
            )
            code_lines.append(
                f"    startIndex: {config.prediction_overlay.start_index},"
            )
            code_lines.append(
                f"    trend: TrendType.{config.prediction_overlay.trend.value},"
            )
            code_lines.append("  ),")

        code_lines.append("  tradeSignals: [")
        for signal in config.trade_signals:
            sig_type = "SignalType.buy" if signal.signal_type == SignalType.BUY else "SignalType.sell"
            code_lines.append(
                f"    TradeSignal("
                f"type: {sig_type}, "
                f"price: {signal.price:.2f}, "
                f"timeIndex: {signal.time_index}, "
                f"confidence: {signal.confidence:.2f}"
                f"),"
            )
        code_lines.append("  ],")

        code_lines.append(");")

        return "\n".join(code_lines)

    def generate_widget_code(
        self,
        config: ChartConfig,
        widget_name: Optional[str] = None,
    ) -> str:
        """
        生成完整的Flutter Widget代码

        Args:
            config: 图表配置
            widget_name: Widget名称 (默认使用 self.chart_widget_name)

        Returns:
            完整的Dart Widget代码
        """
        name = widget_name or self.chart_widget_name

        config_code = self.generate_chart_config_code(config)

        widget_code = f"""
{''.join(self._imports)}

/// 智能炒股分析图表
/// 由代码生成器自动创建
class {name} extends StatelessWidget {{
  final ChartConfig config;

  const {name}({{
    Key? key,
    required this.config,
  }}) : super(key: key);

  @override
  Widget build(BuildContext context) {{
    return Container(
      height: 400,
      padding: const EdgeInsets.all(8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _buildHeader(),
          const SizedBox(height: 8),
          Expanded(
            child: _buildChart(),
          ),
        ],
      ),
    );
  }}

  Widget _buildHeader() {{
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          config.chartTitle,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        if (config.predictionOverlay != null)
          _buildTrendBadge(),
      ],
    );
  }}

  Widget _buildTrendBadge() {{
    final trend = config.predictionOverlay!.trend;
    Color color;
    String text;
    switch (trend) {{
      case TrendType.up:
        color = Colors.green;
        text = '看涨';
        break;
      case TrendType.down:
        color = Colors.red;
        text = '看跌';
        break;
      case TrendType.neutral:
        color = Colors.grey;
        text = '震荡';
        break;
    }}
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color),
      ),
      child: Text(
        text,
        style: TextStyle(color: color, fontSize: 12),
      ),
    );
  }}

  Widget _buildChart() {{
    return KChartWidget(
      config.candles,
      isLine: false,
      isShowMA: true,
      isShowVOL: config.showVolume,
      isShowMACD: config.showMacd,
      isShowRSI: config.showRsi,
      // 预测覆盖层
      predictionLine: config.predictionOverlay?.toPaintData(),
      // 交易信号
      tradeSignals: config.tradeSignals,
    );
  }}
}}

/// 图表配置数据类
class ChartConfig {{
  final String chartTitle;
  final List<KLineEntity> candles;
  final bool showVolume;
  final bool showMacd;
  final bool showRsi;
  final PredictionOverlay? predictionOverlay;
  final List<TradeSignal> tradeSignals;

  const ChartConfig({{
    required this.chartTitle,
    required this.candles,
    this.showVolume = true,
    this.showMacd = true,
    this.showRsi = false,
    this.predictionOverlay,
    this.tradeSignals = const [],
  }});
}}

/// 预测覆盖层数据
class PredictionOverlay {{
  final List<double> predictedPrices;
  final List<double> confidenceUpper;
  final List<double> confidenceLower;
  final int startIndex;
  final TrendType trend;

  const PredictionOverlay({{
    required this.predictedPrices,
    required this.confidenceUpper,
    required this.confidenceLower,
    required this.startIndex,
    required this.trend,
  }});
}}

/// 趋势类型
enum TrendType {{ up, down, neutral }}

/// 交易信号
class TradeSignal {{
  final SignalType type;
  final double price;
  final int timeIndex;
  final double confidence;

  const TradeSignal({{
    required this.type,
    required this.price,
    required this.timeIndex,
    this.confidence = 1.0,
  }});
}}

/// 信号类型
enum SignalType {{ buy, sell }}

{config_code}
"""
        return widget_code.strip()

    def generate_signal_panel_code(self, signals: List[TradeSignalMarker]) -> str:
        """
        生成交易信号面板代码

        Args:
            signals: 交易信号列表

        Returns:
            Dart Widget代码
        """
        code = """
/// 交易信号面板
class SignalPanel extends StatelessWidget {
  final List<TradeSignal> signals;

  const SignalPanel({
    Key? key,
    required this.signals,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'AI交易信号',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
"""

        for i, signal in enumerate(signals[:5]):
            is_buy = signal.signal_type == SignalType.BUY
            color = "Colors.green" if is_buy else "Colors.red"
            icon = "Icons.trending_up" if is_buy else "Icons.trending_down"
            action = "买入" if is_buy else "卖出"

            code += f"""
            _buildSignalRow(
              icon: {icon},
              color: {color},
              label: '{action}信号',
              price: {signal.price:.2f},
              confidence: {signal.confidence:.1%},
            ),
"""

        code += """
          ],
        ),
      ),
    );
  }

  Widget _buildSignalRow({
    required IconData icon,
    required Color color,
    required String label,
    required double price,
    required String confidence,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(width: 8),
          Text(label),
          const Spacer(),
          Text(
            '¥price.toStringAsFixed(2)',
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              confidence,
              style: TextStyle(color: color, fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }
}
"""
        return code

    def generate_mock_data_link(self) -> str:
        """
        生成模拟数据链路代码

        模拟从后端API获取数据 → 解析 → 渲染的完整流程
        """
        code = """
/// 模拟数据链路服务
/// 演示从后端推理到前端渲染的完整数据流程
class MockDataService {
  static Future<ChartConfig> fetchPrediction() async {
    // 模拟网络延迟
    await Future.delayed(const Duration(milliseconds: 500));

    // 模拟后端返回的JSON数据
    final mockResponse = '''
    {
      "chartTitle": "BTC/USDT 日线",
      "showVolume": true,
      "showMacd": true,
      "showRsi": false,
      "candles": [],
      "predictionOverlay": {
        "predictedPrices": [42500, 42800, 43100, 43500, 44000],
        "confidenceUpper": [42700, 43100, 43500, 44000, 44600],
        "confidenceLower": [42300, 42500, 42700, 43000, 43400],
        "startIndex": 49,
        "trend": "up"
      },
      "tradeSignals": [
        {"type": "buy", "price": 42500, "timeIndex": 49, "confidence": 0.78}
      ]
    }
    ''';

    return _parseChartConfig(mockResponse);
  }

  static ChartConfig _parseChartConfig(String jsonString) {
    // JSON解析逻辑
    return ChartConfig(
      chartTitle: 'BTC/USDT 日线',
      candles: _generateMockCandles(),
      showVolume: true,
      showMacd: true,
      predictionOverlay: const PredictionOverlay(
        predictedPrices: [42500, 42800, 43100, 43500, 44000],
        confidenceUpper: [42700, 43100, 43500, 44000, 44600],
        confidenceLower: [42300, 42500, 42700, 43000, 43400],
        startIndex: 49,
        trend: TrendType.up,
      ),
      tradeSignals: const [
        TradeSignal(
          type: SignalType.buy,
          price: 42500,
          timeIndex: 49,
          confidence: 0.78,
        ),
      ],
    );
  }

  static List<KLineEntity> _generateMockCandles() {
    final List<KLineEntity> candles = [];
    double price = 40000;
    for (int i = 0; i < 50; i++) {
      final change = (i % 7 - 3) * 100 + (i % 3) * 50;
      final open = price;
      final close = price + change;
      final high = max(open, close) + 100;
      final low = min(open, close) - 100;
      final vol = 1000000 + i * 10000.0;

      candles.add(KLineEntity(
        time: DateTime.now().subtract(Duration(days: 50 - i)).millisecondsSinceEpoch,
        open: open,
        high: high,
        low: low,
        close: close,
        vol: vol,
      ));

      price = close;
    }
    return candles;
  }
}
"""
        return code


class PredictionJsonMapper:
    """
    预测结果JSON映射器

    将模型预测结果转换为图表配置的核心映射逻辑。
    这是后端→前端的桥梁层。
    """

    def __init__(self):
        pass

    def map_prediction_to_chart_config(
        self,
        prediction_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
    ) -> ChartConfig:
        """
        将模型预测结果映射为图表配置

        Args:
            prediction_data: 模型预测结果
                {
                    "predicted_prices": [...],
                    "trend_probabilities": [...],
                    "confidence": [...],
                    "buy_signals": [...],
                    "sell_signals": [...],
                }
            historical_data: 历史K线数据

        Returns:
            ChartConfig: 图表配置对象
        """
        candles = [
            CandleData(
                time=int(c.get("time", c.get("timestamp", 0))),
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=float(c.get("volume", c.get("vol", 0))),
            )
            for c in historical_data
        ]

        predicted_prices = prediction_data.get("predicted_prices", [])
        confidence = prediction_data.get("confidence", [])
        start_index = len(candles) - 1

        confidence_upper = []
        confidence_lower = []
        last_price = candles[-1].close if candles else 0

        for i, (price, conf) in enumerate(zip(predicted_prices, confidence)):
            volatility = 0.02 * (1 - conf)
            band = price * volatility
            confidence_upper.append(price + band)
            confidence_lower.append(price - band)

        trend_probs = prediction_data.get("trend_probabilities", [])
        if trend_probs and isinstance(trend_probs[0], list):
            avg_prob = [
                sum(p[i] for p in trend_probs) / len(trend_probs)
                for i in range(len(trend_probs[0]))
            ]
            max_idx = avg_prob.index(max(avg_prob))
            if max_idx == 0:
                trend = TrendType.DOWN
            elif max_idx == 2:
                trend = TrendType.UP
            else:
                trend = TrendType.NEUTRAL
        else:
            if predicted_prices and last_price:
                if predicted_prices[-1] > last_price * 1.01:
                    trend = TrendType.UP
                elif predicted_prices[-1] < last_price * 0.99:
                    trend = TrendType.DOWN
                else:
                    trend = TrendType.NEUTRAL
            else:
                trend = TrendType.NEUTRAL

        prediction_overlay = PredictionOverlay(
            predicted_prices=predicted_prices,
            confidence_upper=confidence_upper,
            confidence_lower=confidence_lower,
            start_index=start_index,
            trend=trend,
        )

        trade_signals = []

        buy_signals = prediction_data.get("buy_signals", [])
        for i, is_buy in enumerate(buy_signals):
            if is_buy and i < len(predicted_prices):
                conf_val = confidence[i] if i < len(confidence) else 0.5
                trade_signals.append(
                    TradeSignalMarker(
                        signal_type=SignalType.BUY,
                        price=predicted_prices[i],
                        time_index=start_index + i + 1,
                        confidence=conf_val,
                        label="AI买入",
                    )
                )

        sell_signals = prediction_data.get("sell_signals", [])
        for i, is_sell in enumerate(sell_signals):
            if is_sell and i < len(predicted_prices):
                conf_val = confidence[i] if i < len(confidence) else 0.5
                trade_signals.append(
                    TradeSignalMarker(
                        signal_type=SignalType.SELL,
                        price=predicted_prices[i],
                        time_index=start_index + i + 1,
                        confidence=conf_val,
                        label="AI卖出",
                    )
                )

        return ChartConfig(
            candles=candles,
            prediction_overlay=prediction_overlay,
            trade_signals=trade_signals,
            chart_title=prediction_data.get("symbol", "K线图"),
            show_volume=True,
            show_macd=True,
        )

    def map_prediction_to_json(
        self,
        prediction_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
    ) -> str:
        """映射并序列化为JSON"""
        config = self.map_prediction_to_chart_config(prediction_data, historical_data)
        return config.to_json()
