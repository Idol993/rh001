import 'package:flutter/material.dart';

/// K线数据实体
class KLineEntity {
  final int time;
  final double open;
  final double high;
  final double low;
  final double close;
  final double vol;

  const KLineEntity({
    required this.time,
    required this.open,
    required this.high,
    required this.low,
    required this.close,
    required this.vol,
  });

  factory KLineEntity.fromJson(Map<String, dynamic> json) {
    return KLineEntity(
      time: json['time'] ?? 0,
      open: (json['open'] as num).toDouble(),
      high: (json['high'] as num).toDouble(),
      low: (json['low'] as num).toDouble(),
      close: (json['close'] as num).toDouble(),
      vol: (json['vol'] ?? json['volume'] ?? 0).toDouble(),
    );
  }

  Map<String, dynamic> toJson() => {
        'time': time,
        'open': open,
        'high': high,
        'low': low,
        'close': close,
        'vol': vol,
      };
}

/// 趋势类型
enum TrendType { up, down, neutral }

/// 信号类型
enum SignalType { buy, sell }

/// 交易信号
class TradeSignal {
  final SignalType type;
  final double price;
  final int timeIndex;
  final double confidence;
  final String label;

  const TradeSignal({
    required this.type,
    required this.price,
    required this.timeIndex,
    this.confidence = 1.0,
    this.label = '',
  });

  factory TradeSignal.fromJson(Map<String, dynamic> json) {
    return TradeSignal(
      type: json['type'] == 'buy' ? SignalType.buy : SignalType.sell,
      price: (json['price'] as num).toDouble(),
      timeIndex: json['timeIndex'] ?? 0,
      confidence: (json['confidence'] ?? 1.0).toDouble(),
      label: json['label'] ?? '',
    );
  }

  Color get color =>
      type == SignalType.buy ? const Color(0xFF4CAF50) : const Color(0xFFF44336);

  IconData get icon =>
      type == SignalType.buy ? Icons.trending_up : Icons.trending_down;
}

/// 预测覆盖层
class PredictionOverlay {
  final List<double> predictedPrices;
  final List<double> confidenceUpper;
  final List<double> confidenceLower;
  final int startIndex;
  final TrendType trend;

  const PredictionOverlay({
    required this.predictedPrices,
    required this.confidenceUpper,
    required this.confidenceLower,
    required this.startIndex,
    required this.trend,
  });

  factory PredictionOverlay.fromJson(Map<String, dynamic> json) {
    return PredictionOverlay(
      predictedPrices:
          (json['predictedPrices'] as List).map((e) => e.toDouble()).toList(),
      confidenceUpper:
          (json['confidenceUpper'] as List).map((e) => e.toDouble()).toList(),
      confidenceLower:
          (json['confidenceLower'] as List).map((e) => e.toDouble()).toList(),
      startIndex: json['startIndex'] ?? 0,
      trend: _parseTrend(json['trend']),
    );
  }

  static TrendType _parseTrend(String? value) {
    switch (value) {
      case 'up':
        return TrendType.up;
      case 'down':
        return TrendType.down;
      default:
        return TrendType.neutral;
    }
  }

  Color get lineColor {
    switch (trend) {
      case TrendType.up:
        return const Color(0xFF4CAF50);
      case TrendType.down:
        return const Color(0xFFF44336);
      case TrendType.neutral:
        return const Color(0xFFFF9800);
    }
  }

  Color get fillColor => lineColor.withOpacity(0.15);
}

/// 图表配置
class ChartConfig {
  final String chartTitle;
  final List<KLineEntity> candles;
  final bool showVolume;
  final bool showMacd;
  final bool showRsi;
  final PredictionOverlay? predictionOverlay;
  final List<TradeSignal> tradeSignals;

  const ChartConfig({
    required this.chartTitle,
    required this.candles,
    this.showVolume = true,
    this.showMacd = true,
    this.showRsi = false,
    this.predictionOverlay,
    this.tradeSignals = const [],
  });

  factory ChartConfig.fromJson(Map<String, dynamic> json) {
    return ChartConfig(
      chartTitle: json['chartTitle'] ?? 'K线图',
      candles: (json['candles'] as List? ?? [])
          .map((e) => KLineEntity.fromJson(e))
          .toList(),
      showVolume: json['showVolume'] ?? true,
      showMacd: json['showMacd'] ?? true,
      showRsi: json['showRsi'] ?? false,
      predictionOverlay: json['predictionOverlay'] != null
          ? PredictionOverlay.fromJson(json['predictionOverlay'])
          : null,
      tradeSignals: (json['tradeSignals'] as List? ?? [])
          .map((e) => TradeSignal.fromJson(e))
          .toList(),
    );
  }

  Map<String, dynamic> toJson() => {
        'chartTitle': chartTitle,
        'candles': candles.map((e) => e.toJson()).toList(),
        'showVolume': showVolume,
        'showMacd': showMacd,
        'showRsi': showRsi,
        'predictionOverlay': predictionOverlay?.toJson(),
        'tradeSignals': tradeSignals.map((e) => e.toJson()).toList(),
      };
}

/// 预测结果摘要
class PredictionSummary {
  final bool hasBuySignal;
  final bool hasSellSignal;
  final int buySignalStep;
  final int sellSignalStep;
  final double buyConfidence;
  final double sellConfidence;
  final List<double> predictedPrices;
  final double maxPredictedPrice;
  final double minPredictedPrice;
  final double priceChangePct;
  final double averageConfidence;

  const PredictionSummary({
    this.hasBuySignal = false,
    this.hasSellSignal = false,
    this.buySignalStep = -1,
    this.sellSignalStep = -1,
    this.buyConfidence = 0.0,
    this.sellConfidence = 0.0,
    this.predictedPrices = const [],
    this.maxPredictedPrice = 0.0,
    this.minPredictedPrice = 0.0,
    this.priceChangePct = 0.0,
    this.averageConfidence = 0.0,
  });

  factory PredictionSummary.fromJson(Map<String, dynamic> json) {
    return PredictionSummary(
      hasBuySignal: json['has_buy_signal'] ?? false,
      hasSellSignal: json['has_sell_signal'] ?? false,
      buySignalStep: json['buy_signal_step'] ?? -1,
      sellSignalStep: json['sell_signal_step'] ?? -1,
      buyConfidence: (json['buy_confidence'] ?? 0.0).toDouble(),
      sellConfidence: (json['sell_confidence'] ?? 0.0).toDouble(),
      predictedPrices:
          (json['predicted_prices'] as List? ?? []).cast<double>(),
      maxPredictedPrice: (json['max_predicted_price'] ?? 0.0).toDouble(),
      minPredictedPrice: (json['min_predicted_price'] ?? 0.0).toDouble(),
      priceChangePct: (json['price_change_pct'] ?? 0.0).toDouble(),
      averageConfidence: (json['average_confidence'] ?? 0.0).toDouble(),
    );
  }
}
