import 'dart:convert';
import 'package:flutter/foundation.dart';
import '../models/chart_models.dart';

/// 预测服务 - 模拟后端API调用
///
/// 数据链路:
///   UI → PredictionService → 后端API → 模型推理
///   ← 响应JSON ← 图表配置 ←
///
/// 该类模拟完整的请求-响应流程
class PredictionService {
  static const baseUrl = 'https://api.example.com/prediction';

  /// 获取预测结果 (模拟)
  static Future<ChartConfig> fetchPrediction({
    String symbol = 'BTC/USDT',
    String interval = '1d',
    int limit = 60,
  }) async {
    await Future.delayed(const Duration(milliseconds: 800));

    final mockResponse = _generateMockResponse(symbol, limit);

    return compute(_parseChartConfig, mockResponse);
  }

  /// 获取预测摘要
  static Future<PredictionSummary> fetchSummary({
    String symbol = 'BTC/USDT',
  }) async {
    await Future.delayed(const Duration(milliseconds: 300));

    return PredictionSummary(
      hasBuySignal: true,
      hasSellSignal: false,
      buySignalStep: 0,
      sellSignalStep: -1,
      buyConfidence: 0.78,
      sellConfidence: 0.0,
      predictedPrices: [42500, 42800, 43100, 43500, 44000],
      maxPredictedPrice: 44000,
      minPredictedPrice: 42500,
      priceChangePct: 3.52,
      averageConfidence: 0.72,
    );
  }

  /// 解析图表配置 (Isolate中执行)
  static ChartConfig _parseChartConfig(String jsonString) {
    final Map<String, dynamic> data = json.decode(jsonString);
    return ChartConfig.fromJson(data);
  }

  /// 生成模拟响应数据
  static String _generateMockResponse(String symbol, int limit) {
    final candles = <Map<String, dynamic>>[];
    double price = 40000;

    for (int i = 0; i < limit; i++) {
      final trend = i < limit * 0.6 ? 1 : (i < limit * 0.8 ? -1 : 1);
      final change = trend * 80 + (i % 7 - 3) * 40 + (i % 5) * 20;
      final open = price;
      final close = price + change;
      final high = (open > close ? open : close) + 60 + (i % 3) * 30;
      final low = (open < close ? open : close) - 60 - (i % 4) * 20;
      final vol = 800000 + i * 15000.0 + (i % 10) * 50000;

      candles.add({
        'time': DateTime.now()
            .subtract(Duration(days: limit - i))
            .millisecondsSinceEpoch,
        'open': open,
        'high': high,
        'low': low,
        'close': close,
        'vol': vol,
      });

      price = close;
    }

    final lastPrice = price;
    final predictedPrices = <double>[];
    final confidenceUpper = <double>[];
    final confidenceLower = <double>[];
    var p = lastPrice;

    for (int i = 0; i < 5; i++) {
      p += 500 + i * 100;
      final volatility = 0.015 * (1 - 0.7 + i * 0.05);
      final band = p * volatility;
      predictedPrices.add(p);
      confidenceUpper.add(p + band);
      confidenceLower.add(p - band);
    }

    final response = {
      'chartTitle': '$symbol 智能分析',
      'showVolume': true,
      'showMacd': true,
      'showRsi': false,
      'candles': candles,
      'predictionOverlay': {
        'predictedPrices': predictedPrices,
        'confidenceUpper': confidenceUpper,
        'confidenceLower': confidenceLower,
        'startIndex': limit - 1,
        'trend': 'up',
      },
      'tradeSignals': [
        {
          'type': 'buy',
          'price': predictedPrices.first,
          'timeIndex': limit,
          'confidence': 0.78,
          'label': 'AI买入',
        },
      ],
    };

    return json.encode(response);
  }

  /// 流式获取实时预测 (模拟SSE/WebSocket)
  static Stream<ChartConfig> streamPrediction({
    String symbol = 'BTC/USDT',
    Duration interval = const Duration(seconds: 5),
  }) async* {
    while (true) {
      yield await fetchPrediction(symbol: symbol);
      await Future.delayed(interval);
    }
  }
}

/// 本地推理服务 (模拟移动端ONNX推理)
///
/// 当模型部署到移动端时，使用此服务进行本地推理
class LocalInferenceService {
  bool _isInitialized = false;

  Future<void> initialize() async {
    await Future.delayed(const Duration(milliseconds: 500));
    _isInitialized = true;
    debugPrint('本地推理引擎初始化完成');
  }

  bool get isInitialized => _isInitialized;

  /// 本地推理 (模拟)
  Future<PredictionSummary> predict(List<KLineEntity> history) async {
    if (!_isInitialized) {
      throw StateError('推理引擎未初始化');
    }

    await Future.delayed(const Duration(milliseconds: 50));

    final lastPrice = history.isNotEmpty ? history.last.close : 100.0;

    final predictedPrices = <double>[];
    var p = lastPrice;
    for (int i = 0; i < 5; i++) {
      p *= 1.005;
      predictedPrices.add(p);
    }

    return PredictionSummary(
      hasBuySignal: true,
      hasSellSignal: false,
      buySignalStep: 0,
      sellSignalStep: -1,
      buyConfidence: 0.72,
      sellConfidence: 0.0,
      predictedPrices: predictedPrices,
      maxPredictedPrice: predictedPrices.last,
      minPredictedPrice: predictedPrices.first,
      priceChangePct: (predictedPrices.last - lastPrice) / lastPrice * 100,
      averageConfidence: 0.68,
    );
  }
}
