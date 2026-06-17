import 'package:flutter/material.dart';
import '../models/chart_models.dart';

/// K线图组件
/// 自定义绘制，无需外部依赖
class KChartWidget extends StatelessWidget {
  final List<KLineEntity> candles;
  final bool showVolume;
  final bool showMacd;
  final bool showRsi;
  final PredictionOverlay? predictionOverlay;
  final List<TradeSignal> tradeSignals;

  const KChartWidget({
    Key? key,
    required this.candles,
    this.showVolume = true,
    this.showMacd = false,
    this.showRsi = false,
    this.predictionOverlay,
    this.tradeSignals = const [],
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (candles.isEmpty) {
      return const Center(child: Text('暂无数据'));
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        return CustomPaint(
          size: Size(constraints.maxWidth, constraints.maxHeight),
          painter: KChartPainter(
            candles: candles,
            showVolume: showVolume,
            showMacd: showMacd,
            predictionOverlay: predictionOverlay,
            tradeSignals: tradeSignals,
          ),
        );
      },
    );
  }
}

/// K线图画师
class KChartPainter extends CustomPainter {
  final List<KLineEntity> candles;
  final bool showVolume;
  final bool showMacd;
  final PredictionOverlay? predictionOverlay;
  final List<TradeSignal> tradeSignals;

  KChartPainter({
    required this.candles,
    required this.showVolume,
    required this.showMacd,
    this.predictionOverlay,
    this.tradeSignals = const [],
  });

  @override
  void paint(Canvas canvas, Size size) {
    final priceHeight = showVolume ? size.height * 0.65 : size.height;
    final volumeHeight = showVolume ? size.height * 0.2 : 0;
    final topPadding = 20.0;
    final bottomPadding = 10.0;
    final leftPadding = 50.0;
    final rightPadding = 10.0;

    final chartWidth = size.width - leftPadding - rightPadding;
    final chartHeight = priceHeight - topPadding - bottomPadding;

    final prices = candles.map((e) => e.high).followedBy(candles.map((e) => e.low));
    final maxPrice = prices.reduce((a, b) => a > b ? a : b);
    final minPrice = prices.reduce((a, b) => a < b ? a : b);
    final priceRange = maxPrice - minPrice;

    double priceToY(double price) {
      return topPadding + (1 - (price - minPrice) / priceRange) * chartHeight;
    }

    double indexToX(int index) {
      final candleWidth = chartWidth / candles.length;
      return leftPadding + index * candleWidth + candleWidth / 2;
    }

    _drawGrid(canvas, size, leftPadding, rightPadding, topPadding, chartHeight,
        priceToY, maxPrice, minPrice);

    _drawCandles(canvas, leftPadding, chartWidth, priceToY, indexToX);

    if (predictionOverlay != null) {
      _drawPredictionOverlay(
          canvas, priceToY, indexToX, chartWidth / candles.length);
    }

    if (tradeSignals.isNotEmpty) {
      _drawTradeSignals(canvas, priceToY, indexToX);
    }

    if (showVolume) {
      _drawVolume(
          canvas, size.height - volumeHeight - bottomPadding, volumeHeight, leftPadding, chartWidth);
    }
  }

  void _drawGrid(
      Canvas canvas,
      Size size,
      double leftPadding,
      double rightPadding,
      double topPadding,
      double chartHeight,
      double Function(double) priceToY,
      double maxPrice,
      double minPrice) {
    final gridPaint = Paint()
      ..color = Colors.grey.withOpacity(0.2)
      ..strokeWidth = 0.5;

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
      textAlign: TextAlign.right,
    );

    for (int i = 0; i <= 4; i++) {
      final y = topPadding + chartHeight * i / 4;
      final price = maxPrice - (maxPrice - minPrice) * i / 4;

      canvas.drawLine(
        Offset(leftPadding, y),
        Offset(size.width - rightPadding, y),
        gridPaint,
      );

      textPainter.text = TextSpan(
        text: price.toStringAsFixed(2),
        style: const TextStyle(fontSize: 10, color: Colors.grey),
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(5, y - 6));
    }
  }

  void _drawCandles(
    Canvas canvas,
    double leftPadding,
    double chartWidth,
    double Function(double) priceToY,
    double Function(int) indexToX,
  ) {
    final candleWidth = chartWidth / candles.length * 0.7;

    for (int i = 0; i < candles.length; i++) {
      final candle = candles[i];
      final x = indexToX(i);
      final isUp = candle.close >= candle.open;

      final candlePaint = Paint()
        ..color = isUp ? const Color(0xFFEF5350) : const Color(0xFF26A69A)
        ..style = PaintingStyle.stroke
        ..strokeWidth = candleWidth;

      final bodyTop = priceToY(candle.close > candle.open ? candle.close : candle.open);
      final bodyBottom = priceToY(candle.close > candle.open ? candle.open : candle.close);

      final wickPaint = Paint()
        ..color = isUp ? const Color(0xFFEF5350) : const Color(0xFF26A69A)
        ..strokeWidth = 1;

      canvas.drawLine(
        Offset(x, priceToY(candle.high)),
        Offset(x, priceToY(candle.low)),
        wickPaint,
      );

      if (isUp) {
        final rectPaint = Paint()
          ..color = const Color(0xFFEF5350)
          ..style = PaintingStyle.fill;
        canvas.drawRect(
          Rect.fromLTRB(
              x - candleWidth / 2, bodyTop, x + candleWidth / 2, bodyBottom),
          rectPaint,
        );
      } else {
        final rectPaint = Paint()
          ..color = const Color(0xFF26A69A)
          ..style = PaintingStyle.fill;
        canvas.drawRect(
          Rect.fromLTRB(
              x - candleWidth / 2, bodyTop, x + candleWidth / 2, bodyBottom),
          rectPaint,
        );
      }
    }
  }

  void _drawPredictionOverlay(
    Canvas canvas,
    double Function(double) priceToY,
    double Function(int) indexToX,
    double candleWidth,
  ) {
    final overlay = predictionOverlay!;
    final startX = indexToX(overlay.startIndex);

    final predictionPath = Path();
    final upperPath = Path();
    final lowerPath = Path();

    for (int i = 0; i < overlay.predictedPrices.length; i++) {
      final x = startX + i * candleWidth;
      final y = priceToY(overlay.predictedPrices[i]);
      final upperY = priceToY(overlay.confidenceUpper[i]);
      final lowerY = priceToY(overlay.confidenceLower[i]);

      if (i == 0) {
        predictionPath.moveTo(x, y);
        upperPath.moveTo(x, upperY);
        lowerPath.moveTo(x, lowerY);
      } else {
        predictionPath.lineTo(x, y);
        upperPath.lineTo(x, upperY);
        lowerPath.lineTo(x, lowerY);
      }
    }

    final fillPath = Path()..addPath(upperPath, Offset.zero);
    final lowerReversed = Path();
    for (int i = overlay.predictedPrices.length - 1; i >= 0; i--) {
      final x = startX + i * candleWidth;
      final y = priceToY(overlay.confidenceLower[i]);
      if (i == overlay.predictedPrices.length - 1) {
        lowerReversed.moveTo(x, y);
      } else {
        lowerReversed.lineTo(x, y);
      }
    }
    fillPath.addPath(lowerReversed, Offset.zero);
    fillPath.close();

    final fillPaint = Paint()
      ..color = overlay.fillColor
      ..style = PaintingStyle.fill;
    canvas.drawPath(fillPath, fillPaint);

    final linePaint = Paint()
      ..color = overlay.lineColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawPath(predictionPath, linePaint);

    final endX = startX + (overlay.predictedPrices.length - 1) * candleWidth;
    final arrowPainter = Paint()
      ..color = overlay.lineColor
      ..style = PaintingStyle.fill;
    final endY = priceToY(overlay.predictedPrices.last);
    final arrowPath = Path();
    arrowPath.moveTo(endX, endY - 6);
    arrowPath.lineTo(endX + 8, endY);
    arrowPath.lineTo(endX, endY + 6);
    arrowPath.close();
    canvas.drawPath(arrowPath, arrowPainter);
  }

  void _drawTradeSignals(
    Canvas canvas,
    double Function(double) priceToY,
    double Function(int) indexToX,
  ) {
    for (final signal in tradeSignals) {
      if (signal.timeIndex < 0 || signal.timeIndex >= candles.length + 20) continue;

      final x = indexToX((signal.timeIndex).clamp(0, candles.length - 1));
      final y = priceToY(signal.price);

      final paint = Paint()
        ..color = signal.color
        ..style = PaintingStyle.fill;

      if (signal.type == SignalType.buy) {
        final arrowPath = Path();
        arrowPath.moveTo(x, y - 12);
        arrowPath.lineTo(x - 6, y);
        arrowPath.lineTo(x + 6, y);
        arrowPath.close();
        canvas.drawPath(arrowPath, paint);

        final circlePaint = Paint()
          ..color = Colors.white
          ..style = PaintingStyle.fill;
        canvas.drawCircle(Offset(x, y - 16), 4, circlePaint);
        canvas.drawCircle(Offset(x, y - 16), 4, paint..style = PaintingStyle.stroke..strokeWidth = 1.5);
      } else {
        final arrowPath = Path();
        arrowPath.moveTo(x, y + 12);
        arrowPath.lineTo(x - 6, y);
        arrowPath.lineTo(x + 6, y);
        arrowPath.close();
        canvas.drawPath(arrowPath, paint);

        final circlePaint = Paint()
          ..color = Colors.white
          ..style = PaintingStyle.fill;
        canvas.drawCircle(Offset(x, y + 16), 4, circlePaint);
        canvas.drawCircle(Offset(x, y + 16), 4, paint..style = PaintingStyle.stroke..strokeWidth = 1.5);
      }
    }
  }

  void _drawVolume(
    Canvas canvas,
    double top,
    double height,
    double leftPadding,
    double chartWidth,
  ) {
    final maxVol = candles.map((e) => e.vol).reduce((a, b) => a > b ? a : b);
    final barWidth = chartWidth / candles.length * 0.6;

    for (int i = 0; i < candles.length; i++) {
      final candle = candles[i];
      final barHeight = (candle.vol / maxVol) * height;
      final x = leftPadding + i * (chartWidth / candles.length) + (chartWidth / candles.length - barWidth) / 2;

      final isUp = candle.close >= candle.open;
      final barPaint = Paint()
        ..color = isUp ? const Color(0xFFEF5350) : const Color(0xFF26A69A)
        ..style = PaintingStyle.fill;

      canvas.drawRect(
        Rect.fromLTRB(x, top + height - barHeight, x + barWidth, top + height),
        barPaint,
      );
    }

    final labelPainter = TextPainter(
      text: const TextSpan(
        text: 'VOL',
        style: TextStyle(fontSize: 10, color: Colors.grey),
      ),
      textDirection: TextDirection.ltr,
    );
    labelPainter.layout();
    labelPainter.paint(canvas, Offset(5, top + 2));
  }

  @override
  bool shouldRepaint(covariant KChartPainter oldDelegate) {
    return oldDelegate.candles != candles ||
        oldDelegate.predictionOverlay != predictionOverlay ||
        oldDelegate.tradeSignals != tradeSignals;
  }
}
