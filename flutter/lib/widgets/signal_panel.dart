import 'package:flutter/material.dart';
import '../models/chart_models.dart';

/// 交易信号面板
class SignalPanel extends StatelessWidget {
  final List<TradeSignal> signals;
  final PredictionSummary? summary;

  const SignalPanel({
    Key? key,
    this.signals = const [],
    this.summary,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.auto_awesome, color: Colors.blue, size: 18),
                const SizedBox(width: 6),
                const Text(
                  'AI智能分析',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Spacer(),
                if (summary != null) _buildTrendBadge(),
              ],
            ),
            const SizedBox(height: 12),
            if (summary != null) _buildSummaryRow(),
            const SizedBox(height: 12),
            const Divider(height: 1),
            const SizedBox(height: 8),
            ...signals.take(3).map((signal) => _buildSignalRow(signal)),
            if (signals.isEmpty)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 8),
                child: Center(
                  child: Text(
                    '暂无交易信号',
                    style: TextStyle(color: Colors.grey, fontSize: 12),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTrendBadge() {
    final trend = _getTrend();
    Color color;
    String text;
    switch (trend) {
      case TrendType.up:
        color = Colors.green;
        text = '看涨';
        break;
      case TrendType.down:
        color = Colors.red;
        text = '看跌';
        break;
      case TrendType.neutral:
        color = Colors.orange;
        text = '震荡';
        break;
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color),
      ),
      child: Text(
        text,
        style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.bold),
      ),
    );
  }

  TrendType _getTrend() {
    if (summary?.hasBuySignal ?? false) return TrendType.up;
    if (summary?.hasSellSignal ?? false) return TrendType.down;
    return TrendType.neutral;
  }

  Widget _buildSummaryRow() {
    return Row(
      children: [
        Expanded(
          child: _buildStatItem(
            '预测涨幅',
            '${summary!.priceChangePct.toStringAsFixed(2)}%',
            summary!.priceChangePct >= 0 ? Colors.green : Colors.red,
          ),
        ),
        Expanded(
          child: _buildStatItem(
            '平均置信度',
            '${(summary!.averageConfidence * 100).toStringAsFixed(1)}%',
            Colors.blue,
          ),
        ),
        Expanded(
          child: _buildStatItem(
            '预测目标',
            summary!.maxPredictedPrice.toStringAsFixed(2),
            Colors.purple,
          ),
        ),
      ],
    );
  }

  Widget _buildStatItem(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: const TextStyle(fontSize: 11, color: Colors.grey),
        ),
      ],
    );
  }

  Widget _buildSignalRow(TradeSignal signal) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          Icon(signal.icon, color: signal.color, size: 18),
          const SizedBox(width: 8),
          Text(
            signal.type == SignalType.buy ? '买入信号' : '卖出信号',
            style: const TextStyle(fontSize: 12),
          ),
          const Spacer(),
          Text(
            '¥${signal.price.toStringAsFixed(2)}',
            style: TextStyle(
              color: signal.color,
              fontWeight: FontWeight.bold,
              fontSize: 12,
            ),
          ),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: signal.color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              '${(signal.confidence * 100).toStringAsFixed(0)}%',
              style: TextStyle(color: signal.color, fontSize: 10),
            ),
          ),
        ],
      ),
    );
  }
}
