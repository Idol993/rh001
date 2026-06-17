import 'package:flutter/material.dart';
import 'models/chart_models.dart';
import 'widgets/k_chart_widget.dart';
import 'widgets/signal_panel.dart';
import 'services/prediction_service.dart';

void main() {
  runApp(const SmartTradingApp());
}

class SmartTradingApp extends StatelessWidget {
  const SmartTradingApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '智能炒股分析',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0A0E17),
        cardColor: const Color(0xFF1A1F2E),
        dividerColor: Colors.grey.withOpacity(0.2),
      ),
      home: const TradingHomePage(),
    );
  }
}

class TradingHomePage extends StatefulWidget {
  const TradingHomePage({Key? key}) : super(key: key);

  @override
  State<TradingHomePage> createState() => _TradingHomePageState();
}

class _TradingHomePageState extends State<TradingHomePage> {
  ChartConfig? _chartConfig;
  PredictionSummary? _summary;
  bool _isLoading = true;
  String _selectedSymbol = 'BTC/USDT';

  final List<String> _symbols = const [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'SOL/USDT',
  ];

  @override
  void initState() {
    super.initState();
    _loadPrediction();
  }

  Future<void> _loadPrediction() async {
    setState(() => _isLoading = true);

    try {
      final results = await Future.wait([
        PredictionService.fetchPrediction(symbol: _selectedSymbol),
        PredictionService.fetchSummary(symbol: _selectedSymbol),
      ]);

      setState(() {
        _chartConfig = results[0] as ChartConfig;
        _summary = results[1] as PredictionSummary;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('加载失败: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('智能炒股分析'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadPrediction,
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadPrediction,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildSymbolSelector(),
              const SizedBox(height: 12),
              _buildPriceCard(),
              const SizedBox(height: 12),
              _buildChartContainer(),
              const SizedBox(height: 12),
              _buildSignalPanel(),
              const SizedBox(height: 12),
              _buildBottomInfo(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSymbolSelector() {
    return Container(
      height: 40,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: _symbols.length,
        itemBuilder: (context, index) {
          final symbol = _symbols[index];
          final isSelected = symbol == _selectedSymbol;
          return GestureDetector(
            onTap: () {
              setState(() => _selectedSymbol = symbol);
              _loadPrediction();
            },
            child: Container(
              margin: EdgeInsets.only(right: index < _symbols.length - 1 ? 8 : 0),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: isSelected ? Colors.blue : Colors.white.withOpacity(0.05),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: isSelected ? Colors.blue : Colors.white.withOpacity(0.1),
                ),
              ),
              child: Text(
                symbol,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                  color: isSelected ? Colors.white : Colors.grey,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildPriceCard() {
    final price = _chartConfig?.candles.last.close ?? 0;
    final open = _chartConfig?.candles.first.close ?? price;
    final change = price - open;
    final changePct = open > 0 ? (change / open * 100) : 0.0;
    final isUp = change >= 0;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: isUp
              ? [const Color(0xFF1B5E20), const Color(0xFF2E7D32)]
              : [const Color(0xFFB71C1C), const Color(0xFFC62828)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _selectedSymbol,
            style: const TextStyle(
              fontSize: 14,
              color: Colors.white70,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '\$${price.toStringAsFixed(2)}',
            style: const TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            '${isUp ? '+' : ''}${change.toStringAsFixed(2)} '
            '(${isUp ? '+' : ''}${changePct.toStringAsFixed(2)}%)',
            style: TextStyle(
              fontSize: 14,
              color: Colors.white.withOpacity(0.9),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChartContainer() {
    return Container(
      height: 350,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: Row(
              children: [
                const Text(
                  'K线图',
                  style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
                ),
                const Spacer(),
                if (_chartConfig?.predictionOverlay != null)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.blue.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.auto_awesome, size: 12, color: Colors.blue),
                        SizedBox(width: 4),
                        Text('AI预测',
                            style: TextStyle(fontSize: 10, color: Colors.blue)),
                      ],
                    ),
                  ),
              ],
            ),
          ),
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : _chartConfig != null
                    ? KChartWidget(
                        candles: _chartConfig!.candles,
                        showVolume: _chartConfig!.showVolume,
                        showMacd: _chartConfig!.showMacd,
                        predictionOverlay: _chartConfig!.predictionOverlay,
                        tradeSignals: _chartConfig!.tradeSignals,
                      )
                    : const Center(child: Text('暂无数据')),
          ),
        ],
      ),
    );
  }

  Widget _buildSignalPanel() {
    return SignalPanel(
      signals: _chartConfig?.tradeSignals ?? const [],
      summary: _summary,
    );
  }

  Widget _buildBottomInfo() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '技术指标',
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              _buildIndicatorItem('MACD', '金叉', Colors.green),
              _buildIndicatorItem('RSI', '52.3', Colors.blue),
              _buildIndicatorItem('MA5', '上行', Colors.orange),
              _buildIndicatorItem('布林带', '中轨上方', Colors.purple),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildIndicatorItem(String name, String value, Color color) {
    return Expanded(
      child: Column(
        children: [
          Text(
            value,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            name,
            style: const TextStyle(fontSize: 10, color: Colors.grey),
          ),
        ],
      ),
    );
  }
}
