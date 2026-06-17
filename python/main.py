"""
智能炒股分析APP - 核心逻辑主入口
================================

完整数据链路演示:
    原始OHLCV数据
      ↓ (多因子特征工程管道)
    归一化张量序列
      ↓ (时序预测模型)
    趋势预测结果 (JSON)
      ↓ (跨端代码生成适配器)
    Flutter图表配置代码
      ↓ (前端渲染)
    K线图 + 预测覆盖层 + 买卖信号
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from features.pipeline import FeatureEngineeringPipeline, generate_sample_data, MarketData
from models.predictor import TrendPredictor, SignalGenerator, PredictionResult
from models.loss import ReversalWeightedLoss, TrendProbabilityLoss
from adapter.code_generator import (
    FlutterCodeGenerator,
    PredictionJsonMapper,
    ChartConfig,
    TradeSignalMarker,
    PredictionOverlay,
    TrendType,
    SignalType,
)


def demo_feature_pipeline():
    """演示1: 多因子特征工程管道"""
    print("=" * 60)
    print("模块1: 多因子特征工程管道")
    print("=" * 60)

    data = generate_sample_data(n_days=500)
    print(f"\n原始数据: {len(data)} 条K线")
    print(f"  价格范围: {data.low.min():.2f} ~ {data.high.max():.2f}")
    print(f"  成交量范围: {data.volume.min():.0f} ~ {data.volume.max():.0f}")

    pipeline = FeatureEngineeringPipeline(
        window_size=60,
        pred_steps=5,
        norm_method="zscore",
        use_volume_features=True,
        light_weight=False,
    )

    X, y = pipeline.fit_transform(data)

    print(f"\n特征维度: {pipeline.n_features}")
    print(f"特征列表: {pipeline.feature_names}")
    print(f"\n训练样本形状:")
    print(f"  X (特征): {X.shape}  [n_samples, window_size, n_features]")
    print(f"  y (目标): {y.shape}  [n_samples, pred_steps]")

    pipeline_light = FeatureEngineeringPipeline(
        window_size=30,
        pred_steps=3,
        light_weight=True,
    )
    X_light, _ = pipeline_light.fit_transform(data)
    print(f"\n轻量化模式:")
    print(f"  特征维度: {pipeline_light.n_features}")
    print(f"  样本形状: {X_light.shape}")

    return pipeline, X, y, data


def demo_model(X: np.ndarray, y: np.ndarray):
    """演示2: 时序预测与信号生成模型"""
    print("\n" + "=" * 60)
    print("模块2: 时序预测与信号生成模型")
    print("=" * 60)

    input_dim = X.shape[2]
    pred_steps = y.shape[1]

    model = TrendPredictor(
        input_dim=input_dim,
        pred_steps=pred_steps,
        hidden_dim=128,
        num_lstm_layers=2,
        num_transformer_layers=2,
        nhead=4,
        num_trend_classes=3,
        dropout=0.1,
        use_transformer=True,
        light_weight=False,
    )

    print(f"\n模型架构:")
    print(f"  输入维度: {input_dim}")
    print(f"  预测步数: {pred_steps}")
    print(f"  参数量: {model.count_parameters():,}")

    X_tensor = torch.from_numpy(X[:32])
    price_pred, trend_logits, confidence = model(X_tensor)

    print(f"\n模型输出:")
    print(f"  价格预测: {price_pred.shape}  [batch, pred_steps]")
    print(f"  趋势logits: {trend_logits.shape}  [batch, pred_steps, num_classes]")
    print(f"  置信度: {confidence.shape}  [batch, pred_steps]")

    prediction = model.predict(X_tensor[-1:])

    print(f"\n预测结果样例 (最后一个样本):")
    print(f"  预测价格: {prediction.predicted_prices[0].cpu().numpy().round(3)}")
    print(f"  趋势类别: {prediction.trend_labels[0].cpu().numpy()}")
    print(f"  置信度: {prediction.confidence[0].cpu().numpy().round(3)}")
    print(f"  买入信号: {prediction.buy_signals[0].cpu().numpy()}")
    print(f"  卖出信号: {prediction.sell_signals[0].cpu().numpy()}")

    model_light = TrendPredictor(
        input_dim=input_dim,
        pred_steps=pred_steps,
        hidden_dim=64,
        light_weight=True,
    )
    print(f"\n轻量化模型参数量: {model_light.count_parameters():,}")

    print(f"\n损失函数:")
    print("  1. 反转加权损失 (ReversalWeightedLoss)")
    print("     - 对趋势反转点施加更高惩罚")
    print("     - 提升模型对趋势变化的敏感度")

    reversal_loss = ReversalWeightedLoss(alpha=2.0, base_loss="mse")
    y_tensor = torch.from_numpy(y[:32])
    loss_val = reversal_loss(price_pred, y_tensor)
    print(f"     - 示例损失值: {loss_val.item():.6f}")

    print("\n  2. 趋势概率损失 (TrendProbabilityLoss)")
    print("     - 混合分类+回归损失")
    print("     - 同时优化价格预测和趋势判断")

    trend_loss = TrendProbabilityLoss(
        num_trend_classes=3,
        threshold=0.01,
        lambda_cls=0.5,
        lambda_reg=1.0,
        reversal_alpha=2.0,
    )
    total_loss, loss_dict = trend_loss(price_pred, y_tensor, trend_logits)
    print(f"     - 总损失: {loss_dict['total_loss']:.6f}")
    print(f"     - 分类损失: {loss_dict['classification_loss']:.6f}")
    print(f"     - 回归损失: {loss_dict['regression_loss']:.6f}")

    signal_generator = SignalGenerator(
        buy_threshold=0.6,
        sell_threshold=0.6,
        min_signal_gap=3,
        momentum_steps=1,
    )

    buy_sig, sell_sig = signal_generator.generate_signals(
        prediction.trend_probabilities,
        prediction.confidence,
    )
    print(f"\n信号生成器:")
    print(f"  买入信号数: {buy_sig.sum().item()}")
    print(f"  卖出信号数: {sell_sig.sum().item()}")

    return model, prediction


def demo_adapter(prediction: PredictionResult, data: MarketData, pipeline: FeatureEngineeringPipeline):
    """演示3: 跨端代码生成适配器"""
    print("\n" + "=" * 60)
    print("模块3: 跨端代码生成适配器")
    print("=" * 60)

    predicted_prices_norm = prediction.predicted_prices[0].cpu().numpy()
    predicted_prices = np.array([
        pipeline.inverse_transform_price(p) for p in predicted_prices_norm
    ])

    confidence = prediction.confidence[0].cpu().numpy().tolist()
    trend_probs = prediction.trend_probabilities[0].cpu().numpy().tolist()

    prediction_data = {
        "symbol": "BTC/USDT",
        "predicted_prices": predicted_prices.tolist(),
        "trend_probabilities": trend_probs,
        "confidence": confidence,
        "buy_signals": prediction.buy_signals[0].cpu().numpy().tolist(),
        "sell_signals": prediction.sell_signals[0].cpu().numpy().tolist(),
    }

    historical_data = []
    for i in range(len(data)):
        historical_data.append({
            "time": int(data.timestamps[i]),
            "open": float(data.open[i]),
            "high": float(data.high[i]),
            "low": float(data.low[i]),
            "close": float(data.close[i]),
            "volume": float(data.volume[i]),
        })

    print(f"\n输入数据:")
    print(f"  历史K线: {len(historical_data)} 条")
    print(f"  预测步数: {len(predicted_prices)} 步")

    mapper = PredictionJsonMapper()
    chart_config = mapper.map_prediction_to_chart_config(
        prediction_data,
        historical_data[-60:],
    )

    print(f"\n生成的图表配置:")
    print(f"  K线数量: {len(chart_config.candles)}")
    print(f"  预测价格: {[round(p, 2) for p in chart_config.prediction_overlay.predicted_prices]}")
    print(f"  趋势: {chart_config.prediction_overlay.trend.value}")
    print(f"  交易信号: {len(chart_config.trade_signals)} 个")

    json_output = mapper.map_prediction_to_json(
        prediction_data,
        historical_data[-60:],
    )

    print(f"\nJSON输出 (前500字符):")
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)

    code_generator = FlutterCodeGenerator()

    widget_code = code_generator.generate_widget_code(chart_config)

    print(f"\n生成的Flutter Widget代码行数: {widget_code.count(chr(10))} 行")

    signal_panel_code = code_generator.generate_signal_panel_code(
        [
            TradeSignalMarker(
                signal_type=SignalType.BUY,
                price=predicted_prices[0],
                time_index=60,
                confidence=confidence[0],
                label="AI买入",
            )
        ]
    )
    print(f"生成的信号面板代码行数: {signal_panel_code.count(chr(10))} 行")

    mock_service_code = code_generator.generate_mock_data_link()
    print(f"生成的模拟数据链路代码行数: {mock_service_code.count(chr(10))} 行")

    return chart_config, json_output, widget_code


def demo_full_pipeline():
    """完整数据链路演示"""
    print("\n" + "=" * 60)
    print("完整数据链路演示")
    print("=" * 60)

    print("\n步骤1: 生成/获取原始OHLCV数据")
    data = generate_sample_data(n_days=200, seed=42)
    print(f"  ✓ 获得 {len(data)} 条K线数据")

    print("\n步骤2: 特征工程管道处理")
    pipeline = FeatureEngineeringPipeline(
        window_size=60,
        pred_steps=5,
        norm_method="zscore",
        light_weight=False,
    )
    X, y = pipeline.fit_transform(data)
    print(f"  ✓ 生成 {X.shape[0]} 个训练样本")
    print(f"    特征维度: {pipeline.n_features}")

    print("\n步骤3: 模型训练 (简化演示)")
    model = TrendPredictor(
        input_dim=pipeline.n_features,
        pred_steps=pipeline.pred_steps,
        hidden_dim=64,
        num_lstm_layers=1,
        num_transformer_layers=1,
        nhead=2,
        num_trend_classes=3,
        light_weight=False,
    )
    print(f"  ✓ 模型初始化完成")
    print(f"    参数量: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = TrendProbabilityLoss(
        num_trend_classes=3,
        threshold=0.02,
        lambda_cls=0.3,
        lambda_reg=1.0,
        reversal_alpha=1.5,
    )

    X_train = torch.from_numpy(X)
    y_train = torch.from_numpy(y)

    model.train()
    losses = []
    for epoch in range(5):
        optimizer.zero_grad()
        price_pred, trend_logits, _ = model(X_train)
        loss, loss_dict = loss_fn(price_pred, y_train, trend_logits)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  ✓ 训练完成 (5 epochs)")
    print(f"    初始损失: {losses[0]:.6f}")
    print(f"    最终损失: {losses[-1]:.6f}")

    print("\n步骤4: 模型推理")
    model.eval()
    prediction = model.predict(X_train[-1:])

    last_price = data.close[-1]
    predicted_prices_norm = prediction.predicted_prices[0].cpu().numpy()
    predicted_prices = np.array([
        pipeline.inverse_transform_price(p) for p in predicted_prices_norm
    ])

    print(f"  ✓ 推理完成")
    print(f"    最新价格: {last_price:.2f}")
    print(f"    预测价格: {predicted_prices.round(2)}")
    print(f"    涨跌幅: {(predicted_prices[-1] - last_price) / last_price * 100:.2f}%")

    print("\n步骤5: 生成前端图表配置")
    mapper = PredictionJsonMapper()

    prediction_dict = {
        "symbol": "BTC/USDT",
        "predicted_prices": predicted_prices.tolist(),
        "trend_probabilities": prediction.trend_probabilities[0].cpu().numpy().tolist(),
        "confidence": prediction.confidence[0].cpu().numpy().tolist(),
        "buy_signals": prediction.buy_signals[0].cpu().numpy().tolist(),
        "sell_signals": prediction.sell_signals[0].cpu().numpy().tolist(),
    }

    history = []
    for i in range(max(0, len(data) - 60), len(data)):
        history.append({
            "time": int(data.timestamps[i]),
            "open": float(data.open[i]),
            "high": float(data.high[i]),
            "low": float(data.low[i]),
            "close": float(data.close[i]),
            "volume": float(data.volume[i]),
        })

    chart_config = mapper.map_prediction_to_chart_config(prediction_dict, history)
    print(f"  ✓ 图表配置生成完成")
    print(f"    K线数: {len(chart_config.candles)}")
    print(f"    信号数: {len(chart_config.trade_signals)}")
    print(f"    趋势: {chart_config.prediction_overlay.trend.value}")

    print("\n步骤6: 生成Flutter代码")
    code_gen = FlutterCodeGenerator()
    widget_code = code_gen.generate_widget_code(chart_config)
    print(f"  ✓ Flutter Widget代码生成完成")
    print(f"    代码行数: {widget_code.count(chr(10))}")

    print("\n" + "=" * 60)
    print("✓ 完整数据链路演示完成!")
    print("=" * 60)

    return {
        "pipeline": pipeline,
        "model": model,
        "prediction": prediction,
        "chart_config": chart_config,
        "widget_code": widget_code,
    }


def save_outputs(results: dict):
    """保存输出文件"""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    chart_json = results["chart_config"].to_json()
    with open(os.path.join(output_dir, "chart_config.json"), "w", encoding="utf-8") as f:
        f.write(chart_json)
    print(f"\n图表配置已保存: outputs/chart_config.json")

    with open(os.path.join(output_dir, "trading_chart_widget.dart"), "w", encoding="utf-8") as f:
        f.write(results["widget_code"])
    print(f"Flutter Widget已保存: outputs/trading_chart_widget.dart")

    model_path = os.path.join(output_dir, "model.pt")
    torch.save(results["model"].state_dict(), model_path)
    print(f"模型权重已保存: outputs/model.pt")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("智能炒股分析APP - 核心逻辑系统")
    print("  多因子特征工程 | 时序预测模型 | 跨端代码生成")
    print("=" * 60)

    pipeline, X, y, data = demo_feature_pipeline()
    model, prediction = demo_model(X, y)
    chart_config, json_output, widget_code = demo_adapter(prediction, data, pipeline)

    results = demo_full_pipeline()
    save_outputs(results)

    print("\n📁 项目结构:")
    print("  python/")
    print("    ├── features/")
    print("    │   └── pipeline.py       # 多因子特征工程管道")
    print("    ├── models/")
    print("    │   ├── predictor.py      # 时序预测模型")
    print("    │   └── loss.py           # 自定义损失函数")
    print("    ├── adapter/")
    print("    │   └── code_generator.py # 跨端代码生成适配器")
    print("    └── main.py               # 主入口")
    print("  flutter/lib/")
    print("    ├── main.dart             # Flutter主入口")
    print("    ├── models/")
    print("    │   └── chart_models.dart # 数据模型")
    print("    ├── widgets/")
    print("    │   ├── k_chart_widget.dart  # K线图组件")
    print("    │   └── signal_panel.dart    # 信号面板组件")
    print("    └── services/")
    print("        └── prediction_service.dart # 预测服务")
    print("  outputs/")
    print("    ├── chart_config.json     # 生成的图表配置")
    print("    ├── trading_chart_widget.dart # 生成的Flutter代码")
    print("    └── model.pt              # 模型权重")


if __name__ == "__main__":
    main()
