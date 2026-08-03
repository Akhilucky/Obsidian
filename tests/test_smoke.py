import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODULES = [
    "analytics.risk",
    "analytics.signals",
    "analytics.technical",
    "core.event_bus",
    "core.feature_store",
    "portfolio.manager",
    "strategies.trend_following",
    "strategies.mean_reversion",
    "strategies.multi_factor",
    "strategies.sentiment",
    "strategies.market_making",
    "strategies.ml_models",
    "data.ingest",
    "data.market_data",
    "data.indian_markets",
    "data.alternative_data",
    "data.crypto",
    "data.company_intelligence",
    "data.openbb_integration",
    "data.mutual_funds",
    "execution.paper_trading",
    "execution.omega",
    "research.ai_analysis",
    "research.backtester",
    "research.advanced_backtest",
    "research.grid_search",
    "research.model_lab",
    "research.screener",
    "research.strategy_builder",
    "research.strategy_marketplace",
]


def test_all_modules_importable():
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failed, "\n".join(failed)


def test_strategies_package_exports():
    from strategies import (
        SMAStrategy,
        EMAStrategy,
        BreakoutStrategy,
        TrendFollowingEnsemble,
        MeanReversionSuite,
        MarketMakingSuite,
        SentimentSuite,
        MLStrategySuite,
    )

    assert all(callable(c) for c in [
        SMAStrategy, EMAStrategy, BreakoutStrategy, TrendFollowingEnsemble,
        MeanReversionSuite, MarketMakingSuite, SentimentSuite, MLStrategySuite,
    ])
