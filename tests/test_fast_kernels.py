import numpy as np
import pandas as pd
import pytest

import core.fast_kernels as fk
import core.java_optimizer as jo
from strategies.trend_following import SMAStrategy, Signal
from research.advanced_backtest import MonteCarloSimulator


@pytest.fixture
def price_series():
    rng = np.random.default_rng(7)
    return 100 + np.cumsum(rng.normal(0.1, 1.2, 260))


def test_native_kernel_loaded():
    assert fk.native_library_available(), "C++ library not built - run: make cpp"


def test_sma_kernel_matches_strategy(price_series):
    df = pd.DataFrame(
        {"close": price_series},
        index=pd.date_range("2020-01-01", periods=len(price_series), freq="D"),
    )
    strategy = SMAStrategy(short_window=20, long_window=50)
    reference = strategy.generate_signals(df, "TEST")
    ref_signals = [s.signal for s in reference]
    ref_conf = [s.confidence for s in reference]

    raw, conf = fk.sma_crossover_signals(price_series)

    kernel_to_signal = {
        0: Signal.HOLD, 1: Signal.BUY, 2: Signal.STRONG_BUY,
        3: Signal.SELL, 4: Signal.STRONG_SELL,
    }
    kernel_signals = [kernel_to_signal[int(s)] for s in raw]
    assert kernel_signals == ref_signals
    assert np.allclose(conf, ref_conf)


def test_sma_kernel_backtest_matches_strategy(price_series):
    df = pd.DataFrame(
        {"close": price_series},
        index=pd.date_range("2020-01-01", periods=len(price_series), freq="D"),
    )
    strategy = SMAStrategy()
    fast = strategy.backtest(df, "TEST", initial_capital=100000, position_size=0.1)

    raw, _ = fk.sma_crossover_signals(price_series)
    sim = fk.sma_trade_simulation(price_series, raw)

    assert abs(fast["return"] - sim["return"]) < 1e-9
    assert fast["num_trades"] == sim["num_trades"]


def test_monte_carlo_consistency():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0005, 0.01, 120)
    finals, drawdowns = fk.monte_carlo(returns, n_simulations=2000,
                                       initial_capital=100000)
    assert finals.shape == (2000,)
    assert drawdowns.shape == (2000,)
    assert np.all(finals > 0)
    assert np.all(drawdowns <= 0)
    assert 90000 < finals.mean() < 110000


def test_monte_carlo_simulator_uses_kernel():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0005, 0.01, 120))
    sim = MonteCarloSimulator(n_simulations=500)
    result = sim.simulate(returns, initial_capital=100000)
    assert 500 == sim.n_simulations
    assert result["mean_final_value"] > 0
    assert result["mean_max_drawdown"] <= 0
    assert 0 <= result["probability_profit"] <= 100


def test_max_drawdown_kernel():
    equity = np.array([100.0, 120.0, 90.0, 80.0, 110.0])
    assert abs(fk.max_drawdown(equity) - (-1 / 3)) < 1e-9
    assert fk.max_drawdown(np.array([1.0])) == 0.0


def test_java_optimizer_matches_numpy():
    if not jo.JAVA_AVAILABLE:
        pytest.skip("Java not available - run: make java")

    cov = np.array([[0.04, 0.01, 0.005],
                    [0.01, 0.03, 0.008],
                    [0.005, 0.008, 0.025]])
    mu = np.array([0.12, 0.09, 0.08])
    tickers = ["AAPL", "MSFT", "GOOG"]

    java_res = jo.optimize_portfolio(tickers, mu, cov)
    numpy_res = jo.optimize_numpy(tickers, mu, cov)

    assert java_res["engine"] == "java"
    for t in tickers:
        assert abs(java_res["min_variance"][t] - numpy_res["min_variance"][t]) < 1e-6
        assert abs(java_res["max_sharpe"][t] - numpy_res["max_sharpe"][t]) < 1e-6
    assert abs(java_res["sharpe"] - numpy_res["sharpe"]) < 1e-6
    assert abs(sum(java_res["min_variance"].values()) - 1.0) < 1e-6


def test_java_optimizer_fallback_without_java(monkeypatch):
    monkeypatch.setattr(jo, "JAVA_AVAILABLE", False)
    cov = np.array([[0.04, 0.01], [0.01, 0.03]])
    mu = np.array([0.1, 0.08])
    res = jo.optimize_portfolio(["A", "B"], mu, cov)
    assert res["engine"] == "numpy"
    assert abs(sum(res["min_variance"].values()) - 1.0) < 1e-9
