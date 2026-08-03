"""
C++ performance kernels for Obsidian Terminal.

Loads libobsidian_core (compiled from cpp/obsidian_core.cpp) via ctypes.
All functions fall back to pure-Python/numpy implementations if the
native library is unavailable, so the system always works.

Build with: clang++ -O3 -std=c++17 -shared -fPIC cpp/obsidian_core.cpp -o cpp/libobsidian_core.dylib
"""
import ctypes
import ctypes.util
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_LIB_NAMES = [
    "libobsidian_core.dylib",
    "libobsidian_core.so",
    "obsidian_core.dll",
]

NATIVE_AVAILABLE = False
_lib = None

# Signal encodings matching strategies.trend_following.Signal
SIGNAL_HOLD = 0
SIGNAL_BUY = 1
SIGNAL_STRONG_BUY = 2
SIGNAL_SELL = 3
SIGNAL_STRONG_SELL = 4

_DIR = Path(__file__).parent.parent / "cpp"
for _name in _LIB_NAMES:
    _path = _DIR / _name
    if _path.exists():
        try:
            _lib = ctypes.CDLL(str(_path))
            NATIVE_AVAILABLE = True
            break
        except OSError:
            _lib = None

if NATIVE_AVAILABLE and _lib is not None:
    _lib.sma_crossover_signals.restype = ctypes.c_int
    _lib.sma_crossover_signals.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
    ]

    _lib.monte_carlo.restype = ctypes.c_int
    _lib.monte_carlo.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]

    _lib.max_drawdown.restype = ctypes.c_int
    _lib.max_drawdown.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double),
    ]


def native_library_available() -> bool:
    """Whether the C++ core library is loaded."""
    return NATIVE_AVAILABLE


def sma_crossover_signals(
    prices: np.ndarray,
    short_window: int = 20,
    long_window: int = 50,
    signal_threshold: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SMA crossover signal series and confidences.

    Mirrors SMAStrategy.generate_signal applied per-bar. Single pass,
    O(n) time and O(1) extra space when the native kernel is available.

    Returns (signals, confidences) int/float arrays of len(prices).
    """
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    n = prices.size

    if NATIVE_AVAILABLE and _lib is not None:
        signals = np.zeros(n, dtype=np.int32)
        confidences = np.zeros(n, dtype=np.float64)
        rc = _lib.sma_crossover_signals(
            prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n, short_window, long_window, signal_threshold,
            signals.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            confidences.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            return _sma_crossover_signals_py(
                prices, short_window, long_window, signal_threshold)
        return signals, confidences

    return _sma_crossover_signals_py(prices, short_window, long_window, signal_threshold)


def _sma_crossover_signals_py(
    prices: np.ndarray,
    short_window: int,
    long_window: int,
    signal_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = prices.size
    signals = np.zeros(n, dtype=np.int32)
    confidences = np.zeros(n, dtype=np.float64)

    short_sma = pd_rolling_mean(prices, short_window)
    long_sma = pd_rolling_mean(prices, long_window)

    for i in range(n):
        if i < long_window - 1:
            continue
        cur_short = short_sma[i]
        cur_long = long_sma[i]
        prev_short = short_sma[i - 1]
        prev_long = long_sma[i - 1]

        cross_above = prev_short <= prev_long and cur_short > cur_long
        cross_below = prev_short >= prev_long and cur_short < cur_long

        distance = (cur_short - cur_long) / cur_long
        confidence = min(abs(distance) / 0.05, 1.0)

        if cross_above:
            sig = SIGNAL_STRONG_BUY if distance > signal_threshold * 2 else SIGNAL_BUY
        elif cross_below:
            sig = SIGNAL_STRONG_SELL if distance < -signal_threshold * 2 else SIGNAL_SELL
        elif cur_short > cur_long * (1 + signal_threshold):
            sig = SIGNAL_BUY
            confidence *= 0.7
        elif cur_short < cur_long * (1 - signal_threshold):
            sig = SIGNAL_SELL
            confidence *= 0.7
        else:
            sig = SIGNAL_HOLD
            confidence = 0.5

        signals[i] = sig
        confidences[i] = confidence

    return signals, confidences


def pd_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean matching pandas behavior (NaN for first window-1)."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    out = np.full(n, np.nan)
    if window <= 0 or n < window:
        return out
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def monte_carlo(
    returns: np.ndarray,
    n_simulations: int,
    initial_capital: float = 100000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap Monte Carlo simulation.

    Returns (final_values, max_drawdowns) arrays of length n_simulations.
    Uses the C++ kernel when available, otherwise a numpy implementation.
    """
    returns = np.ascontiguousarray(returns, dtype=np.float64)
    n_days = returns.size

    if seed is None:
        seed = 0xC0FFEE1234

    if NATIVE_AVAILABLE and _lib is not None:
        finals = np.zeros(n_simulations, dtype=np.float64)
        drawdowns = np.zeros(n_simulations, dtype=np.float64)
        rc = _lib.monte_carlo(
            returns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_days, n_simulations, initial_capital, seed,
            finals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            drawdowns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc == 0:
            return finals, drawdowns

    rng = np.random.default_rng(seed)
    finals = np.zeros(n_simulations)
    drawdowns = np.zeros(n_simulations)
    for sim in range(n_simulations):
        sim_returns = rng.choice(returns, size=n_days, replace=True)
        equity = initial_capital * np.cumprod(1 + sim_returns)
        finals[sim] = equity[-1]
        cummax = np.maximum.accumulate(equity)
        drawdowns[sim] = np.min((equity - cummax) / cummax)
    return finals, drawdowns


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown of an equity curve (negative value)."""
    equity = np.ascontiguousarray(equity, dtype=np.float64)
    if equity.size == 0:
        return 0.0
    if NATIVE_AVAILABLE and _lib is not None:
        out = ctypes.c_double(0.0)
        rc = _lib.max_drawdown(
            equity.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            equity.size, ctypes.byref(out),
        )
        if rc == 0:
            return out.value
    peak = np.maximum.accumulate(equity)
    return float(np.min((equity - peak) / peak))


def sma_trade_simulation(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float = 100000,
    position_size: float = 0.1,
) -> dict:
    """
    Trade simulation over a signal series (BUY on buy signal when flat,
    SELL on sell signal when long). Matches BaseStrategy.backtest logic.
    """
    capital = initial_capital
    position = 0.0
    trades = 0

    for i, sig in enumerate(signals):
        if i >= len(prices):
            break
        price = prices[i]
        if sig in (SIGNAL_BUY, SIGNAL_STRONG_BUY) and position == 0:
            shares = int((capital * position_size) / price)
            if shares > 0:
                position = shares
                capital -= shares * price
                trades += 1
        elif sig in (SIGNAL_SELL, SIGNAL_STRONG_SELL) and position > 0:
            capital += position * price
            position = 0.0
            trades += 1

    if len(prices) > 0:
        final_price = prices[-1]
    else:
        final_price = 0.0
    total_value = capital + position * final_price

    return {
        'initial_capital': initial_capital,
        'final_value': total_value,
        'return': (total_value - initial_capital) / initial_capital,
        'num_trades': trades,
    }
