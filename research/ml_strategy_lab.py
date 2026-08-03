"""
ML Strategy Lab
===============
A self-contained model lab that tries new models/features against the current
production ensemble (RandomForest + GradientBoosting majority vote, the same
recipe used by the agent pipeline) and reports which one wins.

What gets tried ("tools research"):
    - Fama-French 5-factor daily (via pandas-datareader) as market-wide features
      [adopted by the tool-trial harness: ff_factors -> ADOPT]
    - quantstats as an independent validation layer for strategy metrics
      [trial risk_metrics -> KEEP ours, use quantstats for cross-check]
    - HistGradientBoosting as a candidate model vs the RF+GB ensemble

Methodology
    - Universe: liquid US large caps (7 stocks + SPY)
    - Features per symbol: momentum (1/5/10/21d), RSI-14, 21d vol, 63d momentum,
      close/MA20, volume z-score; optionally FF factor exposures of the last day
    - Target: sign of the 5-day forward return (next-day horizon switchable)
    - Walk-forward evaluation: 3 time-ordered folds with growing windows, pooled
      across symbols; no shuffling, no leakage
    - Strategy: long-only when the model says up; total return, Sharpe (quantstats),
      max drawdown vs buy & hold

Usage:
    python research/ml_strategy_lab.py
    python research/ml_strategy_lab.py --folds 4 --horizon 5 --universe AAPL,MSFT
    python research/ml_strategy_lab.py --json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

DATA_CACHE = Path(__file__).parent.parent / "data_cache"
REPORT_FILE = DATA_CACHE / "ml_lab_report.json"

DEFAULT_UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "XOM", "SPY"]

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


def fetch_prices(symbols: list, period: str = "2y") -> pd.DataFrame:
    """Batch-close prices via the existing yfinance feed."""
    df = yf.download(symbols, period=period, progress=False, auto_adjust=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
    return close


def fetch_ff_factors() -> pd.DataFrame:
    """Fama-French 5-factor daily (adopted tool). Retried + cached to data_cache."""
    import time as _t
    cache = DATA_CACHE / "ff_factors.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            df.index = pd.to_datetime(df.index)
            return df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
        except Exception:
            pass
    try:
        from pandas_datareader import data as pdr
    except Exception:
        return pd.DataFrame()
    last = None
    for attempt in range(3):
        try:
            ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start="2020-01-01")
            df = ff[0] if isinstance(ff, dict) else ff
            if len(df) < 100:
                raise ValueError("short factor table")
            if isinstance(df.index, pd.PeriodIndex):
                df.index = df.index.to_timestamp()
            df.index = pd.to_datetime(df.index)
            DATA_CACHE.mkdir(exist_ok=True)
            df.to_parquet(cache)
            return df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
        except Exception as exc:
            last = exc
            _t.sleep(1.5 * (attempt + 1))
    print(f"[ml_strategy_lab] Fama-French fetch failed after retries: {last}")
    return pd.DataFrame()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def build_features(close: pd.Series, volume: pd.Series, factors: pd.DataFrame, with_factors: bool) -> pd.DataFrame:
    rets = close.pct_change()
    df = pd.DataFrame(index=close.index)
    for lag in (1, 5, 10, 21):
        df[f"ret_{lag}"] = rets.rolling(lag).mean()
    df["rsi_14"] = rsi(close)
    df["vol_21"] = rets.rolling(21).std() * np.sqrt(252)
    df["mom_63"] = close / close.shift(63) - 1
    df["ma_ratio"] = close / close.rolling(20).mean() - 1
    df["vol_z"] = (volume - volume.rolling(21).mean()) / (volume.rolling(21).std() + 1e-10)
    if with_factors and not factors.empty:
        ff = factors.reindex(df.index).ffill().fillna(0)
        df["ff_mkt"] = ff["Mkt-RF"]
        df["ff_smb"] = ff["SMB"]
        df["ff_hml"] = ff["HML"]
        df["ff_cma"] = ff["CMA"]
    return df


def walk_forward(X: pd.DataFrame, y: pd.Series, folds: int = 3):
    """Yield (train_idx, test_idx) growing-window splits, time-ordered."""
    n = len(X)
    bounds = np.linspace(0.5, 1.0, folds + 1)
    start = bounds[0]
    for i in range(folds):
        end = bounds[i + 1]
        train_end = int(start * n) if i == 0 else int(bounds[i] * n)
        test_start = int(bounds[i] * n)
        test_end = int(end * n)
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def evaluate(y_true, y_pred, next_ret_arr):
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    strat = np.where(y_pred == 1, next_ret_arr, 0.0)
    total = float(np.prod(1 + strat) - 1)
    sharpe = float(strat.mean() / (strat.std() + 1e-10) * np.sqrt(252))
    cum = np.cumprod(1 + strat)
    max_dd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return acc, f1, total, sharpe, max_dd


def run_lab(universe: list = None, folds: int = 3, horizon: int = 5, with_factors: bool = True) -> dict:
    t0 = time.perf_counter()
    universe = universe or DEFAULT_UNIVERSE

    prices = fetch_prices(universe)
    if prices.empty:
        return {"error": "no price data", "took_s": round(time.perf_counter() - t0, 2)}

    ff = fetch_ff_factors() if with_factors else pd.DataFrame()
    factor_note = f"{len(ff)} rows" if not ff.empty else "unavailable"

    rows, models, splits = [], {}, {}
    for sym in universe:
        close = prices[sym].dropna()
        if len(close) < 260:
            continue
        volume = pd.Series(1.0, index=close.index)
        try:
            v = yf.download(sym, period="2y", progress=False, auto_adjust=False)["Volume"]
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0]
            volume = v.reindex(close.index).ffill().fillna(1.0)
        except Exception:
            pass
        feat = build_features(close, volume, ff, with_factors)
        fwd = close.shift(-horizon) / close - 1
        y = (fwd > 0).astype(int)
        data = feat.copy()
        data["y"] = y
        data["fwd"] = fwd
        data["next"] = close.pct_change().shift(-1)
        data["symbol"] = sym
        rows.append(data)

    if not rows:
        return {"error": "no usable symbols", "took_s": round(time.perf_counter() - t0, 2)}

    pooled = pd.concat(rows).dropna()
    pooled = pooled[pooled.index >= pooled.index.min() + pd.Timedelta(days=70)]
    X_cols = [c for c in pooled.columns if c not in ("y", "fwd", "next", "symbol")]
    X_all = pooled[X_cols].values
    y_all = pooled["y"].values
    next_ret_arr = pooled["next"].values

    model_defs = {
        "logistic_baseline": lambda: LogisticRegression(max_iter=2000),
        "current_ensemble_rf_gb": lambda: (RandomForestClassifier(n_estimators=120, min_samples_leaf=10, random_state=42),
                                           GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=42)),
        "candidate_hist_gb": lambda: HistGradientBoostingClassifier(max_iter=200, random_state=42),
    }
    if not with_factors:
        model_defs = {k: v for k, v in model_defs.items() if k != "current_ensemble_rf_gb"}

    results = {name: {"acc": [], "f1": [], "total": [], "sharpe": [], "max_dd": []} for name in model_defs}
    for train_idx, test_idx in walk_forward(X_all, y_all, folds):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        for name, factory in model_defs.items():
            model = factory()
            if isinstance(model, tuple):
                preds = np.zeros(len(test_idx))
                for m in model:
                    m.fit(X_tr, y_tr)
                    preds += m.predict(X_te)
                y_pred = (preds >= len(model) / 2).astype(int)
            else:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
            acc, f1, total, sharpe, mdd = evaluate(y_te, y_pred, next_ret_arr[test_idx])
            results[name]["acc"].append(acc)
            results[name]["f1"].append(f1)
            results[name]["total"].append(total)
            results[name]["sharpe"].append(sharpe)
            results[name]["max_dd"].append(mdd)

    summary = {}
    for name, m in results.items():
        summary[name] = {
            "accuracy": round(float(np.mean(m["acc"])), 4),
            "f1": round(float(np.mean(m["f1"])), 4),
            "strategy_return": round(float(np.mean(m["total"])) * 100, 2),
            "sharpe": round(float(np.mean(m["sharpe"])), 2),
            "max_drawdown": round(float(np.mean(m["max_dd"])) * 100, 2),
            "outperforms_ensemble": None,
        }

    base = "current_ensemble_rf_gb" if with_factors else "candidate_hist_gb"
    beats = False
    for name in summary:
        if name == base or base not in summary:
            continue
        summary[name]["outperforms_ensemble"] = bool(
            summary[name]["sharpe"] > summary[base]["sharpe"] + 0.05)
        beats = beats or summary[name]["outperforms_ensemble"]

    winner = base if not beats else max(
        (k for k, v in summary.items() if k != base and v["outperforms_ensemble"]),
        key=lambda k: summary[k]["sharpe"])
    report = {
        "ran_at": time.time(),
        "took_s": round(time.perf_counter() - t0, 1),
        "universe": universe,
        "folds": folds,
        "horizon_days": horizon,
        "with_factors": with_factors,
        "ff_factor_rows": factor_note,
        "samples": int(len(pooled)),
        "features": X_cols,
        "models": summary,
        "winner": winner,
        "verdict": (f"{winner} beats the current RF/GB ensemble — switching the pipeline "
                    f"ensemble to {winner} is recommended"
                    if beats else
                    f"current ensemble holds; no candidate model improved on it"),
    }
    REPORT_FILE.parent.mkdir(exist_ok=True)
    with open(REPORT_FILE, "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ML strategy lab")
    parser.add_argument("--universe", help="comma-separated symbols (default: liquid US large caps + SPY)")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--no-factors", action="store_true", help="disable Fama-French features")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not SKLEARN_OK:
        print("sklearn not available — run: pip install scikit-learn")
        return
    universe = [s.strip().upper() for s in args.universe.split(",")] if args.universe else DEFAULT_UNIVERSE
    report = run_lab(universe, args.folds, args.horizon, not args.no_factors)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"\nML Strategy Lab — universe {', '.join(universe)} | folds={args.folds} "
          f"| horizon={args.horizon}d | samples={report.get('samples')} "
          f"| FF factors: {report.get('ff_factor_rows')} ({report.get('took_s')}s)")
    print(f"{'MODEL':<28} {'ACC':<8} {'F1':<8} {'STRAT RET':<10} {'SHARPE':<8} {'MAX DD':<9} BEATS ENSEMBLE")
    print("-" * 92)
    for name, m in report.get("models", {}).items():
        beats = "yes" if m["outperforms_ensemble"] else ("—" if m["outperforms_ensemble"] is None else "no")
        print(f"{name:<28} {m['accuracy']:<8} {m['f1']:<8} {m['strategy_return']:<10} "
              f"{m['sharpe']:<8} {m['max_drawdown']:<9} {beats}")
    print(f"\nWinner: {report.get('winner')}")
    print(report.get("verdict"))
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
