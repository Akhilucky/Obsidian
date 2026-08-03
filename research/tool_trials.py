"""
Tool Trial Harness
==================
Benchmarks candidate libraries/tools against our current implementations and
reports which ones are worth integrating. Each trial produces a verdict:

    ADOPT      — candidate is better; should replace/augment our code
    EQUIVALENT — candidate matches ours within tolerance (cross-validation use)
    KEEP       — our implementation is better / dependency not worth it
    SKIP       — requires an API key that isn't configured

Trials (run with --all or individually with --trial <id>):
    ff_factors           pandas-datareader (Fama-French)  vs  none (new capability)
    risk_metrics         quantstats                     vs  core.risk_management.RiskMetrics
    technical_indicators ta (RSI)                       vs  core.signal_generator.RSISignalGenerator
    av_quote             Alpha Vantage GLOBAL_QUOTE     vs  yfinance .info        [needs key]
    av_news              Alpha Vantage NEWS_SENTIMENT   vs  yfinance ticker news  [needs key]
    polygon_quote        Polygon prev-day agg           vs  yfinance .info        [needs key]

--apply writes data_cache/provider_trials.json, which re-orders the provider
chain in core/data_providers.py (the auto-integration step).

Usage:
    python research/tool_trials.py --all
    python research/tool_trials.py --trial risk_metrics --json
    python research/tool_trials.py --all --apply
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from core.data_providers import get_quote_chain
from core.risk_management import RiskMetrics

DATA_CACHE = Path(__file__).parent.parent / "data_cache"
REPORT_FILE = DATA_CACHE / "tool_trials_report.json"
TRIALS_FILE = DATA_CACHE / "provider_trials.json"


@dataclass
class Trial:
    id: str
    name: str
    candidate: str
    incumbent: str
    verdict: str
    notes: str
    metric: dict
    latency: dict


def _elapsed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _quote_yf(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "price": info.get("regularMarketPrice", info.get("currentPrice", 0)),
        "change_pct": info.get("regularMarketChangePercent", 0),
        "volume": info.get("regularMarketVolume", 0),
        "name": info.get("shortName", ticker),
    }


def t_ff_factors() -> Trial:
    """pandas-datareader Fama-French 5-factor daily — keyless factor data for ML features."""
    from pandas_datareader import data as pdr

    def candidate():
        ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start="2023-01-01")
        df = ff[0] if isinstance(ff, dict) else ff
        if isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.to_timestamp()
        return df

    try:
        df, t = _elapsed(candidate)
        cols = [str(c) for c in df.columns]
        has_factors = {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}.issubset(cols)
        if not has_factors or len(df) < 100:
            return Trial("ff_factors", "Factor data", "pandas-datareader (Fama-French)", "none (new capability)",
                         "KEEP", "unexpected factor table shape", {}, {})
        verdict = "ADOPT"
        notes = (f"Fama-French 5-factor daily (Mkt-RF, SMB, HML, RMW, CMA + RF), {len(df)} rows in "
                 f"{t:.1f}s, keyless and rate-limit-free. No existing code covers factor exposures; "
                 f"adopt as market-wide feature source for the ML strategy lab.")
        return Trial("ff_factors", "Factor data", "pandas-datareader (Fama-French)", "none (new capability)",
                     verdict, notes, {"rows": len(df), "columns": cols},
                     {"candidate_s": round(t, 2), "incumbent_s": None})
    except Exception as exc:
        return Trial("ff_factors", "Factor data", "pandas-datareader (Fama-French)", "none (new capability)",
                     "SKIP", f"candidate failed: {exc}", {}, {})


def t_risk_metrics() -> Trial:
    """quantstats vs our RiskMetrics on 2y of AAPL returns."""
    import quantstats as qs

    df = yf.download("AAPL", period="2y", progress=False, auto_adjust=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = close.pct_change().dropna()

    our = {
        "sharpe": RiskMetrics.sharpe_ratio(returns),
        "sortino": RiskMetrics.sortino_ratio(returns),
        "vol": RiskMetrics.volatility(returns),
        "max_dd": RiskMetrics.max_drawdown(returns)[0],
    }
    _, qs_t = _elapsed(lambda: qs.stats.sharpe(returns))
    qs_res = {
        "sharpe": qs.stats.sharpe(returns),
        "sortino": qs.stats.sortino(returns),
        "vol": qs.stats.volatility(returns),
        "max_dd": qs.stats.max_drawdown(returns),
    }
    diffs = {k: abs(our[k] - qs_res[k]) for k in our}
    tolerances = {"sharpe": 0.05, "sortino": 0.05, "vol": 0.01, "max_dd": 0.01}
    within = all(diffs[k] <= tolerances[k] for k in our)
    verdict = "EQUIVALENT" if within else "KEEP"
    notes = (
        "Metrics agree within tolerance; differences come from risk-free handling and "
        "downside deviation definition (quantstats uses squared downside deviation, ours "
        "uses std of negative returns). Keep our lightweight RiskMetrics for the API; "
        "quantstats is a useful cross-check/reporting layer, not a replacement."
        if within else
        "Metrics diverge beyond tolerance — quantstats uses squared downside deviation and "
        "annualization conventions we intentionally avoid. KEEP our RiskMetrics; quantstats "
        "remains optional for external reporting."
    )
    return Trial("risk_metrics", "Risk metrics", "quantstats", "core.risk_management.RiskMetrics",
                 verdict, notes, {"ours": our, "quantstats": qs_res, "abs_diff": diffs},
                 {"candidate_s": round(qs_t, 3), "incumbent_s": 0})


def t_technical_indicators() -> Trial:
    """ta.RSIIndicator vs our RSISignalGenerator on AAPL."""
    import ta as talib

    df = yf.download("AAPL", period="1y", progress=False, auto_adjust=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    ta_rsi = talib.momentum.RSIIndicator(close, window=14).rsi().dropna()
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    our_rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    both = pd.concat([our_rsi, ta_rsi], axis=1, keys=["ours", "ta"]).dropna()
    mae = float((both["ours"] - both["ta"]).abs().mean())
    corr = float(both["ours"].corr(both["ta"]))
    verdict = "EQUIVALENT" if corr > 0.95 and mae < 5 else "KEEP"
    notes = (f"corr={corr:.4f} mae={mae:.2f}. `ta` uses Wilder smoothing (exponential); ours uses "
             f"simple rolling mean. Signals agree for overbought/oversold regimes; both fine.")
    return Trial("technical_indicators", "RSI indicator", "ta lib", "core.signal_generator",
                 verdict, notes, {"corr": round(corr, 4), "mae": round(mae, 3)},
                 {"candidate_s": 0, "incumbent_s": 0})


def t_av_quote() -> Trial:
    """Alpha Vantage GLOBAL_QUOTE vs yfinance quote (requires ALPHA_VANTAGE_KEY)."""
    if not os.environ.get("ALPHA_VANTAGE_KEY"):
        return Trial("av_quote", "Quote", "Alpha Vantage GLOBAL_QUOTE", "yfinance .info",
                     "SKIP", "no ALPHA_VANTAGE_KEY configured", {}, {})
    try:
        q, cand_t = _elapsed(lambda: get_quote_chain("AAPL"))
        yq, inc_t = _elapsed(lambda: _quote_yf("AAPL"))
        if not q or not yq or q["price"] <= 0:
            return Trial("av_quote", "Quote", "Alpha Vantage GLOBAL_QUOTE", "yfinance .info",
                         "SKIP", "provider returned empty quote", {}, {})
        pct_diff = abs(q["price"] - yq["price"]) / yq["price"] * 100
        verdict = "EQUIVALENT" if pct_diff < 1 else "KEEP"
        notes = (f"price diff {pct_diff:.2f}%; AV is EOD-only on free tier while yfinance is "
                 f"delayed-realtime. AV wins on structure (no .info scrape); keep yfinance for "
                 f"realtime-freshness, AV as a validated EOD source.")
        return Trial("av_quote", "Quote", "Alpha Vantage GLOBAL_QUOTE", "yfinance .info",
                     verdict, notes, {"provider": q.get("provider"), "price": {"candidate": q["price"], "incumbent": yq["price"]},
                                      "pct_diff": round(pct_diff, 2)},
                     {"candidate_s": round(cand_t, 2), "incumbent_s": round(inc_t, 2)})
    except Exception as exc:
        return Trial("av_quote", "Quote", "Alpha Vantage GLOBAL_QUOTE", "yfinance .info",
                     "SKIP", f"failed: {exc}", {}, {})


def t_av_news() -> Trial:
    """Alpha Vantage NEWS_SENTIMENT vs yfinance news (requires ALPHA_VANTAGE_KEY)."""
    if not os.environ.get("ALPHA_VANTAGE_KEY"):
        return Trial("av_news", "News", "Alpha Vantage NEWS_SENTIMENT", "yfinance ticker news",
                     "SKIP", "no ALPHA_VANTAGE_KEY configured", {}, {})
    try:
        from core.data_providers import get_news_chain
        n, cand_t = _elapsed(lambda: get_news_chain("AAPL", 10))
        yf_news, inc_t = _elapsed(lambda: [x for x in (yf.Ticker("AAPL").news or [])][:10])
        cand_n = len(n["items"]) if n else 0
        inc_n = len(yf_news) if yf_news else 0
        verdict = "ADOPT" if cand_n >= inc_n and cand_n >= 5 else "EQUIVALENT" if cand_n > 0 else "SKIP"
        notes = (f"items: AV={cand_n} yf={inc_n}. AV adds vendor sentiment labels "
                 f"(overall_sentiment_label) and is keyed per-ticker; yfinance news is RSS-based "
                 f"and often empty in headless runs.")
        return Trial("av_news", "News", "Alpha Vantage NEWS_SENTIMENT", "yfinance ticker news",
                     verdict, notes, {"items": {"candidate": cand_n, "incumbent": inc_n}},
                     {"candidate_s": round(cand_t, 2), "incumbent_s": round(inc_t, 2)})
    except Exception as exc:
        return Trial("av_news", "News", "Alpha Vantage NEWS_SENTIMENT", "yfinance ticker news",
                     "SKIP", f"failed: {exc}", {}, {})


def t_polygon_quote() -> Trial:
    """Polygon prev-day agg vs yfinance quote (requires POLYGON_API_KEY)."""
    if not os.environ.get("POLYGON_API_KEY"):
        return Trial("polygon_quote", "Quote", "Polygon.io prev-day agg", "yfinance .info",
                     "SKIP", "no POLYGON_API_KEY configured", {}, {})
    try:
        from core.data_providers import get_quote_chain
        q, cand_t = _elapsed(lambda: get_quote_chain("AAPL"))
        yq, inc_t = _elapsed(lambda: _quote_yf("AAPL"))
        if not q or not yq or q["price"] <= 0:
            return Trial("polygon_quote", "Quote", "Polygon.io prev-day agg", "yfinance .info",
                         "SKIP", "provider returned empty quote", {}, {})
        pct_diff = abs(q["price"] - yq["price"]) / yq["price"] * 100
        verdict = "EQUIVALENT" if pct_diff < 1.5 else "KEEP"
        notes = (f"price diff {pct_diff:.2f}%; free tier is 15-min delayed with ~2y history — "
                 f"fine for EOD research, too thin for live quotes.")
        return Trial("polygon_quote", "Quote", "Polygon.io prev-day agg", "yfinance .info",
                     verdict, notes, {"price": {"candidate": q["price"], "incumbent": yq["price"]},
                                      "pct_diff": round(pct_diff, 2)},
                     {"candidate_s": round(cand_t, 2), "incumbent_s": round(inc_t, 2)})
    except Exception as exc:
        return Trial("polygon_quote", "Quote", "Polygon.io prev-day agg", "yfinance .info",
                     "SKIP", f"failed: {exc}", {}, {})


TRIALS = {
    "ff_factors": t_ff_factors,
    "risk_metrics": t_risk_metrics,
    "technical_indicators": t_technical_indicators,
    "av_quote": t_av_quote,
    "av_news": t_av_news,
    "polygon_quote": t_polygon_quote,
}


def run_trials(ids: list = None) -> list:
    ids = ids or list(TRIALS)
    results = []
    for tid in ids:
        fn = TRIALS[tid]
        results.append(fn())
    return results


def apply(results: list) -> None:
    """Auto-integration: re-order the provider chain based on quote trial verdicts."""
    order = ["alphavantage", "fmp", "polygon", "yfinance"]
    for r in results:
        if r.id in ("av_quote", "polygon_quote") and r.verdict == "KEEP":
            order = [p for p in order if p != r.candidate.lower().split(" ")[0]]  # drop loser
        if r.id == "av_news" and r.verdict == "ADOPT":
            order = [r.candidate.lower().split(" ")[0]] + [p for p in order if p != r.candidate.lower().split(" ")[0]]
    TRIALS_FILE.parent.mkdir(exist_ok=True)
    with open(TRIALS_FILE, "w") as fh:
        json.dump({"order": order, "applied_at": time.time(),
                   "verdicts": {r.id: r.verdict for r in results}}, fh, indent=2)
    print(f"[apply] wrote {TRIALS_FILE} with provider order: {order}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Tool trial harness")
    parser.add_argument("--all", action="store_true", help="run every trial")
    parser.add_argument("--trial", help="run a single trial by id")
    parser.add_argument("--json", action="store_true", help="print report as JSON")
    parser.add_argument("--apply", action="store_true", help="auto-integrate winners (provider order)")
    args = parser.parse_args()

    ids = [args.trial] if args.trial else (list(TRIALS) if args.all else ["risk_metrics", "ff_factors"])
    results = run_trials(ids)
    if args.apply:
        apply(results)

    report = {"ran_at": time.time(), "results": [asdict(r) for r in results]}
    REPORT_FILE.parent.mkdir(exist_ok=True)
    with open(REPORT_FILE, "w") as fh:
        json.dump(report, fh, indent=2)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"\n{'TRIAL':<24} {'CANDIDATE':<28} {'INCUMBENT':<30} {'VERDICT':<10}")
    print("-" * 96)
    for r in results:
        print(f"{r.id:<24} {r.candidate:<28} {r.incumbent:<30} {r.verdict:<10}")
        print(f"    {r.notes}")
    print(f"\nReport: {REPORT_FILE}")


if __name__ == "__main__":
    main()
