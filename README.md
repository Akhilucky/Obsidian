# 🏦 Obsidian Terminal

> **An open-source, institutional-grade quantitative trading platform. Combines a React terminal (Bloomberg × Linear aesthetic), ML-powered predictions, multi-agent cooperative reasoning, C++ compute kernels, Java portfolio optimization, and real-time market data (US + India) — all free.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What It Does

Obsidian Terminal is a full-stack quantitative finance platform that ingests market data, engineers features, runs ML models, evaluates risk, and presents everything through a premium React terminal. It supports **US markets** (NYSE, NASDAQ) and **Indian markets** (NSE, BSE) out of the box.

### Core Capabilities

| Area | What's Inside |
|------|---------------|
| **Data Ingestion** | Yahoo Finance, FRED macro, CoinGecko crypto, NSE/BSE India — 700+ feeds with parquet caching |
| **10-Agent Pipeline** | Data → Quality → Features → Regime → Model → Decision → Risk → Scenario → Monitor → Lifecycle |
| **ML Predictions** | Random Forest + Gradient Boosting ensemble, regime-conditioned, walk-forward validated |
| **Feature Store** | 100+ technical indicators with versioning and point-in-time correctness |
| **Portfolio Optimization** | Mean-Variance, Max Sharpe, Risk Parity, HRP, Black-Litterman — Java closed-form engine |
| **Risk Management** | VaR, CVaR, stress testing, position limits, 2/20 fee model |
| **Indian Markets** | NIFTY 50, SENSEX, NIFTY IT, NIFTY Bank — full NSE/BSE support with `.NS`/`.BO` suffixes |
| **Crypto** | 100+ exchanges via CCXT, DeFi analytics, on-chain metrics |
| **Terminal** | React + Next.js institutional terminal — dark-first, animated charts, command palette (⌘K), page transitions, C++ kernels for signal computation & Monte Carlo |

---

## Terminal Pages

| Page | Features |
|-----|----------|
| **Overview** | Focus stock chart (pick any ticker, persisted), market indices strip, VIX & 10Y, market movers, watchlist quotes |
| **Markets** | WEI-style global watch: US / Asia-Pacific / Europe / Commodities & FX (28 instruments), gainers/losers/most-active |
| **Screener** | EQS-style stock screener: US/India, sector, min market cap, max P/E filters |
| **Analysis** | Per-ticker deep dive — SMA/EMA/Bollinger overlays, RSI & MACD panels, ATR, comparative total return (COMP) |
| **Signals** | Composite alpha scores, conviction bars, strategy bias gauge, alert feed |
| **Portfolio** | Aladdin-style analytics — historical & parametric VaR, 5 stress scenarios, compliance violations, sector exposure, editable holdings |
| **Quant Lab** | Tool-trial harness (benchmarks candidate libraries vs ours, auto-integrates winners) + ML strategy lab (walk-forward model comparison vs the RF/GB ensemble) |
| **Stock Profile** | DES + FA/RV/DVD/OMON — fundamentals, peer comparison, dividend history, options chain |
| **India** | NIFTY 50 / SENSEX indices, NSE/BSE stock browser, popular Indian stocks grid (batched quotes) |
| **Research** | Strategy library with Sharpe/win-rate, factor exposures, market performance |
| **Agents** | Pipeline architecture view, per-agent health inspector |
| **Settings** | Cache management, system info, runtime status |

---

## Quick Start

```bash
# Clone
git clone https://github.com/Akhilucky/Obsidian.git
cd Obsidian

# Install Python deps + native kernels + frontend
pip install -r requirements.txt
make cpp          # C++ kernels + Java optimizer
cd web && npm install && cd ..

# Terminal 1 — API server (port 8000)
python api/server.py

# Terminal 2 — React frontend (port 3000)
cd web && npm run dev
```

The terminal opens at `http://localhost:3000`.

### Agent Pipeline

```bash
# Single pass
python run.py --agents --symbols AAPL MSFT GOOGL

# Continuous mode (every 5 minutes)
python run.py --agents --continuous --interval 300

# Indian markets
python run.py --agents --symbols RELIANCE.NS TCS.NS INFY.NS
```

---

## Data Providers

Quotes, history and news are served by a key-gated provider chain — the first configured
provider that returns data wins, with **yfinance as the always-available fallback**:

| Provider | Env var | Free tier | Notes |
|----------|---------|-----------|-------|
| Alpha Vantage | `ALPHA_VANTAGE_KEY` | 25 req/day | EOD quotes, Fama-French-style analytics, **News & Sentiment** with vendor labels |
| Financial Modeling Prep | `FMP_API_KEY` | 250 req/day | Quotes + fundamentals (150+ endpoints), best free fundamentals source |
| Polygon.io / Massive | `POLYGON_API_KEY` | 5 req/min | EOD, ~2y history, 15-min delayed on free tier |
| yfinance | — | unlimited | Default; used when no keys configured or providers fail |

Free-tier daily quotas are tracked in `data_cache/provider_quota.json` so the chain
degrades gracefully instead of burning quota. Order is overridable via `DATA_PROVIDER_ORDER`
or `data_cache/provider_trials.json` (written by the tool-trial harness `--apply`).

---

## Tool Audit (from `research/tool_trials.py`)

Every candidate tool is benchmarked against our current implementation — run
`python research/tool_trials.py --all` or see the Quant Lab page:

| Tool | Verdict | Notes |
|------|---------|-------|
| **pandas-datareader (Fama-French)** | ✅ ADOPT | Keyless 5-factor daily data (Mkt-RF, SMB, HML, RMW, CMA) — now a feature source in the ML lab |
| **Alpha Vantage** (quote/news) | ✅ ADOPT (with key) | EOD + vendor news sentiment labels; wired as chain priority |
| **FMP / Polygon** | ✅ ADOPT (with key) | Wired into the chain; free tiers are research-grade |
| **quantstats** | ⚠️ EQUIVALENT/KEEP | Matches our metrics; keep lightweight `RiskMetrics` in the API, quantstats as validation layer |
| **ta** (indicators) | KEEP | Wilder vs rolling-mean RSI (corr 0.90); ours stays |
| **pandas-datareader (yahoo/stooq)** | ✗ SKIP | Sources decommissioned by the library |
| OpenBB, FinRL, Backtrader, bt, TA-Lib, DBs (Timescale/QuestDB/ClickHouse/Influx), Superset, Grafana, Dash, news scrapers | not integrated | Heavy dependency or infra cost vs our terminal; revisit per need |

---

## Architecture

```
Obsidian/
├── agents/                  # 10-agent cooperative pipeline
│   ├── orchestrator.py      # Pipeline runner (parallel per-symbol)
│   ├── base_agent.py        # Abstract base with observability
│   ├── data_agent.py        # Data ingestion with retry + fallback
│   ├── quality_agent.py     # Data validation + confidence scoring
│   ├── feature_agent.py     # 100+ feature computation
│   ├── regime_agent.py      # Market regime detection (Hurst + vol)
│   ├── model_agent.py       # ML ensemble (RF + GB)
│   ├── decision_agent.py    # Trade idea generation
│   ├── risk_agent.py        # CVaR limits, position checks
│   ├── scenario_agent.py    # Stress testing
│   ├── monitor_agent.py     # Drift detection + alerts
│   └── lifecycle_agent.py   # Strategy stage management
├── core/                    # Infrastructure layer
│   ├── event_bus.py         # Immutable event system (singleton, thread-safe)
│   ├── data_ingest.py       # Multi-source data pipeline
│   ├── feature_store.py     # Versioned feature warehouse
│   ├── fast_kernels.py      # C++ compute kernels (ctypes loader)
│   └── java_optimizer.py    # Java mean-variance optimizer bridge
├── data/
│   └── indian_markets.py    # NSE/BSE data fetcher
├── api/
│   └── server.py            # Flask JSON API (port 8000)
├── web/                     # React terminal (Next.js, port 3000)
│   ├── app/                 # 8 pages: overview/analysis/signals/india/...
│   ├── components/          # Shell, charts, UI kit
│   └── lib/                 # API client, types, formatting
├── cpp/
│   └── obsidian_core.cpp    # C++ kernels (SMA signals, Monte Carlo)
├── java/
│   └── PortfolioOptimizer.java  # Closed-form MV optimization
├── dashboard/
│   └── app_streamlit.py     # Legacy Streamlit dashboard (optional)
├── strategies/              # 6 strategy templates
├── analytics/               # Sentiment, options, alerts, technical
├── portfolio/               # Advanced optimization
├── research/                # ML models, backtesting, screening
├── pyproject.toml           # Package configuration
├── run.py                   # Entry point
└── requirements*.txt        # Tiered dependencies
```

---

## Indian Markets

Full support for NSE and BSE through Yahoo Finance suffixes (`.NS` for NSE, `.BO` for BSE):

```python
from data.indian_markets import IndianMarketDataFetcher

fetcher = IndianMarketDataFetcher()

# Fetch NIFTY 50 stocks
nifty50_data = fetcher.fetch_universe("nifty50")

# Fetch index data
nifty = fetcher.fetch_index("^NSEI")    # NIFTY 50
sensex = fetcher.fetch_index("^BSESN")  # SENSEX

# Individual stock
reliance = fetcher.fetch_stock("RELIANCE", exchange="NSE")
```

Pre-built universes: `nifty50`, `sensex30`, `nifty_it`, `nifty_bank`, `indian_etf`

---

## Usage Examples

### Portfolio Optimization

```python
from core.risk_management import PortfolioOptimizer, OptimizationMethod
import yfinance as yf

data = yf.download(['AAPL', 'MSFT', 'GOOGL', 'AMZN'], period='2y')['Adj Close']
returns = data.pct_change().dropna()

optimizer = PortfolioOptimizer(returns)
weights = optimizer.optimize(OptimizationMethod.MAX_SHARPE)
stats = optimizer.portfolio_stats(weights)

print(f"Sharpe: {stats['sharpe_ratio']:.2f}")
print(f"Max DD: {stats['max_drawdown']*100:.1f}%")
```

### Agent Pipeline

```python
from agents.orchestrator import AgentOrchestrator

orch = AgentOrchestrator()
results = orch.run_pipeline(["AAPL", "RELIANCE.NS", "BTC-USD"])

for symbol, result in results.items():
    d = result.to_dict()
    print(f"{symbol}: signal={d['data'].get('signal')}, confidence={d['data'].get('confidence')}")
```

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Data** | yfinance, CoinGecko, FRED, NSE/BSE via Yahoo |
| **ML/AI** | scikit-learn, XGBoost, LightGBM |
| **NLP** | VADER, TextBlob, Transformers |
| **Optimization** | Java closed-form engine, SciPy, CVXPY |
| **Compute Kernels** | C++ (ctypes) — SMA signals, Monte Carlo, max drawdown |
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind, Framer Motion, Recharts |
| **Backend** | Flask JSON API (port 8000) |
| **Indian Markets** | Yahoo Finance (.NS/.BO), NIFTY/SENSEX indices |

---

## Competitive Advantages

### vs Bloomberg Terminal ($24,000/year)
- **Free & Open Source** — No subscription required; all data comes from free public sources (Yahoo Finance, etc.)
- **Indian Markets** — NSE/BSE support included
- **Customizable ML** — Modify models, add new features
- **Python-Native** — Integrate with any library in the ecosystem
- **Terminal Functionality** — DES (focus profile), FA/RV/DVD/OMON, WEI, MOV, EQS, COMP, ERN, plus Aladdin-style VaR, stress testing & compliance checks

### vs Aladdin ($100k+/year)
- **Transparent Risk Math** — Historical + parametric VaR, 5 stress scenarios, concentration limits — all implemented in readable Python, not a black box
- **Editable Holdings** — Add/remove positions with average cost, live P&L, sector exposure
- **Zero Cost** — Same risk analytics for the price of free

### vs Other Open Source Terminals
- **Multi-Agent Architecture** — 10 specialized agents with event-driven communication
- **Indian Market Support** — NIFTY 50, SENSEX, sector indices
- **Parallel Pipeline** — Concurrent per-symbol processing
- **Production ML** — Walk-forward validation, regime conditioning
- **Risk Management** — CVaR, HRP, Black-Litterman, sector neutralization
- **Native Performance** — C++ kernels (114x faster signal generation) and a Java optimizer, with automatic Python fallbacks

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance API wrapper
- [CCXT](https://github.com/ccxt/ccxt) — Cryptocurrency exchange library
- [Riskfolio-lib](https://github.com/dcajasn/Riskfolio-Lib) — Portfolio optimization
- [Next.js](https://nextjs.org/) — React terminal framework
- [Recharts](https://recharts.org/) — Charting library
- [Framer Motion](https://www.framer.com/motion/) — UI motion

---

<p align="center">
  <b>Built for the quantitative finance community</b>
</p>
