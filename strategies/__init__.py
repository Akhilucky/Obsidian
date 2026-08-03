"""
Trading Strategies Package
===========================

Comprehensive trading strategy modules for the Bloomberg-killer terminal.

Modules:
--------
- trend_following: Moving average crossovers, breakout, momentum strategies
- mean_reversion: Bollinger Bands, RSI reversals, pair trading
- market_making: Bid-ask spread capture, order book analysis, HFT
- sentiment: News analysis, social media signals, options flow
- ml_models: LSTM, Random Forest, Gradient Boosting, DQN
- multi_factor: Factor scoring, portfolio optimization

Usage:
------
```python
from strategies import TrendFollowingSuite, MeanReversionSuite
from strategies import MarketMakingSuite, SentimentSuite
from strategies import MLStrategySuite, MultiFactorStrategy

# Trend Following
trend = TrendFollowingSuite()
signals = trend.get_combined_signals(data)

# Mean Reversion
mr = MeanReversionSuite()
results = mr.analyze_single_asset(data, 'bollinger')

# Market Making
mm = MarketMakingSuite()
bid, ask = mm.generate_quotes('basic', order_book)

# Sentiment Analysis
sent = SentimentSuite()
signal = sent.get_combined_signal('AAPL')

# Machine Learning
ml = MLStrategySuite()
ml.create_default_strategies()
ml.train_all(data)
ensemble = ml.get_ensemble_signal(data)

# Multi-Factor
mf = MultiFactorStrategy()
selected = mf.select_stocks(universe)
weights = mf.construct_portfolio(selected, price_data)
```
"""

# Trend Following
from .trend_following import (
    Signal,
    TradeSignal,
    BaseStrategy,
    SMAStrategy,
    EMAStrategy,
    TripleMAStrategy,
    BreakoutStrategy,
    TurtleBreakout,
    RSIMomentumStrategy,
    MACDMomentumStrategy,
    ADXTrendStrength,
    TrendFollowingEnsemble,
)

# Mean Reversion
from .mean_reversion import (
    MeanReversionSuite,
    BollingerBandReversion,
    RSIReversion,
    PairTradingStrategy,
    MeanReversionConfig,
    PairConfig,
    create_mean_reversion_strategy
)

# Market Making
from .market_making import (
    MarketMakingSuite,
    BasicMarketMaker,
    OrderBookImbalanceStrategy,
    MicroTradingStrategy,
    SpreadCaptureStrategy,
    MarketMakingConfig,
    OrderBook,
    OrderBookLevel,
    Quote,
    Order,
    OrderSide,
    OrderType,
)

# Sentiment
from .sentiment import (
    SentimentSuite,
    NewsSentimentStrategy,
    SocialSentimentStrategy,
    OptionsFlowStrategy,
    SentimentConfig,
    NewsArticle,
    SocialPost,
    OptionsFlow,
    SentimentScore,
    create_sentiment_strategy
)

# Machine Learning
from .ml_models import (
    MLStrategySuite,
    LSTMStrategy,
    RandomForestStrategy,
    GradientBoostingStrategy,
    DQNStrategy,
    FeatureEngineer,
    MLConfig,
    PredictionType,
    SignalType,
    create_ml_strategy
)

# Multi-Factor
from .multi_factor import (
    MultiFactorStrategy,
    MultiFactorSuite,
    FactorCalculator,
    FactorRanker,
    PortfolioOptimizer,
    MultiFactorConfig,
    FactorCategory,
    StockData,
    create_multi_factor_strategy
)

__all__ = [
    # Trend Following
    'Signal',
    'TradeSignal',
    'BaseStrategy',
    'SMAStrategy',
    'EMAStrategy',
    'TripleMAStrategy',
    'BreakoutStrategy',
    'TurtleBreakout',
    'RSIMomentumStrategy',
    'MACDMomentumStrategy',
    'ADXTrendStrength',
    'TrendFollowingEnsemble',
    
    # Mean Reversion
    'MeanReversionSuite',
    'BollingerBandReversion',
    'RSIReversion',
    'PairTradingStrategy',
    'MeanReversionConfig',
    'PairConfig',
    'create_mean_reversion_strategy',
    
    # Market Making
    'MarketMakingSuite',
    'BasicMarketMaker',
    'OrderBookImbalanceStrategy',
    'MicroTradingStrategy',
    'SpreadCaptureStrategy',
    'MarketMakingConfig',
    'OrderBook',
    'OrderBookLevel',
    'Quote',
    'Order',
    'OrderSide',
    'OrderType',
    
    # Sentiment
    'SentimentSuite',
    'NewsSentimentStrategy',
    'SocialSentimentStrategy',
    'OptionsFlowStrategy',
    'SentimentConfig',
    'NewsArticle',
    'SocialPost',
    'OptionsFlow',
    'SentimentScore',
    'create_sentiment_strategy',
    
    # Machine Learning
    'MLStrategySuite',
    'LSTMStrategy',
    'RandomForestStrategy',
    'GradientBoostingStrategy',
    'DQNStrategy',
    'FeatureEngineer',
    'MLConfig',
    'PredictionType',
    'SignalType',
    'create_ml_strategy',
    
    # Multi-Factor
    'MultiFactorStrategy',
    'MultiFactorSuite',
    'FactorCalculator',
    'FactorRanker',
    'PortfolioOptimizer',
    'MultiFactorConfig',
    'FactorCategory',
    'StockData',
    'create_multi_factor_strategy',
]

# Version
__version__ = '1.0.0'
