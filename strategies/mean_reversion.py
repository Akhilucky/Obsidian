"""
Mean Reversion Trading Strategies
==================================
Statistical mean-reverting strategies including Bollinger Bands,
RSI reversals, and pair trading / statistical arbitrage.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ReversalSignal(Enum):
    """Mean reversion signal types."""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategies."""
    lookback_period: int = 20
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.05
    max_holding_days: int = 10
    position_size: float = 1.0
    use_volume_filter: bool = True
    min_volume_ratio: float = 1.0  # Relative to average


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position: int  # 1 for long, -1 for short
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str


class MeanReversionStrategy(ABC):
    """Base class for mean reversion strategies."""
    
    def __init__(self, config: MeanReversionConfig = None):
        self.config = config or MeanReversionConfig()
        self.signals = pd.DataFrame()
        self.trades: List[TradeResult] = []
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators."""
        pass
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest on historical data."""
        df = self.calculate_indicators(data.copy())
        df = self.generate_signals(df)
        
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        holding_days = 0
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            if position == 0:
                # Check for entry
                if current.get('signal', 0) in [ReversalSignal.BUY.value, ReversalSignal.STRONG_BUY.value]:
                    position = 1
                    entry_price = current['close']
                    entry_date = current.name
                    holding_days = 0
                elif current.get('signal', 0) in [ReversalSignal.SELL.value, ReversalSignal.STRONG_SELL.value]:
                    position = -1
                    entry_price = current['close']
                    entry_date = current.name
                    holding_days = 0
            else:
                holding_days += 1
                exit_reason = None
                exit_price = current['close']
                
                # Check exit conditions
                pnl_pct = (exit_price / entry_price - 1) * position
                
                # Stop loss
                if pnl_pct <= -self.config.stop_loss_pct:
                    exit_reason = 'stop_loss'
                # Take profit
                elif pnl_pct >= self.config.take_profit_pct:
                    exit_reason = 'take_profit'
                # Max holding period
                elif holding_days >= self.config.max_holding_days:
                    exit_reason = 'max_holding'
                # Mean reversion complete (price returned to mean)
                elif position == 1 and current.get('exit_signal', 0) >= 0:
                    exit_reason = 'mean_reversion'
                elif position == -1 and current.get('exit_signal', 0) <= 0:
                    exit_reason = 'mean_reversion'
                
                if exit_reason:
                    pnl = (exit_price - entry_price) * position
                    trades.append(TradeResult(
                        entry_date=entry_date,
                        exit_date=current.name,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position=position,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        holding_days=holding_days,
                        exit_reason=exit_reason
                    ))
                    position = 0
        
        self.trades = trades
        return self._calculate_metrics(trades, df)
    
    def _calculate_metrics(self, trades: List[TradeResult], data: pd.DataFrame) -> Dict:
        """Calculate backtest performance metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        
        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # Calculate equity curve
        equity = [1.0]
        for pnl in pnls:
            equity.append(equity[-1] * (1 + pnl))
        
        # Max drawdown
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'avg_pnl_pct': np.mean(pnls) if pnls else 0,
            'total_pnl_pct': sum(pnls),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'sharpe_ratio': np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0,
            'max_drawdown': max_dd,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'avg_holding_days': np.mean([t.holding_days for t in trades]),
            'final_equity': equity[-1]
        }


class BollingerBandReversion(MeanReversionStrategy):
    """
    Bollinger Band Mean Reversion Strategy
    
    Buys when price touches lower band (oversold)
    Sells when price touches upper band (overbought)
    Exits when price returns to middle band (mean)
    """
    
    def __init__(self, 
                 period: int = 20,
                 num_std: float = 2.0,
                 config: MeanReversionConfig = None):
        super().__init__(config)
        self.period = period
        self.num_std = num_std
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df = data.copy()
        
        # Calculate bands
        df['bb_middle'] = df['close'].rolling(window=self.period).mean()
        df['bb_std'] = df['close'].rolling(window=self.period).std()
        df['bb_upper'] = df['bb_middle'] + (self.num_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.num_std * df['bb_std'])
        
        # Bandwidth and %B
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Z-score from mean
        df['z_score'] = (df['close'] - df['bb_middle']) / df['bb_std']
        
        # Volume filter
        if self.config.use_volume_filter and 'volume' in df.columns:
            df['vol_sma'] = df['volume'].rolling(window=self.period).mean()
            df['vol_ratio'] = df['volume'] / df['vol_sma']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Bollinger Bands."""
        df = data.copy()
        df['signal'] = ReversalSignal.NEUTRAL.value
        df['exit_signal'] = 0
        
        for i in range(self.period, len(df)):
            z = df.iloc[i]['z_score']
            pct_b = df.iloc[i]['bb_pct_b']
            
            # Volume filter
            vol_ok = True
            if self.config.use_volume_filter and 'vol_ratio' in df.columns:
                vol_ok = df.iloc[i]['vol_ratio'] >= self.config.min_volume_ratio
            
            # Entry signals
            if z <= -self.config.entry_threshold and vol_ok:
                df.iloc[i, df.columns.get_loc('signal')] = ReversalSignal.STRONG_BUY.value
            elif z <= -1.5 and vol_ok:
                df.iloc[i, df.columns.get_loc('signal')] = ReversalSignal.BUY.value
            elif z >= self.config.entry_threshold and vol_ok:
                df.iloc[i, df.columns.get_loc('signal')] = ReversalSignal.STRONG_SELL.value
            elif z >= 1.5 and vol_ok:
                df.iloc[i, df.columns.get_loc('signal')] = ReversalSignal.SELL.value
            
            # Exit signals (price returning to mean)
            if abs(z) <= self.config.exit_threshold:
                df.iloc[i, df.columns.get_loc('exit_signal')] = 1 if z > 0 else -1
        
        return df


class RSIReversion(MeanReversionStrategy):
    """
    RSI Mean Reversion Strategy
    
    Uses RSI oversold/overbought levels for entry
    with divergence confirmation for stronger signals.
    """
    
    def __init__(self,
                 rsi_period: int = 14,
                 oversold: float = 30,
                 overbought: float = 70,
                 use_divergence: bool = True,
                 config: MeanReversionConfig = None):
        super().__init__(config)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.use_divergence = use_divergence
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and divergence indicators."""
        df = data.copy()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI moving average
        df['rsi_ma'] = df['rsi'].rolling(window=9).mean()
        
        # Stochastic RSI
        rsi_min = df['rsi'].rolling(window=self.rsi_period).min()
        rsi_max = df['rsi'].rolling(window=self.rsi_period).max()
        df['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min)
        
        # Price and RSI divergence detection
        if self.use_divergence:
            df = self._detect_divergence(df)
        
        return df
    
    def _detect_divergence(self, data: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """Detect bullish and bearish divergences."""
        df = data.copy()
        df['bullish_divergence'] = False
        df['bearish_divergence'] = False
        
        for i in range(lookback * 2, len(df)):
            # Look for local lows in price
            price_window = df['close'].iloc[i-lookback:i+1]
            rsi_window = df['rsi'].iloc[i-lookback:i+1]
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (df['close'].iloc[i] < price_window.min() and 
                df['rsi'].iloc[i] > rsi_window.min() and
                df['rsi'].iloc[i] < self.oversold + 10):
                df.iloc[i, df.columns.get_loc('bullish_divergence')] = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (df['close'].iloc[i] > price_window.max() and 
                df['rsi'].iloc[i] < rsi_window.max() and
                df['rsi'].iloc[i] > self.overbought - 10):
                df.iloc[i, df.columns.get_loc('bearish_divergence')] = True
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on RSI."""
        df = data.copy()
        df['signal'] = ReversalSignal.NEUTRAL.value
        df['exit_signal'] = 0
        
        for i in range(self.rsi_period + 10, len(df)):
            rsi = df.iloc[i]['rsi']
            prev_rsi = df.iloc[i-1]['rsi']
            
            # Strong signals with divergence
            if self.use_divergence:
                if df.iloc[i].get('bullish_divergence', False):
                    df.iloc[i, df.columns.get_loc('signal')] = ReversalSignal.STRONG_BUY.value
                    continue
                elif df.iloc[i].get('bearish_divergence', False):
                    df.iloc[i, df.columns.get_loc('signal')] = ReversalSignal.STRONG_SELL.value
                    continue
            
            # Regular RSI signals
            if rsi <= self.oversold and prev_rsi > self.oversold:
                df.iloc[i, df.columns.get_loc('signal')] = ReversalSignal.BUY.value
            elif rsi >= self.overbought and prev_rsi < self.overbought:
                df.iloc[i, df.columns.get_loc('signal')] = ReversalSignal.SELL.value
            
            # Exit when RSI crosses middle
            if 45 <= rsi <= 55:
                df.iloc[i, df.columns.get_loc('exit_signal')] = 1 if rsi > 50 else -1
        
        return df


@dataclass
class PairConfig:
    """Configuration for pair trading."""
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    zscore_stop: float = 4.0
    lookback: int = 60
    hedge_ratio_window: int = 30
    min_correlation: float = 0.7
    min_cointegration_pvalue: float = 0.05


class PairTradingStrategy:
    """
    Statistical Arbitrage / Pair Trading Strategy
    
    Identifies cointegrated pairs and trades the spread
    when it deviates from its mean.
    """
    
    def __init__(self, config: PairConfig = None):
        self.config = config or PairConfig()
        self.pairs: List[Tuple[str, str]] = []
        self.hedge_ratios: Dict[str, float] = {}
        self.spread_stats: Dict[str, Dict] = {}
    
    def find_cointegrated_pairs(self, 
                                 prices: pd.DataFrame,
                                 p_value_threshold: float = 0.05) -> List[Tuple[str, str, float]]:
        """
        Find cointegrated pairs from a universe of stocks.
        
        Args:
            prices: DataFrame with stock prices as columns
            p_value_threshold: Maximum p-value for cointegration
            
        Returns:
            List of (stock1, stock2, p_value) tuples
        """
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            logger.warning("statsmodels not installed. Using simplified cointegration test.")
            return self._simplified_pair_search(prices)
        
        n = len(prices.columns)
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = prices.columns[i]
                stock2 = prices.columns[j]
                
                s1 = prices[stock1].dropna()
                s2 = prices[stock2].dropna()
                
                # Align the series
                common_idx = s1.index.intersection(s2.index)
                if len(common_idx) < self.config.lookback:
                    continue
                
                s1 = s1.loc[common_idx]
                s2 = s2.loc[common_idx]
                
                # Check correlation first
                corr = s1.corr(s2)
                if abs(corr) < self.config.min_correlation:
                    continue
                
                # Cointegration test
                try:
                    _, p_value, _ = coint(s1, s2)
                    if p_value < p_value_threshold:
                        pairs.append((stock1, stock2, p_value))
                        logger.info(f"Found pair: {stock1}/{stock2}, p-value: {p_value:.4f}")
                except Exception as e:
                    logger.debug(f"Cointegration test failed for {stock1}/{stock2}: {e}")
        
        # Sort by p-value
        pairs.sort(key=lambda x: x[2])
        self.pairs = [(p[0], p[1]) for p in pairs]
        
        return pairs
    
    def _simplified_pair_search(self, prices: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Simplified pair search using correlation when statsmodels unavailable."""
        n = len(prices.columns)
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = prices.columns[i]
                stock2 = prices.columns[j]
                
                corr = prices[stock1].corr(prices[stock2])
                if abs(corr) >= self.config.min_correlation:
                    # Use 1 - correlation as pseudo p-value
                    pairs.append((stock1, stock2, 1 - abs(corr)))
        
        pairs.sort(key=lambda x: x[2])
        self.pairs = [(p[0], p[1]) for p in pairs]
        return pairs
    
    def calculate_hedge_ratio(self, 
                               s1: pd.Series, 
                               s2: pd.Series,
                               method: str = 'ols') -> float:
        """
        Calculate the hedge ratio between two securities.
        
        Methods:
            - ols: Ordinary Least Squares
            - rolling: Rolling OLS
            - kalman: Kalman Filter (if available)
        """
        if method == 'ols':
            # Simple OLS: s1 = beta * s2 + alpha
            cov = np.cov(s1, s2)
            hedge_ratio = cov[0, 1] / cov[1, 1]
        elif method == 'rolling':
            # Use last n observations
            n = self.config.hedge_ratio_window
            s1_recent = s1.iloc[-n:]
            s2_recent = s2.iloc[-n:]
            cov = np.cov(s1_recent, s2_recent)
            hedge_ratio = cov[0, 1] / cov[1, 1]
        else:
            # Default to OLS
            cov = np.cov(s1, s2)
            hedge_ratio = cov[0, 1] / cov[1, 1]
        
        return hedge_ratio
    
    def calculate_spread(self,
                          s1: pd.Series,
                          s2: pd.Series,
                          hedge_ratio: float = None) -> pd.Series:
        """Calculate the spread between two securities."""
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(s1, s2)
        
        spread = s1 - hedge_ratio * s2
        return spread
    
    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate z-score of the spread."""
        mean = spread.rolling(window=self.config.lookback).mean()
        std = spread.rolling(window=self.config.lookback).std()
        zscore = (spread - mean) / std
        return zscore
    
    def generate_pair_signals(self,
                               prices1: pd.Series,
                               prices2: pd.Series,
                               pair_name: str = None) -> pd.DataFrame:
        """
        Generate trading signals for a pair.
        
        Returns DataFrame with:
            - spread
            - zscore  
            - signal (1 for long spread, -1 for short spread, 0 for neutral)
            - position
        """
        # Align series
        common_idx = prices1.index.intersection(prices2.index)
        s1 = prices1.loc[common_idx]
        s2 = prices2.loc[common_idx]
        
        # Calculate hedge ratio and spread
        hedge_ratio = self.calculate_hedge_ratio(s1, s2)
        spread = self.calculate_spread(s1, s2, hedge_ratio)
        zscore = self.calculate_zscore(spread)
        
        # Store stats
        if pair_name:
            self.hedge_ratios[pair_name] = hedge_ratio
            self.spread_stats[pair_name] = {
                'mean': spread.mean(),
                'std': spread.std(),
                'half_life': self._calculate_half_life(spread)
            }
        
        # Generate signals
        signals = pd.DataFrame(index=common_idx)
        signals['price1'] = s1
        signals['price2'] = s2
        signals['hedge_ratio'] = hedge_ratio
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['signal'] = 0
        signals['position'] = 0
        
        position = 0
        
        for i in range(self.config.lookback, len(signals)):
            z = signals.iloc[i]['zscore']
            
            if position == 0:
                # Entry signals
                if z <= -self.config.zscore_entry:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1  # Long spread
                    position = 1
                elif z >= self.config.zscore_entry:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1  # Short spread
                    position = -1
            else:
                # Exit signals
                if position == 1 and z >= -self.config.zscore_exit:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
                elif position == -1 and z <= self.config.zscore_exit:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
                # Stop loss
                elif position == 1 and z <= -self.config.zscore_stop:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
                elif position == -1 and z >= self.config.zscore_stop:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
            
            signals.iloc[i, signals.columns.get_loc('position')] = position
        
        return signals
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life using OLS."""
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # Remove NaN
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()
        
        # Align
        common_idx = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag.loc[common_idx]
        spread_diff = spread_diff.loc[common_idx]
        
        # OLS: spread_diff = lambda * spread_lag
        if len(spread_lag) > 0:
            lambda_coef = np.dot(spread_lag, spread_diff) / np.dot(spread_lag, spread_lag)
            if lambda_coef < 0:
                half_life = -np.log(2) / lambda_coef
                return half_life
        
        return float('inf')
    
    def backtest_pair(self,
                       prices1: pd.Series,
                       prices2: pd.Series,
                       capital: float = 100000) -> Dict:
        """
        Backtest a pair trading strategy.
        
        Args:
            prices1: First security prices
            prices2: Second security prices
            capital: Starting capital
            
        Returns:
            Performance metrics dictionary
        """
        signals = self.generate_pair_signals(prices1, prices2)
        
        # Track P&L
        pnl = []
        position = 0
        entry_spread = 0
        
        for i in range(1, len(signals)):
            current = signals.iloc[i]
            prev = signals.iloc[i-1]
            
            if prev['position'] != 0:
                # Calculate daily P&L
                spread_change = current['spread'] - signals.iloc[i-1]['spread']
                daily_pnl = spread_change * prev['position']
                pnl.append(daily_pnl)
            else:
                pnl.append(0)
        
        # Calculate metrics
        pnl = np.array(pnl)
        cumulative_pnl = np.cumsum(pnl)
        
        # Drawdown
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = peak - cumulative_pnl
        max_drawdown = np.max(drawdown)
        
        # Number of trades
        position_changes = np.diff(signals['position'].values)
        num_trades = np.sum(position_changes != 0) // 2
        
        return {
            'total_pnl': cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0,
            'sharpe_ratio': np.mean(pnl) / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': np.sum(pnl > 0) / len(pnl[pnl != 0]) if np.any(pnl != 0) else 0,
            'avg_trade_pnl': np.mean(pnl[pnl != 0]) if np.any(pnl != 0) else 0,
            'half_life': self.spread_stats.get(None, {}).get('half_life', float('inf'))
        }


class MeanReversionSuite:
    """
    Unified interface for all mean reversion strategies.
    """
    
    def __init__(self):
        self.strategies = {
            'bollinger': BollingerBandReversion(),
            'rsi': RSIReversion(),
        }
        self.pair_strategy = PairTradingStrategy()
    
    def add_strategy(self, name: str, strategy: MeanReversionStrategy):
        """Add a custom strategy."""
        self.strategies[name] = strategy
    
    def analyze_single_asset(self,
                              data: pd.DataFrame,
                              strategy_name: str = None) -> Dict:
        """
        Run mean reversion analysis on a single asset.
        
        Args:
            data: OHLCV DataFrame
            strategy_name: Specific strategy to use, or None for all
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        strategies_to_run = ([strategy_name] if strategy_name 
                            else list(self.strategies.keys()))
        
        for name in strategies_to_run:
            if name in self.strategies:
                strategy = self.strategies[name]
                df = strategy.calculate_indicators(data.copy())
                df = strategy.generate_signals(df)
                backtest = strategy.backtest(data.copy())
                
                results[name] = {
                    'signals': df,
                    'backtest': backtest,
                    'trades': strategy.trades
                }
        
        return results
    
    def analyze_pairs(self,
                       prices: pd.DataFrame,
                       top_n: int = 5) -> Dict:
        """
        Find and analyze the best cointegrated pairs.
        
        Args:
            prices: DataFrame with multiple stock prices
            top_n: Number of top pairs to analyze
            
        Returns:
            Dictionary with pair analysis results
        """
        # Find pairs
        pairs = self.pair_strategy.find_cointegrated_pairs(prices)
        
        results = {
            'pairs_found': len(pairs),
            'pair_details': []
        }
        
        for stock1, stock2, pvalue in pairs[:top_n]:
            signals = self.pair_strategy.generate_pair_signals(
                prices[stock1],
                prices[stock2],
                f"{stock1}_{stock2}"
            )
            
            backtest = self.pair_strategy.backtest_pair(
                prices[stock1],
                prices[stock2]
            )
            
            results['pair_details'].append({
                'pair': (stock1, stock2),
                'p_value': pvalue,
                'hedge_ratio': self.pair_strategy.hedge_ratios.get(f"{stock1}_{stock2}"),
                'signals': signals,
                'backtest': backtest
            })
        
        return results
    
    def get_current_signals(self, data: pd.DataFrame) -> Dict:
        """Get current trading signals from all strategies."""
        signals = {}
        
        for name, strategy in self.strategies.items():
            df = strategy.calculate_indicators(data.copy())
            df = strategy.generate_signals(df)
            
            latest = df.iloc[-1]
            signals[name] = {
                'signal': latest.get('signal', 0),
                'signal_strength': abs(latest.get('z_score', 0)) if 'z_score' in df.columns else 0,
                'exit_signal': latest.get('exit_signal', 0)
            }
        
        return signals
    
    def visualize_bollinger(self, data: pd.DataFrame) -> Dict:
        """Prepare Bollinger Band visualization data."""
        bb_strategy = self.strategies.get('bollinger', BollingerBandReversion())
        df = bb_strategy.calculate_indicators(data.copy())
        df = bb_strategy.generate_signals(df)
        
        return {
            'dates': df.index.tolist(),
            'close': df['close'].tolist(),
            'upper_band': df['bb_upper'].tolist(),
            'middle_band': df['bb_middle'].tolist(),
            'lower_band': df['bb_lower'].tolist(),
            'z_score': df['z_score'].tolist(),
            'signals': df['signal'].tolist()
        }
    
    def visualize_rsi(self, data: pd.DataFrame) -> Dict:
        """Prepare RSI visualization data."""
        rsi_strategy = self.strategies.get('rsi', RSIReversion())
        df = rsi_strategy.calculate_indicators(data.copy())
        df = rsi_strategy.generate_signals(df)
        
        return {
            'dates': df.index.tolist(),
            'close': df['close'].tolist(),
            'rsi': df['rsi'].tolist(),
            'rsi_ma': df['rsi_ma'].tolist(),
            'signals': df['signal'].tolist(),
            'oversold': rsi_strategy.oversold,
            'overbought': rsi_strategy.overbought
        }


# Factory function
def create_mean_reversion_strategy(
    strategy_type: str,
    **kwargs
) -> MeanReversionStrategy:
    """
    Factory function to create mean reversion strategies.
    
    Args:
        strategy_type: 'bollinger', 'rsi', or 'pair'
        **kwargs: Strategy-specific parameters
    """
    strategies = {
        'bollinger': BollingerBandReversion,
        'rsi': RSIReversion,
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_type}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_type](**kwargs)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="2y")
    data.columns = [c.lower() for c in data.columns]
    
    # Create suite
    suite = MeanReversionSuite()
    
    # Run analysis
    print("=== Bollinger Band Analysis ===")
    results = suite.analyze_single_asset(data, 'bollinger')
    if 'bollinger' in results:
        backtest = results['bollinger']['backtest']
        print(f"Total Trades: {backtest['total_trades']}")
        print(f"Win Rate: {backtest['win_rate']:.2%}")
        print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest['max_drawdown']:.2%}")
    
    print("\n=== RSI Analysis ===")
    results = suite.analyze_single_asset(data, 'rsi')
    if 'rsi' in results:
        backtest = results['rsi']['backtest']
        print(f"Total Trades: {backtest['total_trades']}")
        print(f"Win Rate: {backtest['win_rate']:.2%}")
        print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
    
    print("\n=== Current Signals ===")
    signals = suite.get_current_signals(data)
    for strategy, signal in signals.items():
        print(f"{strategy}: Signal={signal['signal']}, Strength={signal['signal_strength']:.2f}")
