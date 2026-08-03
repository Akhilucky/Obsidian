"""
Trend Following Strategies
==========================
Moving average crossovers, breakout strategies, and momentum indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

try:
    from core.fast_kernels import sma_crossover_signals, NATIVE_AVAILABLE
    _NATIVE = NATIVE_AVAILABLE
except ImportError:
    _NATIVE = False

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class TradeSignal:
    """Container for trade signals."""
    symbol: str
    signal: Signal
    confidence: float  # 0-1
    price: float
    timestamp: pd.Timestamp
    strategy: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal': self.signal.name,
            'signal_value': self.signal.value,
            'confidence': self.confidence,
            'price': self.price,
            'timestamp': str(self.timestamp),
            'strategy': self.strategy,
            'metadata': self.metadata
        }


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.signals_history: List[TradeSignal] = []
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate trading signal from data."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict:
        """Get strategy parameters."""
        pass
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """
        Generate a signal per bar (one bar at a time, in order).
        Subclasses may override with batch/vectorized implementations.
        """
        signals = []
        for i in range(len(data)):
            window = data.iloc[:i+1]
            signals.append(self.generate_signal(window, symbol))
        return signals
    
    def backtest(
        self,
        data: pd.DataFrame,
        symbol: str,
        initial_capital: float = 100000,
        position_size: float = 0.1
    ) -> Dict:
        """Run simple backtest on historical data."""
        signals = self.generate_signals(data, symbol)
        
        # Calculate returns
        capital = initial_capital
        position = 0
        trades = []
        
        for i, signal in enumerate(signals):
            price = signal.price
            
            if signal.signal in [Signal.BUY, Signal.STRONG_BUY] and position == 0:
                # Buy
                shares = int((capital * position_size) / price)
                if shares > 0:
                    position = shares
                    capital -= shares * price
                    trades.append({'type': 'BUY', 'price': price, 'shares': shares})
            
            elif signal.signal in [Signal.SELL, Signal.STRONG_SELL] and position > 0:
                # Sell
                capital += position * price
                trades.append({'type': 'SELL', 'price': price, 'shares': position})
                position = 0
        
        # Final value
        final_price = data['close'].iloc[-1]
        total_value = capital + position * final_price
        
        return {
            'initial_capital': initial_capital,
            'final_value': total_value,
            'return': (total_value - initial_capital) / initial_capital,
            'num_trades': len(trades),
            'trades': trades
        }


# ============================================================================
# Moving Average Strategies
# ============================================================================

class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Generates buy signal when short MA crosses above long MA,
    sell signal when short MA crosses below long MA.
    """
    
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        signal_threshold: float = 0.001
    ):
        super().__init__("SMA_Crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.signal_threshold = signal_threshold

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """
        Batch signal generation using the C++ single-pass kernel.
        Falls back to the per-bar Python path if the kernel is unavailable.
        """
        try:
            import core.fast_kernels as fk

            prices = data['close'].to_numpy(dtype=np.float64)
            raw_signals, confidences = fk.sma_crossover_signals(
                prices,
                short_window=self.short_window,
                long_window=self.long_window,
                signal_threshold=self.signal_threshold,
            )

            kernel_to_signal = {
                0: Signal.HOLD,
                1: Signal.BUY,
                2: Signal.STRONG_BUY,
                3: Signal.SELL,
                4: Signal.STRONG_SELL,
            }

            signals = []
            for i in range(len(data)):
                raw = int(raw_signals[i])
                signals.append(TradeSignal(
                    symbol=symbol,
                    signal=kernel_to_signal[raw],
                    confidence=float(confidences[i]),
                    price=float(prices[i]),
                    timestamp=data.index[i],
                    strategy=self.name,
                    metadata={'fast_kernel': True},
                ))
            return signals
        except Exception:
            return super().generate_signals(data, symbol)
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate signal based on SMA crossover."""
        if len(data) < self.long_window:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strategy=self.name
            )
        
        # Calculate SMAs
        short_sma = data['close'].rolling(window=self.short_window).mean()
        long_sma = data['close'].rolling(window=self.long_window).mean()
        
        current_short = short_sma.iloc[-1]
        current_long = long_sma.iloc[-1]
        prev_short = short_sma.iloc[-2]
        prev_long = long_sma.iloc[-2]
        
        current_price = data['close'].iloc[-1]
        
        # Calculate crossover
        cross_above = prev_short <= prev_long and current_short > current_long
        cross_below = prev_short >= prev_long and current_short < current_long
        
        # Calculate distance from long MA for confidence
        distance = (current_short - current_long) / current_long
        confidence = min(abs(distance) / 0.05, 1.0)  # Max confidence at 5% distance
        
        if cross_above:
            signal = Signal.STRONG_BUY if distance > self.signal_threshold * 2 else Signal.BUY
        elif cross_below:
            signal = Signal.STRONG_SELL if distance < -self.signal_threshold * 2 else Signal.SELL
        elif current_short > current_long * (1 + self.signal_threshold):
            signal = Signal.BUY
            confidence *= 0.7
        elif current_short < current_long * (1 - self.signal_threshold):
            signal = Signal.SELL
            confidence *= 0.7
        else:
            signal = Signal.HOLD
            confidence = 0.5
        
        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=current_price,
            timestamp=data.index[-1],
            strategy=self.name,
            metadata={
                'short_sma': current_short,
                'long_sma': current_long,
                'distance': distance
            }
        )
    
    def get_parameters(self) -> Dict:
        return {
            'short_window': self.short_window,
            'long_window': self.long_window,
            'signal_threshold': self.signal_threshold
        }


class EMAStrategy(BaseStrategy):
    """
    Exponential Moving Average Crossover Strategy.
    
    Uses EMA for faster response to price changes.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        super().__init__("EMA_Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate signal based on EMA crossover."""
        if len(data) < self.slow_period + self.signal_period:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strategy=self.name
            )
        
        # Calculate EMAs
        fast_ema = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # MACD line
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        current_price = data['close'].iloc[-1]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        
        # Crossover detection
        cross_above = prev_hist <= 0 and current_hist > 0
        cross_below = prev_hist >= 0 and current_hist < 0
        
        # Confidence based on histogram strength
        hist_std = histogram.std()
        confidence = min(abs(current_hist) / (2 * hist_std) if hist_std > 0 else 0.5, 1.0)
        
        if cross_above:
            signal = Signal.STRONG_BUY
        elif cross_below:
            signal = Signal.STRONG_SELL
        elif current_hist > 0:
            signal = Signal.BUY
        elif current_hist < 0:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=current_price,
            timestamp=data.index[-1],
            strategy=self.name,
            metadata={
                'fast_ema': fast_ema.iloc[-1],
                'slow_ema': slow_ema.iloc[-1],
                'macd': macd_line.iloc[-1],
                'signal': signal_line.iloc[-1],
                'histogram': current_hist
            }
        )
    
    def get_parameters(self) -> Dict:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period
        }


class TripleMAStrategy(BaseStrategy):
    """
    Triple Moving Average Strategy.
    
    Uses three MAs: fast, medium, slow for trend confirmation.
    """
    
    def __init__(
        self,
        fast_period: int = 10,
        medium_period: int = 20,
        slow_period: int = 50
    ):
        super().__init__("Triple_MA")
        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate signal based on triple MA alignment."""
        if len(data) < self.slow_period:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strategy=self.name
            )
        
        fast_ma = data['close'].rolling(self.fast_period).mean().iloc[-1]
        medium_ma = data['close'].rolling(self.medium_period).mean().iloc[-1]
        slow_ma = data['close'].rolling(self.slow_period).mean().iloc[-1]
        
        current_price = data['close'].iloc[-1]
        
        # Check alignment
        bullish_aligned = fast_ma > medium_ma > slow_ma
        bearish_aligned = fast_ma < medium_ma < slow_ma
        
        # Calculate trend strength
        spread = (fast_ma - slow_ma) / slow_ma
        confidence = min(abs(spread) / 0.1, 1.0)
        
        if bullish_aligned:
            signal = Signal.STRONG_BUY if spread > 0.02 else Signal.BUY
        elif bearish_aligned:
            signal = Signal.STRONG_SELL if spread < -0.02 else Signal.SELL
        else:
            signal = Signal.HOLD
            confidence = 0.3
        
        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=current_price,
            timestamp=data.index[-1],
            strategy=self.name,
            metadata={
                'fast_ma': fast_ma,
                'medium_ma': medium_ma,
                'slow_ma': slow_ma,
                'spread': spread
            }
        )
    
    def get_parameters(self) -> Dict:
        return {
            'fast_period': self.fast_period,
            'medium_period': self.medium_period,
            'slow_period': self.slow_period
        }


# ============================================================================
# Breakout Strategies
# ============================================================================

class BreakoutStrategy(BaseStrategy):
    """
    Price Breakout Strategy.
    
    Detects breakouts from consolidation ranges using
    Donchian channels and volatility confirmation.
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        volume_confirmation: bool = True
    ):
        super().__init__("Breakout")
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.volume_confirmation = volume_confirmation
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate signal based on price breakout."""
        if len(data) < max(self.lookback_period, self.atr_period) + 5:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strategy=self.name
            )
        
        # Donchian channels
        high_channel = data['high'].rolling(self.lookback_period).max()
        low_channel = data['low'].rolling(self.lookback_period).min()
        
        current_price = data['close'].iloc[-1]
        prev_high = high_channel.iloc[-2]
        prev_low = low_channel.iloc[-2]
        
        # ATR for volatility
        atr = self._calculate_atr(data)
        current_atr = atr.iloc[-1]
        
        # Volume confirmation
        if self.volume_confirmation and 'volume' in data.columns:
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            volume_surge = current_volume > avg_volume * 1.5
        else:
            volume_surge = True
        
        # Breakout detection
        breakout_up = current_price > prev_high
        breakout_down = current_price < prev_low
        
        # Calculate breakout strength
        if breakout_up:
            breakout_distance = (current_price - prev_high) / current_atr
            confidence = min(breakout_distance / 2, 1.0) * (1.2 if volume_surge else 0.8)
            signal = Signal.STRONG_BUY if breakout_distance > 1 and volume_surge else Signal.BUY
        elif breakout_down:
            breakout_distance = (prev_low - current_price) / current_atr
            confidence = min(breakout_distance / 2, 1.0) * (1.2 if volume_surge else 0.8)
            signal = Signal.STRONG_SELL if breakout_distance > 1 and volume_surge else Signal.SELL
        else:
            signal = Signal.HOLD
            confidence = 0.5
        
        confidence = min(confidence, 1.0)
        
        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=current_price,
            timestamp=data.index[-1],
            strategy=self.name,
            metadata={
                'upper_channel': high_channel.iloc[-1],
                'lower_channel': low_channel.iloc[-1],
                'atr': current_atr,
                'volume_surge': volume_surge
            }
        )
    
    def get_parameters(self) -> Dict:
        return {
            'lookback_period': self.lookback_period,
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
            'volume_confirmation': self.volume_confirmation
        }


class TurtleBreakout(BaseStrategy):
    """
    Turtle Trading Breakout Strategy.
    
    Classic trend-following system using 20/55 day breakouts.
    """
    
    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        atr_stop_multiplier: float = 2.0
    ):
        super().__init__("Turtle_Breakout")
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self._position = 0  # Track position internally
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate ATR."""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            abs(high - close),
            abs(low - close)
        ], axis=1).max(axis=1)
        
        return tr.rolling(self.atr_period).mean().iloc[-1]
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate Turtle trading signals."""
        if len(data) < max(self.entry_period, self.exit_period, self.atr_period) + 5:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strategy=self.name
            )
        
        current_price = data['close'].iloc[-1]
        
        # Entry channels (excluding today)
        entry_high = data['high'].iloc[-(self.entry_period+1):-1].max()
        entry_low = data['low'].iloc[-(self.entry_period+1):-1].min()
        
        # Exit channels
        exit_high = data['high'].iloc[-(self.exit_period+1):-1].max()
        exit_low = data['low'].iloc[-(self.exit_period+1):-1].min()
        
        atr = self._calculate_atr(data)
        
        # Entry signals
        if current_price > entry_high and self._position <= 0:
            signal = Signal.STRONG_BUY
            self._position = 1
            confidence = 0.9
        elif current_price < entry_low and self._position >= 0:
            signal = Signal.STRONG_SELL
            self._position = -1
            confidence = 0.9
        # Exit signals
        elif self._position > 0 and current_price < exit_low:
            signal = Signal.SELL
            self._position = 0
            confidence = 0.8
        elif self._position < 0 and current_price > exit_high:
            signal = Signal.BUY
            self._position = 0
            confidence = 0.8
        else:
            signal = Signal.HOLD
            confidence = 0.5
        
        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=current_price,
            timestamp=data.index[-1],
            strategy=self.name,
            metadata={
                'entry_high': entry_high,
                'entry_low': entry_low,
                'exit_high': exit_high,
                'exit_low': exit_low,
                'atr': atr,
                'stop_distance': atr * self.atr_stop_multiplier
            }
        )
    
    def get_parameters(self) -> Dict:
        return {
            'entry_period': self.entry_period,
            'exit_period': self.exit_period,
            'atr_period': self.atr_period,
            'atr_stop_multiplier': self.atr_stop_multiplier
        }


# ============================================================================
# Momentum Strategies
# ============================================================================

class RSIMomentumStrategy(BaseStrategy):
    """
    RSI Momentum Strategy.
    
    Uses RSI for momentum with trend confirmation.
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        ma_period: int = 50
    ):
        super().__init__("RSI_Momentum")
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.ma_period = ma_period
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate signal based on RSI momentum."""
        if len(data) < max(self.rsi_period, self.ma_period) + 5:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strategy=self.name
            )
        
        rsi = self._calculate_rsi(data)
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # Trend filter
        ma = data['close'].rolling(self.ma_period).mean()
        current_price = data['close'].iloc[-1]
        uptrend = current_price > ma.iloc[-1]
        
        # RSI momentum
        rsi_rising = current_rsi > prev_rsi
        rsi_falling = current_rsi < prev_rsi
        
        # Signal generation
        if current_rsi < self.oversold and rsi_rising and uptrend:
            signal = Signal.STRONG_BUY
            confidence = (self.oversold - current_rsi) / self.oversold
        elif current_rsi < 40 and rsi_rising and uptrend:
            signal = Signal.BUY
            confidence = 0.7
        elif current_rsi > self.overbought and rsi_falling and not uptrend:
            signal = Signal.STRONG_SELL
            confidence = (current_rsi - self.overbought) / (100 - self.overbought)
        elif current_rsi > 60 and rsi_falling and not uptrend:
            signal = Signal.SELL
            confidence = 0.7
        else:
            signal = Signal.HOLD
            confidence = 0.5
        
        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=min(confidence, 1.0),
            price=current_price,
            timestamp=data.index[-1],
            strategy=self.name,
            metadata={
                'rsi': current_rsi,
                'uptrend': uptrend,
                'rsi_direction': 'rising' if rsi_rising else 'falling'
            }
        )
    
    def get_parameters(self) -> Dict:
        return {
            'rsi_period': self.rsi_period,
            'overbought': self.overbought,
            'oversold': self.oversold,
            'ma_period': self.ma_period
        }


class MACDMomentumStrategy(BaseStrategy):
    """
    MACD Momentum Strategy.
    
    Uses MACD histogram for momentum with divergence detection.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        histogram_threshold: float = 0.0
    ):
        super().__init__("MACD_Momentum")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.histogram_threshold = histogram_threshold
    
    def _calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components."""
        fast_ema = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _detect_divergence(
        self,
        price: pd.Series,
        macd: pd.Series,
        lookback: int = 20
    ) -> Tuple[bool, bool]:
        """Detect bullish and bearish divergence."""
        if len(price) < lookback:
            return False, False
        
        price_window = price.iloc[-lookback:]
        macd_window = macd.iloc[-lookback:]
        
        # Find local minima/maxima
        price_min_idx = price_window.idxmin()
        price_max_idx = price_window.idxmax()
        
        # Bullish divergence: price lower low, MACD higher low
        recent_price_low = price_window.iloc[-5:].min()
        prev_price_low = price_window.iloc[:10].min()
        recent_macd_low = macd_window.iloc[-5:].min()
        prev_macd_low = macd_window.iloc[:10].min()
        
        bullish_div = recent_price_low < prev_price_low and recent_macd_low > prev_macd_low
        
        # Bearish divergence: price higher high, MACD lower high
        recent_price_high = price_window.iloc[-5:].max()
        prev_price_high = price_window.iloc[:10].max()
        recent_macd_high = macd_window.iloc[-5:].max()
        prev_macd_high = macd_window.iloc[:10].max()
        
        bearish_div = recent_price_high > prev_price_high and recent_macd_high < prev_macd_high
        
        return bullish_div, bearish_div
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate signal based on MACD momentum."""
        if len(data) < self.slow_period + self.signal_period + 20:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strategy=self.name
            )
        
        macd_line, signal_line, histogram = self._calculate_macd(data)
        
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        current_macd = macd_line.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Detect crossovers
        cross_above = prev_hist <= 0 and current_hist > 0
        cross_below = prev_hist >= 0 and current_hist < 0
        
        # Detect divergence
        bullish_div, bearish_div = self._detect_divergence(data['close'], macd_line)
        
        # Histogram momentum
        hist_increasing = current_hist > prev_hist
        hist_decreasing = current_hist < prev_hist
        
        # Calculate confidence
        hist_std = histogram.rolling(20).std().iloc[-1]
        confidence = min(abs(current_hist) / (2 * hist_std) if hist_std > 0 else 0.5, 1.0)
        
        # Signal generation
        if cross_above or (bullish_div and hist_increasing):
            signal = Signal.STRONG_BUY if bullish_div else Signal.BUY
            confidence = min(confidence * 1.2 if bullish_div else confidence, 1.0)
        elif cross_below or (bearish_div and hist_decreasing):
            signal = Signal.STRONG_SELL if bearish_div else Signal.SELL
            confidence = min(confidence * 1.2 if bearish_div else confidence, 1.0)
        elif current_hist > self.histogram_threshold and hist_increasing:
            signal = Signal.BUY
        elif current_hist < -self.histogram_threshold and hist_decreasing:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
            confidence = 0.5
        
        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=current_price,
            timestamp=data.index[-1],
            strategy=self.name,
            metadata={
                'macd': current_macd,
                'signal': signal_line.iloc[-1],
                'histogram': current_hist,
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div
            }
        )
    
    def get_parameters(self) -> Dict:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'histogram_threshold': self.histogram_threshold
        }


class ADXTrendStrength(BaseStrategy):
    """
    ADX Trend Strength Strategy.
    
    Uses ADX to measure trend strength and +DI/-DI for direction.
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        trend_threshold: float = 25,
        strong_trend: float = 40
    ):
        super().__init__("ADX_Trend")
        self.adx_period = adx_period
        self.trend_threshold = trend_threshold
        self.strong_trend = strong_trend
    
    def _calculate_adx(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, and -DI."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.adx_period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)
        
        # Smoothed DM
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx, plus_di, minus_di
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate signal based on ADX trend strength."""
        if len(data) < self.adx_period * 2 + 5:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strategy=self.name
            )
        
        adx, plus_di, minus_di = self._calculate_adx(data)
        
        current_adx = adx.iloc[-1]
        current_plus = plus_di.iloc[-1]
        current_minus = minus_di.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Trend strength
        trending = current_adx > self.trend_threshold
        strong_trending = current_adx > self.strong_trend
        
        # Direction
        bullish = current_plus > current_minus
        bearish = current_minus > current_plus
        
        # DI crossover
        prev_plus = plus_di.iloc[-2]
        prev_minus = minus_di.iloc[-2]
        cross_bullish = prev_plus <= prev_minus and current_plus > current_minus
        cross_bearish = prev_plus >= prev_minus and current_plus < current_minus
        
        # Confidence based on ADX strength
        confidence = min(current_adx / 50, 1.0)
        
        if trending and bullish:
            signal = Signal.STRONG_BUY if strong_trending or cross_bullish else Signal.BUY
        elif trending and bearish:
            signal = Signal.STRONG_SELL if strong_trending or cross_bearish else Signal.SELL
        else:
            signal = Signal.HOLD
            confidence = 0.3
        
        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=current_price,
            timestamp=data.index[-1],
            strategy=self.name,
            metadata={
                'adx': current_adx,
                'plus_di': current_plus,
                'minus_di': current_minus,
                'trending': trending,
                'direction': 'bullish' if bullish else 'bearish'
            }
        )
    
    def get_parameters(self) -> Dict:
        return {
            'adx_period': self.adx_period,
            'trend_threshold': self.trend_threshold,
            'strong_trend': self.strong_trend
        }


# ============================================================================
# Strategy Ensemble
# ============================================================================

class TrendFollowingEnsemble:
    """
    Ensemble of trend following strategies.
    
    Combines signals from multiple strategies with weighted voting.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.strategies = [
            SMAStrategy(),
            EMAStrategy(),
            TripleMAStrategy(),
            BreakoutStrategy(),
            TurtleBreakout(),
            RSIMomentumStrategy(),
            MACDMomentumStrategy(),
            ADXTrendStrength()
        ]
        
        # Default equal weights
        self.weights = weights or {s.name: 1.0 for s in self.strategies}
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate ensemble signal from all strategies."""
        signals = []
        total_weight = 0
        weighted_score = 0
        
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(data, symbol)
                weight = self.weights.get(strategy.name, 1.0)
                signals.append((strategy.name, signal))
                
                # Weighted score
                weighted_score += signal.signal.value * signal.confidence * weight
                total_weight += weight * signal.confidence
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")
        
        # Average weighted score
        if total_weight > 0:
            avg_score = weighted_score / total_weight
        else:
            avg_score = 0
        
        # Convert to signal
        if avg_score >= 1.5:
            final_signal = Signal.STRONG_BUY
        elif avg_score >= 0.5:
            final_signal = Signal.BUY
        elif avg_score <= -1.5:
            final_signal = Signal.STRONG_SELL
        elif avg_score <= -0.5:
            final_signal = Signal.SELL
        else:
            final_signal = Signal.HOLD
        
        # Consensus confidence
        signal_values = [s.signal.value for _, s in signals]
        consensus = 1 - np.std(signal_values) / 2  # Higher consensus = lower std
        confidence = min(abs(avg_score) / 2 * consensus, 1.0)
        
        return TradeSignal(
            symbol=symbol,
            signal=final_signal,
            confidence=confidence,
            price=data['close'].iloc[-1],
            timestamp=data.index[-1],
            strategy="TrendFollowing_Ensemble",
            metadata={
                'avg_score': avg_score,
                'consensus': consensus,
                'individual_signals': {name: s.signal.name for name, s in signals}
            }
        )
    
    def get_all_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """Get individual signals from all strategies."""
        signals = []
        for strategy in self.strategies:
            try:
                signals.append(strategy.generate_signal(data, symbol))
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")
        return signals


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Fetch sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y")
    data.columns = [c.lower() for c in data.columns]
    
    # Test individual strategies
    sma = SMAStrategy()
    signal = sma.generate_signal(data, "AAPL")
    print(f"SMA Strategy: {signal.signal.name} (confidence: {signal.confidence:.2f})")
    
    # Test ensemble
    ensemble = TrendFollowingEnsemble()
    ensemble_signal = ensemble.generate_signal(data, "AAPL")
    print(f"\nEnsemble Signal: {ensemble_signal.signal.name}")
    print(f"Confidence: {ensemble_signal.confidence:.2f}")
    print(f"Individual signals: {ensemble_signal.metadata['individual_signals']}")
