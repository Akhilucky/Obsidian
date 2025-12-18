"""
Signal Generation Module - Alpha Signal Generator
==================================================
Nightly feature pull → Alpha table generation

Features:
- Multi-factor alpha signals
- Ensemble signal combination
- Signal decay and half-life
- Cross-sectional and time-series signals
- Walk-forward signal validation
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

import pandas as pd
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Signals directory
SIGNALS_DIR = Path(__file__).parent.parent / "signals"
SIGNALS_DIR.mkdir(exist_ok=True)


class SignalType(Enum):
    """Types of trading signals."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    COMPOSITE = "composite"


@dataclass
class Signal:
    """Individual trading signal."""
    name: str
    signal_type: SignalType
    value: float  # -1 to 1 scale
    confidence: float  # 0 to 1
    timestamp: datetime
    symbol: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.signal_type.value,
            'value': self.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'metadata': self.metadata
        }


@dataclass
class AlphaSignal:
    """Alpha signal with decay and tracking."""
    signal_id: str
    symbol: str
    raw_alpha: float
    decayed_alpha: float
    z_score: float
    rank: int
    total_symbols: int
    signal_date: datetime
    decay_half_life: int = 5
    
    @property
    def percentile_rank(self) -> float:
        return (self.total_symbols - self.rank) / self.total_symbols * 100


class SignalGenerator:
    """Base class for signal generators."""
    
    def __init__(self, name: str, signal_type: SignalType):
        self.name = name
        self.signal_type = signal_type
    
    def generate(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate signal from data. Override in subclasses."""
        raise NotImplementedError
    
    def _normalize_signal(self, value: float, min_val: float = -1, max_val: float = 1) -> float:
        """Normalize signal to [-1, 1] range."""
        return np.clip(value, min_val, max_val)


class MomentumSignalGenerator(SignalGenerator):
    """Momentum-based signal generator."""
    
    def __init__(self, lookback: int = 20, skip: int = 1):
        super().__init__("momentum", SignalType.MOMENTUM)
        self.lookback = lookback
        self.skip = skip
    
    def generate(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate momentum signal."""
        if len(df) < self.lookback + self.skip:
            return Signal(self.name, self.signal_type, 0, 0, datetime.now(), symbol)
        
        # Calculate momentum (skip recent days to avoid reversal)
        returns = df['close'].pct_change(self.lookback).iloc[-1 - self.skip]
        
        # Normalize to [-1, 1]
        # Assume ±50% is extreme
        signal_value = self._normalize_signal(returns * 4)
        
        # Confidence based on trend consistency
        recent_returns = df['close'].pct_change().tail(self.lookback)
        win_rate = (recent_returns > 0).mean()
        confidence = abs(win_rate - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%
        
        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            value=signal_value,
            confidence=confidence,
            timestamp=datetime.now(),
            symbol=symbol,
            metadata={'lookback': self.lookback, 'raw_return': returns}
        )


class MeanReversionSignalGenerator(SignalGenerator):
    """Mean reversion signal generator."""
    
    def __init__(self, lookback: int = 20, z_threshold: float = 2.0):
        super().__init__("mean_reversion", SignalType.MEAN_REVERSION)
        self.lookback = lookback
        self.z_threshold = z_threshold
    
    def generate(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate mean reversion signal."""
        if len(df) < self.lookback:
            return Signal(self.name, self.signal_type, 0, 0, datetime.now(), symbol)
        
        # Calculate z-score
        prices = df['close'].tail(self.lookback)
        current_price = prices.iloc[-1]
        mean_price = prices.mean()
        std_price = prices.std()
        
        if std_price == 0:
            return Signal(self.name, self.signal_type, 0, 0, datetime.now(), symbol)
        
        z_score = (current_price - mean_price) / std_price
        
        # Mean reversion: negative signal when price is high (expect reversion down)
        signal_value = self._normalize_signal(-z_score / self.z_threshold)
        
        # Confidence higher when z-score is extreme
        confidence = min(abs(z_score) / self.z_threshold, 1.0)
        
        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            value=signal_value,
            confidence=confidence,
            timestamp=datetime.now(),
            symbol=symbol,
            metadata={'z_score': z_score, 'mean': mean_price, 'std': std_price}
        )


class RSISignalGenerator(SignalGenerator):
    """RSI-based signal generator."""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("rsi", SignalType.TECHNICAL)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate RSI signal."""
        if len(df) < self.period + 1:
            return Signal(self.name, self.signal_type, 0, 0, datetime.now(), symbol)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Convert RSI to signal
        # Oversold (RSI < 30) → Buy signal (positive)
        # Overbought (RSI > 70) → Sell signal (negative)
        if current_rsi < self.oversold:
            signal_value = (self.oversold - current_rsi) / self.oversold
        elif current_rsi > self.overbought:
            signal_value = -(current_rsi - self.overbought) / (100 - self.overbought)
        else:
            # Neutral zone
            signal_value = 0
        
        signal_value = self._normalize_signal(signal_value)
        confidence = abs(signal_value)
        
        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            value=signal_value,
            confidence=confidence,
            timestamp=datetime.now(),
            symbol=symbol,
            metadata={'rsi': current_rsi}
        )


class MACDSignalGenerator(SignalGenerator):
    """MACD-based signal generator."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("macd", SignalType.TECHNICAL)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def generate(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate MACD signal."""
        if len(df) < self.slow + self.signal_period:
            return Signal(self.name, self.signal_type, 0, 0, datetime.now(), symbol)
        
        # Calculate MACD
        ema_fast = df['close'].ewm(span=self.fast).mean()
        ema_slow = df['close'].ewm(span=self.slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        histogram = macd_line - signal_line
        
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        
        # Normalize histogram to price percentage
        price = df['close'].iloc[-1]
        normalized_hist = current_hist / price * 100
        
        # Signal based on histogram
        signal_value = self._normalize_signal(normalized_hist * 10)
        
        # Confidence based on histogram change direction
        if (current_hist > 0 and current_hist > prev_hist) or \
           (current_hist < 0 and current_hist < prev_hist):
            confidence = 0.8
        else:
            confidence = 0.5
        
        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            value=signal_value,
            confidence=confidence,
            timestamp=datetime.now(),
            symbol=symbol,
            metadata={'macd': macd_line.iloc[-1], 'signal': signal_line.iloc[-1], 'histogram': current_hist}
        )


class VolatilitySignalGenerator(SignalGenerator):
    """Volatility-based signal generator."""
    
    def __init__(self, short_window: int = 10, long_window: int = 60):
        super().__init__("volatility", SignalType.VOLATILITY)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate volatility signal."""
        if len(df) < self.long_window:
            return Signal(self.name, self.signal_type, 0, 0, datetime.now(), symbol)
        
        # Calculate short and long term volatility
        returns = df['close'].pct_change()
        short_vol = returns.tail(self.short_window).std() * np.sqrt(252)
        long_vol = returns.tail(self.long_window).std() * np.sqrt(252)
        
        # Volatility ratio
        vol_ratio = short_vol / (long_vol + 1e-10)
        
        # High short-term vol vs long-term → potential reversal (mean reversion)
        # Low short-term vol → potential breakout coming
        signal_value = self._normalize_signal(1 - vol_ratio)
        
        confidence = min(abs(1 - vol_ratio), 1.0)
        
        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            value=signal_value,
            confidence=confidence,
            timestamp=datetime.now(),
            symbol=symbol,
            metadata={'short_vol': short_vol, 'long_vol': long_vol, 'vol_ratio': vol_ratio}
        )


class AlphaTableGenerator:
    """
    Generate alpha tables from multiple signal sources.
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or SIGNALS_DIR
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize signal generators
        self.generators: Dict[str, SignalGenerator] = {
            'momentum_20': MomentumSignalGenerator(lookback=20),
            'momentum_60': MomentumSignalGenerator(lookback=60),
            'mean_reversion': MeanReversionSignalGenerator(),
            'rsi': RSISignalGenerator(),
            'macd': MACDSignalGenerator(),
            'volatility': VolatilitySignalGenerator(),
        }
        
        # Weights for ensemble
        self.weights = {
            'momentum_20': 0.15,
            'momentum_60': 0.15,
            'mean_reversion': 0.15,
            'rsi': 0.20,
            'macd': 0.20,
            'volatility': 0.15,
        }
    
    def add_generator(self, name: str, generator: SignalGenerator, weight: float = 0.1):
        """Add a signal generator."""
        self.generators[name] = generator
        self.weights[name] = weight
        # Renormalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> Dict[str, Signal]:
        """Generate all signals for a symbol."""
        signals = {}
        for name, generator in self.generators.items():
            try:
                signals[name] = generator.generate(df, symbol)
            except Exception as e:
                logger.error(f"Error generating {name} signal for {symbol}: {e}")
        return signals
    
    def compute_composite_alpha(self, signals: Dict[str, Signal]) -> Tuple[float, float]:
        """Compute weighted composite alpha signal."""
        if not signals:
            return 0.0, 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        
        for name, signal in signals.items():
            weight = self.weights.get(name, 0.1)
            weighted_sum += signal.value * signal.confidence * weight
            total_weight += weight * signal.confidence
            confidence_sum += signal.confidence
        
        if total_weight == 0:
            return 0.0, 0.0
        
        composite_alpha = weighted_sum / total_weight
        avg_confidence = confidence_sum / len(signals)
        
        return composite_alpha, avg_confidence
    
    def generate_alpha_table(self, data: Dict[str, pd.DataFrame],
                            apply_decay: bool = True,
                            decay_half_life: int = 5) -> pd.DataFrame:
        """
        Generate alpha table for multiple symbols.
        
        Args:
            data: Dict of symbol -> OHLCV DataFrame
            apply_decay: Whether to apply signal decay
            decay_half_life: Half-life for decay in days
        
        Returns:
            Alpha table DataFrame
        """
        records = []
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            # Generate signals
            signals = self.generate_signals(df, symbol)
            
            # Compute composite
            composite_alpha, confidence = self.compute_composite_alpha(signals)
            
            # Individual signal values
            signal_values = {name: sig.value for name, sig in signals.items()}
            
            records.append({
                'symbol': symbol,
                'date': datetime.now().date(),
                'composite_alpha': composite_alpha,
                'confidence': confidence,
                **signal_values
            })
        
        if not records:
            return pd.DataFrame()
        
        alpha_df = pd.DataFrame(records)
        
        # Compute cross-sectional z-scores and ranks
        alpha_df['alpha_zscore'] = stats.zscore(alpha_df['composite_alpha']) if SCIPY_AVAILABLE else \
            (alpha_df['composite_alpha'] - alpha_df['composite_alpha'].mean()) / alpha_df['composite_alpha'].std()
        alpha_df['alpha_rank'] = alpha_df['composite_alpha'].rank(ascending=False).astype(int)
        alpha_df['alpha_percentile'] = alpha_df['composite_alpha'].rank(pct=True) * 100
        
        # Apply decay to historical signals (if we had them)
        if apply_decay:
            alpha_df['decay_factor'] = 1.0  # Current signals have no decay
            alpha_df['decayed_alpha'] = alpha_df['composite_alpha']
        
        return alpha_df.sort_values('alpha_rank')
    
    def save_alpha_table(self, alpha_df: pd.DataFrame, name: str = None):
        """Save alpha table to disk."""
        if alpha_df.empty:
            return
        
        name = name or datetime.now().strftime("%Y%m%d")
        path = self.output_dir / f"alpha_table_{name}.parquet"
        alpha_df.to_parquet(path, index=False)
        logger.info(f"Saved alpha table to {path}")
        
        # Also save as CSV for easy viewing
        csv_path = self.output_dir / f"alpha_table_{name}.csv"
        alpha_df.to_csv(csv_path, index=False)
    
    def load_alpha_table(self, name: str = None) -> pd.DataFrame:
        """Load alpha table from disk."""
        if name is None:
            # Load latest
            files = sorted(self.output_dir.glob("alpha_table_*.parquet"))
            if not files:
                return pd.DataFrame()
            path = files[-1]
        else:
            path = self.output_dir / f"alpha_table_{name}.parquet"
        
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()
    
    def get_top_signals(self, alpha_df: pd.DataFrame, n: int = 10,
                       long_only: bool = False) -> Tuple[List[str], List[str]]:
        """Get top long and short signals."""
        if alpha_df.empty:
            return [], []
        
        sorted_df = alpha_df.sort_values('composite_alpha', ascending=False)
        
        longs = sorted_df.head(n)['symbol'].tolist()
        shorts = [] if long_only else sorted_df.tail(n)['symbol'].tolist()
        
        return longs, shorts


# Convenience functions
def generate_signals(df: pd.DataFrame, symbol: str) -> Dict[str, Signal]:
    """Generate all signals for a symbol."""
    generator = AlphaTableGenerator()
    return generator.generate_signals(df, symbol)


def get_alpha_table(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generate alpha table for multiple symbols."""
    generator = AlphaTableGenerator()
    return generator.generate_alpha_table(data)


if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing Signal Generation...")
    
    # Fetch sample data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
    data = {}
    
    for symbol in symbols:
        df = yf.download(symbol, period="1y", progress=False)
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            data[symbol] = df
    
    # Generate alpha table
    generator = AlphaTableGenerator()
    alpha_table = generator.generate_alpha_table(data)
    
    print("\n=== Alpha Table ===")
    print(alpha_table[['symbol', 'composite_alpha', 'confidence', 'alpha_rank', 'alpha_percentile']].to_string())
    
    # Get top signals
    longs, shorts = generator.get_top_signals(alpha_table, n=3)
    print(f"\nTop Long Signals: {longs}")
    print(f"Top Short Signals: {shorts}")
    
    # Save
    generator.save_alpha_table(alpha_table)
    print("\nAlpha table saved!")
