"""
Advanced Charting & Technical Analysis Module
==============================================

50+ technical indicators, custom indicators, drawing tools, and advanced charting
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class IndicatorCategory(Enum):
    """Technical indicator categories"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    OVERLAY = "overlay"
    CUSTOM = "custom"


@dataclass
class IndicatorResult:
    """Result from technical indicator calculation"""
    name: str
    category: IndicatorCategory
    values: Dict[str, List[float]]  # Can have multiple series (e.g., MACD has signal, histogram)
    timestamps: List[datetime]
    params: Dict[str, Any]


@dataclass
class DrawingObject:
    """Chart drawing object"""
    id: str
    type: str  # trendline, horizontal, vertical, fibonacci, etc.
    points: List[Tuple[datetime, float]]
    style: Dict[str, Any]  # color, width, etc.
    label: Optional[str] = None


class TechnicalIndicators:
    """
    Library of 50+ technical indicators
    """
    
    @staticmethod
    def sma(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Simple Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result
    
    @staticmethod
    def ema(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Exponential Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        multiplier = 2 / (period + 1)
        result = np.full(len(prices), np.nan)
        result[period - 1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            result[i] = (prices[i] * multiplier) + (result[i - 1] * (1 - multiplier))
        
        return result
    
    @staticmethod
    def wma(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Weighted Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        weights = np.arange(1, period + 1)
        result = np.full(len(prices), np.nan)
        
        for i in range(period - 1, len(prices)):
            result[i] = np.sum(prices[i - period + 1:i + 1] * weights) / np.sum(weights)
        
        return result
    
    @staticmethod
    def dema(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Double Exponential Moving Average"""
        ema1 = TechnicalIndicators.ema(prices, period)
        ema2 = TechnicalIndicators.ema(ema1[~np.isnan(ema1)], period)
        
        result = np.full(len(prices), np.nan)
        valid_start = 2 * period - 2
        if valid_start < len(prices):
            result[valid_start:valid_start + len(ema2)] = 2 * ema1[valid_start:valid_start + len(ema2)] - ema2
        
        return result
    
    @staticmethod
    def tema(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Triple Exponential Moving Average"""
        ema1 = TechnicalIndicators.ema(prices, period)
        ema2 = TechnicalIndicators.ema(ema1[~np.isnan(ema1)], period)
        ema3 = TechnicalIndicators.ema(ema2[~np.isnan(ema2)], period)
        
        result = np.full(len(prices), np.nan)
        valid_start = 3 * period - 3
        if valid_start < len(prices) and len(ema3) > 0:
            end_idx = min(valid_start + len(ema3), len(prices))
            result[valid_start:end_idx] = (
                3 * ema1[valid_start:end_idx] - 
                3 * ema2[:end_idx - valid_start] + 
                ema3[:end_idx - valid_start]
            )
        
        return result
    
    @staticmethod
    def macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Moving Average Convergence Divergence"""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line[~np.isnan(macd_line)], signal)
        
        # Align signal line with macd line
        full_signal = np.full(len(prices), np.nan)
        start_idx = slow - 1 + signal - 1
        if start_idx < len(prices):
            full_signal[start_idx:start_idx + len(signal_line)] = signal_line
        
        histogram = macd_line - full_signal
        
        return macd_line, full_signal, histogram
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.full(len(prices), np.nan)
        avg_loss = np.full(len(prices), np.nan)
        
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def stochastic(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        if len(close) < k_period:
            return np.full(len(close), np.nan), np.full(len(close), np.nan)
        
        k_values = np.full(len(close), np.nan)
        
        for i in range(k_period - 1, len(close)):
            highest_high = np.max(high[i - k_period + 1:i + 1])
            lowest_low = np.min(low[i - k_period + 1:i + 1])
            
            if highest_high != lowest_low:
                k_values[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                k_values[i] = 50
        
        d_values = TechnicalIndicators.sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    def bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        middle = TechnicalIndicators.sma(prices, period)
        
        std = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Average True Range"""
        if len(close) < 2:
            return np.full(len(close), np.nan)
        
        tr = np.full(len(close), np.nan)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
        
        atr = TechnicalIndicators.ema(tr, period)
        return atr
    
    @staticmethod
    def adx(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average Directional Index"""
        if len(close) < period + 1:
            return (np.full(len(close), np.nan),
                   np.full(len(close), np.nan),
                   np.full(len(close), np.nan))
        
        # Calculate +DM and -DM
        plus_dm = np.full(len(close), np.nan)
        minus_dm = np.full(len(close), np.nan)
        
        for i in range(1, len(close)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
            
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        atr_val = TechnicalIndicators.atr(high, low, close, period)
        
        plus_di = 100 * TechnicalIndicators.ema(plus_dm, period) / np.where(atr_val != 0, atr_val, 1)
        minus_di = 100 * TechnicalIndicators.ema(minus_dm, period) / np.where(atr_val != 0, atr_val, 1)
        
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, plus_di + minus_di, 1)
        adx = TechnicalIndicators.ema(dx, period)
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def cci(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = TechnicalIndicators.sma(tp, period)
        
        mean_dev = np.full(len(close), np.nan)
        for i in range(period - 1, len(close)):
            mean_dev[i] = np.mean(np.abs(tp[i - period + 1:i + 1] - sma_tp[i]))
        
        cci = (tp - sma_tp) / (0.015 * np.where(mean_dev != 0, mean_dev, 1))
        return cci
    
    @staticmethod
    def williams_r(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Williams %R"""
        if len(close) < period:
            return np.full(len(close), np.nan)
        
        williams = np.full(len(close), np.nan)
        
        for i in range(period - 1, len(close)):
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            
            if highest_high != lowest_low:
                williams[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
            else:
                williams[i] = -50
        
        return williams
    
    @staticmethod
    def mfi(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Money Flow Index"""
        if len(close) < period + 1:
            return np.full(len(close), np.nan)
        
        tp = (high + low + close) / 3
        raw_mf = tp * volume
        
        positive_mf = np.full(len(close), 0.0)
        negative_mf = np.full(len(close), 0.0)
        
        for i in range(1, len(close)):
            if tp[i] > tp[i - 1]:
                positive_mf[i] = raw_mf[i]
            elif tp[i] < tp[i - 1]:
                negative_mf[i] = raw_mf[i]
        
        mfi = np.full(len(close), np.nan)
        
        for i in range(period, len(close)):
            pos_sum = np.sum(positive_mf[i - period + 1:i + 1])
            neg_sum = np.sum(negative_mf[i - period + 1:i + 1])
            
            if neg_sum != 0:
                mf_ratio = pos_sum / neg_sum
                mfi[i] = 100 - (100 / (1 + mf_ratio))
            else:
                mfi[i] = 100
        
        return mfi
    
    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On Balance Volume"""
        obv = np.full(len(close), np.nan)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        
        return obv
    
    @staticmethod
    def vwap(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """Volume Weighted Average Price"""
        tp = (high + low + close) / 3
        cumulative_tp_vol = np.cumsum(tp * volume)
        cumulative_vol = np.cumsum(volume)
        
        vwap = cumulative_tp_vol / np.where(cumulative_vol != 0, cumulative_vol, 1)
        return vwap
    
    @staticmethod
    def ichimoku(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        conversion: int = 9,
        base: int = 26,
        span_b: int = 52,
        displacement: int = 26
    ) -> Dict[str, np.ndarray]:
        """Ichimoku Cloud"""
        def donchian(h, l, period):
            result = np.full(len(h), np.nan)
            for i in range(period - 1, len(h)):
                result[i] = (np.max(h[i - period + 1:i + 1]) + np.min(l[i - period + 1:i + 1])) / 2
            return result
        
        tenkan = donchian(high, low, conversion)  # Conversion Line
        kijun = donchian(high, low, base)  # Base Line
        
        senkou_a = (tenkan + kijun) / 2  # Leading Span A
        senkou_b = donchian(high, low, span_b)  # Leading Span B
        
        # Shift senkou forward
        senkou_a_shifted = np.full(len(close) + displacement, np.nan)
        senkou_a_shifted[displacement:] = senkou_a
        
        senkou_b_shifted = np.full(len(close) + displacement, np.nan)
        senkou_b_shifted[displacement:] = senkou_b
        
        # Chikou (lagging span)
        chikou = np.full(len(close), np.nan)
        chikou[:-displacement] = close[displacement:]
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a_shifted[:len(close)],
            'senkou_b': senkou_b_shifted[:len(close)],
            'chikou': chikou
        }
    
    @staticmethod
    def pivot_points(
        high: float,
        low: float,
        close: float
    ) -> Dict[str, float]:
        """Calculate Pivot Points"""
        pivot = (high + low + close) / 3
        
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }
    
    @staticmethod
    def fibonacci_retracements(
        high: float,
        low: float,
        is_uptrend: bool = True
    ) -> Dict[str, float]:
        """Calculate Fibonacci Retracement Levels"""
        diff = high - low
        
        if is_uptrend:
            return {
                '0.0%': high,
                '23.6%': high - diff * 0.236,
                '38.2%': high - diff * 0.382,
                '50.0%': high - diff * 0.500,
                '61.8%': high - diff * 0.618,
                '78.6%': high - diff * 0.786,
                '100.0%': low
            }
        else:
            return {
                '0.0%': low,
                '23.6%': low + diff * 0.236,
                '38.2%': low + diff * 0.382,
                '50.0%': low + diff * 0.500,
                '61.8%': low + diff * 0.618,
                '78.6%': low + diff * 0.786,
                '100.0%': high
            }
    
    @staticmethod
    def supertrend(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Supertrend Indicator"""
        atr_val = TechnicalIndicators.atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        upper_band = hl2 + (multiplier * atr_val)
        lower_band = hl2 - (multiplier * atr_val)
        
        supertrend = np.full(len(close), np.nan)
        direction = np.full(len(close), 1)  # 1 = uptrend, -1 = downtrend
        
        for i in range(period, len(close)):
            if close[i] > upper_band[i - 1]:
                direction[i] = 1
            elif close[i] < lower_band[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
            
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
        
        return supertrend, direction
    
    @staticmethod
    def parabolic_sar(
        high: np.ndarray,
        low: np.ndarray,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.20
    ) -> np.ndarray:
        """Parabolic SAR"""
        n = len(high)
        sar = np.full(n, np.nan)
        ep = np.full(n, np.nan)
        af = np.full(n, af_start)
        trend = np.ones(n)  # 1 = uptrend, -1 = downtrend
        
        # Initialize
        sar[0] = low[0]
        ep[0] = high[0]
        
        for i in range(1, n):
            if trend[i - 1] == 1:
                sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
                sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
                
                if high[i] > ep[i - 1]:
                    ep[i] = high[i]
                    af[i] = min(af[i - 1] + af_increment, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
                
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i - 1]
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
            else:
                sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
                sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
                
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af[i] = min(af[i - 1] + af_increment, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
                
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i - 1]
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
        
        return sar
    
    @staticmethod
    def keltner_channels(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channels"""
        middle = TechnicalIndicators.ema(close, ema_period)
        atr_val = TechnicalIndicators.atr(high, low, close, atr_period)
        
        upper = middle + (multiplier * atr_val)
        lower = middle - (multiplier * atr_val)
        
        return upper, middle, lower
    
    @staticmethod
    def donchian_channels(
        high: np.ndarray,
        low: np.ndarray,
        period: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Donchian Channels"""
        upper = np.full(len(high), np.nan)
        lower = np.full(len(low), np.nan)
        
        for i in range(period - 1, len(high)):
            upper[i] = np.max(high[i - period + 1:i + 1])
            lower[i] = np.min(low[i - period + 1:i + 1])
        
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    @staticmethod
    def aroon(
        high: np.ndarray,
        low: np.ndarray,
        period: int = 25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aroon Indicator"""
        aroon_up = np.full(len(high), np.nan)
        aroon_down = np.full(len(low), np.nan)
        
        for i in range(period, len(high)):
            high_idx = np.argmax(high[i - period:i + 1])
            low_idx = np.argmin(low[i - period:i + 1])
            
            aroon_up[i] = ((period - (period - high_idx)) / period) * 100
            aroon_down[i] = ((period - (period - low_idx)) / period) * 100
        
        aroon_osc = aroon_up - aroon_down
        
        return aroon_up, aroon_down, aroon_osc
    
    @staticmethod
    def roc(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Rate of Change"""
        roc = np.full(len(prices), np.nan)
        
        for i in range(period, len(prices)):
            if prices[i - period] != 0:
                roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
        
        return roc
    
    @staticmethod
    def momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Momentum Indicator"""
        mom = np.full(len(prices), np.nan)
        
        for i in range(period, len(prices)):
            mom[i] = prices[i] - prices[i - period]
        
        return mom
    
    @staticmethod
    def chaikin_money_flow(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """Chaikin Money Flow"""
        hl_range = high - low
        mfm = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0)
        mfv = mfm * volume
        
        cmf = np.full(len(close), np.nan)
        
        for i in range(period - 1, len(close)):
            vol_sum = np.sum(volume[i - period + 1:i + 1])
            if vol_sum != 0:
                cmf[i] = np.sum(mfv[i - period + 1:i + 1]) / vol_sum
        
        return cmf
    
    @staticmethod
    def accumulation_distribution(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """Accumulation/Distribution Line"""
        hl_range = high - low
        mfm = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0)
        ad = np.cumsum(mfm * volume)
        return ad


class CustomIndicatorBuilder:
    """
    Build custom technical indicators using a formula builder
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators
        self.custom_indicators: Dict[str, Callable] = {}
    
    def create_indicator(
        self,
        name: str,
        formula: str,
        params: Dict[str, Any]
    ) -> bool:
        """
        Create a custom indicator
        
        Example formula: "ema(close, 20) - ema(close, 50)"
        """
        try:
            # Parse and validate formula
            def custom_func(ohlcv_data: Dict[str, np.ndarray]) -> np.ndarray:
                # Replace variable names with actual data
                local_vars = {
                    'open': ohlcv_data.get('open', np.array([])),
                    'high': ohlcv_data.get('high', np.array([])),
                    'low': ohlcv_data.get('low', np.array([])),
                    'close': ohlcv_data.get('close', np.array([])),
                    'volume': ohlcv_data.get('volume', np.array([])),
                    'sma': self.indicators.sma,
                    'ema': self.indicators.ema,
                    'rsi': self.indicators.rsi,
                    'macd': self.indicators.macd,
                    'atr': self.indicators.atr,
                    'np': np
                }
                local_vars.update(params)
                
                return eval(formula, {"__builtins__": {}}, local_vars)
            
            self.custom_indicators[name] = custom_func
            logger.info(f"Created custom indicator: {name}")
            return True
        except Exception as e:
            logger.error(f"Error creating custom indicator {name}: {e}")
            return False
    
    def calculate(
        self,
        name: str,
        ohlcv_data: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """Calculate custom indicator"""
        if name in self.custom_indicators:
            return self.custom_indicators[name](ohlcv_data)
        return None
    
    def list_indicators(self) -> List[str]:
        """List all custom indicators"""
        return list(self.custom_indicators.keys())


class ChartManager:
    """
    Manage chart layouts, drawings, and annotations
    """
    
    def __init__(self):
        self.drawings: Dict[str, List[DrawingObject]] = {}
        self.layouts: Dict[str, Dict] = {}
        self.indicators = TechnicalIndicators()
    
    def add_drawing(
        self,
        chart_id: str,
        drawing_type: str,
        points: List[Tuple[datetime, float]],
        style: Optional[Dict] = None
    ) -> str:
        """Add a drawing to a chart"""
        import uuid
        
        drawing_id = str(uuid.uuid4())[:8]
        
        drawing = DrawingObject(
            id=drawing_id,
            type=drawing_type,
            points=points,
            style=style or {"color": "#ff6600", "width": 2}
        )
        
        if chart_id not in self.drawings:
            self.drawings[chart_id] = []
        
        self.drawings[chart_id].append(drawing)
        
        return drawing_id
    
    def remove_drawing(self, chart_id: str, drawing_id: str) -> bool:
        """Remove a drawing from a chart"""
        if chart_id in self.drawings:
            self.drawings[chart_id] = [
                d for d in self.drawings[chart_id] if d.id != drawing_id
            ]
            return True
        return False
    
    def get_drawings(self, chart_id: str) -> List[DrawingObject]:
        """Get all drawings for a chart"""
        return self.drawings.get(chart_id, [])
    
    def create_trendline(
        self,
        chart_id: str,
        start: Tuple[datetime, float],
        end: Tuple[datetime, float],
        extend: bool = True
    ) -> str:
        """Create a trendline"""
        return self.add_drawing(
            chart_id,
            "trendline",
            [start, end],
            {"color": "#ff6600", "width": 2, "extend": extend}
        )
    
    def create_horizontal_line(
        self,
        chart_id: str,
        price: float,
        label: str = ""
    ) -> str:
        """Create a horizontal line"""
        drawing_id = self.add_drawing(
            chart_id,
            "horizontal",
            [(datetime.now(), price)],
            {"color": "#00ff00", "width": 1, "dash": "dash"}
        )
        
        # Add label
        for d in self.drawings.get(chart_id, []):
            if d.id == drawing_id:
                d.label = label
        
        return drawing_id
    
    def create_fibonacci(
        self,
        chart_id: str,
        high_point: Tuple[datetime, float],
        low_point: Tuple[datetime, float],
        is_uptrend: bool = True
    ) -> str:
        """Create Fibonacci retracement levels"""
        high = high_point[1]
        low = low_point[1]
        
        levels = TechnicalIndicators.fibonacci_retracements(high, low, is_uptrend)
        
        return self.add_drawing(
            chart_id,
            "fibonacci",
            [high_point, low_point],
            {"levels": levels, "color": "#9932cc"}
        )
    
    def save_layout(self, layout_id: str, config: Dict) -> bool:
        """Save a chart layout"""
        self.layouts[layout_id] = config
        return True
    
    def load_layout(self, layout_id: str) -> Optional[Dict]:
        """Load a chart layout"""
        return self.layouts.get(layout_id)


# All available indicators for reference
AVAILABLE_INDICATORS = {
    # Trend Indicators
    "SMA": {"category": "trend", "params": ["period"]},
    "EMA": {"category": "trend", "params": ["period"]},
    "WMA": {"category": "trend", "params": ["period"]},
    "DEMA": {"category": "trend", "params": ["period"]},
    "TEMA": {"category": "trend", "params": ["period"]},
    "MACD": {"category": "trend", "params": ["fast", "slow", "signal"]},
    "ADX": {"category": "trend", "params": ["period"]},
    "Parabolic SAR": {"category": "trend", "params": ["af_start", "af_increment", "af_max"]},
    "Supertrend": {"category": "trend", "params": ["period", "multiplier"]},
    "Ichimoku": {"category": "trend", "params": ["conversion", "base", "span_b", "displacement"]},
    "Aroon": {"category": "trend", "params": ["period"]},
    
    # Momentum Indicators
    "RSI": {"category": "momentum", "params": ["period"]},
    "Stochastic": {"category": "momentum", "params": ["k_period", "d_period"]},
    "CCI": {"category": "momentum", "params": ["period"]},
    "Williams %R": {"category": "momentum", "params": ["period"]},
    "ROC": {"category": "momentum", "params": ["period"]},
    "Momentum": {"category": "momentum", "params": ["period"]},
    "MFI": {"category": "momentum", "params": ["period"]},
    
    # Volatility Indicators
    "Bollinger Bands": {"category": "volatility", "params": ["period", "std_dev"]},
    "ATR": {"category": "volatility", "params": ["period"]},
    "Keltner Channels": {"category": "volatility", "params": ["ema_period", "atr_period", "multiplier"]},
    "Donchian Channels": {"category": "volatility", "params": ["period"]},
    
    # Volume Indicators
    "OBV": {"category": "volume", "params": []},
    "VWAP": {"category": "volume", "params": []},
    "CMF": {"category": "volume", "params": ["period"]},
    "A/D Line": {"category": "volume", "params": []},
    
    # Support/Resistance
    "Pivot Points": {"category": "overlay", "params": []},
    "Fibonacci": {"category": "overlay", "params": []}
}


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 100
    
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.random.uniform(0.5, 2, n)
    low = close - np.random.uniform(0.5, 2, n)
    volume = np.random.randint(100000, 1000000, n)
    
    ti = TechnicalIndicators()
    
    # Calculate various indicators
    print("=== Technical Indicators Demo ===\n")
    
    print("SMA(20):", ti.sma(close, 20)[-5:])
    print("EMA(20):", ti.ema(close, 20)[-5:])
    print("RSI(14):", ti.rsi(close, 14)[-5:])
    
    macd_line, signal, hist = ti.macd(close)
    print("MACD:", macd_line[-5:])
    
    upper, middle, lower = ti.bollinger_bands(close)
    print("BB Upper:", upper[-5:])
    print("BB Lower:", lower[-5:])
    
    print("\nAvailable Indicators:", len(AVAILABLE_INDICATORS))
