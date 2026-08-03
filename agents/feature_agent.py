"""
Feature Engineering Agent
==========================
Generate signals and model-ready features.

Frequency: Follows DATA_VALIDATED
Consumes: DATA_VALIDATED
Produces: FEATURE_MATRIX_READY

Output:
  FEATURE_MATRIX_READY
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import Event, EventType

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# Integration with core feature store
try:
    from core.feature_store import FeatureStore, compute_features as core_compute_features
    CORE_FEATURE_STORE = True
except ImportError:
    CORE_FEATURE_STORE = False

logger = logging.getLogger(__name__)


class FeatureEngineeringAgent(BaseAgent):
    """
    Agent 3: Generate signals and model-ready features.
    
    Integrates with core.feature_store.FeatureStore for 100+
    pre-built features with versioning and lineage tracking.
    
    Responsibilities:
    - Technical indicators (RSI, MACD, etc.)
    - Sentiment fusion
    - Factor exposures
    - Regime-sensitive feature scaling
    """
    
    def __init__(self):
        super().__init__(
            name="FeatureEngineeringAgent",
            subscriptions=[EventType.DATA_VALIDATED.value]
        )
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._feature_registry: Dict[str, callable] = {}
        self._core_store: Optional[Any] = None
    
    def initialize(self):
        """Register all feature computation functions."""
        self._feature_registry = {
            # Price-based
            "returns_1d": self._returns,
            "returns_5d": lambda df: self._returns(df, period=5),
            "returns_20d": lambda df: self._returns(df, period=20),
            "log_returns": self._log_returns,
            
            # Volatility
            "volatility_20d": lambda df: self._rolling_vol(df, window=20),
            "volatility_60d": lambda df: self._rolling_vol(df, window=60),
            "atr_14": self._atr,
            
            # Momentum
            "rsi_14": lambda df: self._rsi(df, period=14),
            "rsi_28": lambda df: self._rsi(df, period=28),
            "macd": self._macd,
            "macd_signal": self._macd_signal,
            "macd_hist": self._macd_hist,
            
            # Trend
            "sma_20": lambda df: self._sma(df, window=20),
            "sma_50": lambda df: self._sma(df, window=50),
            "sma_200": lambda df: self._sma(df, window=200),
            "ema_12": lambda df: self._ema(df, span=12),
            "ema_26": lambda df: self._ema(df, span=26),
            
            # Volume
            "volume_sma_20": self._volume_sma,
            "obv": self._obv,
            
            # Bollinger Bands
            "bb_upper": lambda df: self._bollinger(df, "upper"),
            "bb_lower": lambda df: self._bollinger(df, "lower"),
            "bb_width": lambda df: self._bollinger(df, "width"),
            
            # Mean reversion
            "z_score_20": lambda df: self._z_score(df, window=20),
            "z_score_60": lambda df: self._z_score(df, window=60),
        }
        
        # Wire up core feature store if available (100+ features)
        if CORE_FEATURE_STORE:
            try:
                self._core_store = FeatureStore()
                self._log(f"Core FeatureStore connected ({len(self._core_store.registry.list_features())} features available)")
            except Exception as e:
                self._log(f"Core FeatureStore init failed: {e}", level="warning")
        
        self._log(f"Registered {len(self._feature_registry)} agent features" +
                   (f" + core store" if self._core_store else ""))
    
    def consume(self, event: Event):
        """React to DATA_VALIDATED — features will be built on demand via compute()."""
        payload = event.get_payload()
        symbol = payload.get("symbol", "UNKNOWN")
        confidence = payload.get("confidence", 0)
        
        if confidence < 0.5:
            self._log(f"Skipping feature engineering for {symbol} (low confidence={confidence})", level="warning")
            return
        
        self._log(f"Ready to compute features for {symbol} (confidence={confidence})")
    
    def produce(self) -> Optional[Event]:
        return None
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "registered_features": len(self._feature_registry),
            "cached_symbols": list(self._feature_cache.keys()),
            "ta_available": TA_AVAILABLE,
        }
    
    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    
    def compute(self, df: pd.DataFrame, symbol: str,
                features: Optional[List[str]] = None,
                regime: Optional[str] = None) -> pd.DataFrame:
        """
        Compute feature matrix for a symbol.
        
        Uses both agent-level features and core FeatureStore
        (100+ pre-built features with versioning).
        
        Args:
            df: OHLCV DataFrame
            symbol: Ticker symbol
            features: Specific features to compute (None = all)
            regime: Current regime for regime-sensitive scaling
        
        Returns:
            DataFrame with computed features
        """
        feature_names = features or list(self._feature_registry.keys())
        result = df.copy()
        
        for fname in feature_names:
            if fname in self._feature_registry:
                try:
                    result[fname] = self._feature_registry[fname](df)
                except Exception as e:
                    self._log(f"Feature {fname} failed for {symbol}: {e}", level="warning")
                    result[fname] = np.nan
        
        # Also compute core features if available
        if self._core_store and not features:
            try:
                core_df = self._core_store.compute_all_features(df, symbol=symbol)
                # Merge core features that don't overlap
                for col in core_df.columns:
                    if col not in result.columns:
                        result[col] = core_df[col]
                self._log(f"Added {len(core_df.columns)} core features for {symbol}")
            except Exception as e:
                self._log(f"Core feature computation failed: {e}", level="warning")
        
        # Regime-sensitive feature scaling
        if regime:
            result = self._regime_scale(result, regime)
        
        self._feature_cache[symbol] = result
        
        # Publish FEATURE_MATRIX_READY
        self._publish(
            EventType.FEATURE_MATRIX_READY.value,
            {
                "symbol": symbol,
                "features_computed": len(feature_names),
                "rows": len(result),
                "regime": regime,
            }
        )
        
        return result
    
    # ──────────────────────────────────────────────
    # Feature computation functions
    # ──────────────────────────────────────────────
    
    def _returns(self, df: pd.DataFrame, period: int = 1) -> pd.Series:
        return df['close'].pct_change(period)
    
    def _log_returns(self, df: pd.DataFrame) -> pd.Series:
        return np.log(df['close'] / df['close'].shift(1))
    
    def _rolling_vol(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        return df['close'].pct_change().rolling(window).std() * np.sqrt(252)
    
    def _rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        if TA_AVAILABLE:
            return ta.momentum.rsi(df['close'], window=period)
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def _macd(self, df: pd.DataFrame) -> pd.Series:
        return df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    
    def _macd_signal(self, df: pd.DataFrame) -> pd.Series:
        macd = self._macd(df)
        return macd.ewm(span=9).mean()
    
    def _macd_hist(self, df: pd.DataFrame) -> pd.Series:
        return self._macd(df) - self._macd_signal(df)
    
    def _sma(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df['close'].rolling(window).mean()
    
    def _ema(self, df: pd.DataFrame, span: int) -> pd.Series:
        return df['close'].ewm(span=span).mean()
    
    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _volume_sma(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        return df['volume'].rolling(window).mean()
    
    def _obv(self, df: pd.DataFrame) -> pd.Series:
        direction = np.sign(df['close'].diff())
        return (df['volume'] * direction).cumsum()
    
    def _bollinger(self, df: pd.DataFrame, band: str, window: int = 20, std: float = 2.0):
        sma = df['close'].rolling(window).mean()
        rolling_std = df['close'].rolling(window).std()
        if band == "upper":
            return sma + std * rolling_std
        elif band == "lower":
            return sma - std * rolling_std
        elif band == "width":
            return (2 * std * rolling_std) / sma
    
    def _z_score(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        return (df['close'] - sma) / std.replace(0, np.nan)
    
    def _regime_scale(self, df: pd.DataFrame, regime: str) -> pd.DataFrame:
        """
        Regime-sensitive feature scaling.
        
        CRISIS: scale down momentum features, scale up volatility
        TRENDING: scale up trend features
        MEAN_REVERTING: scale up z-score features
        LOW_VOL: standard scaling
        """
        scaling = {
            "CRISIS": {"volatility": 1.5, "momentum": 0.5, "trend": 0.7},
            "TRENDING": {"volatility": 1.0, "momentum": 1.2, "trend": 1.5},
            "MEAN_REVERTING": {"volatility": 1.0, "momentum": 0.8, "trend": 0.8},
            "LOW_VOL": {"volatility": 1.0, "momentum": 1.0, "trend": 1.0},
        }
        
        scales = scaling.get(regime, scaling["LOW_VOL"])
        
        vol_features = [c for c in df.columns if 'volatility' in c or 'atr' in c or 'bb_' in c]
        mom_features = [c for c in df.columns if 'rsi' in c or 'macd' in c or 'returns' in c]
        trend_features = [c for c in df.columns if 'sma' in c or 'ema' in c]
        
        for col in vol_features:
            if col in df.columns:
                df[col] = df[col] * scales["volatility"]
        for col in mom_features:
            if col in df.columns:
                df[col] = df[col] * scales["momentum"]
        for col in trend_features:
            if col in df.columns:
                df[col] = df[col] * scales["trend"]
        
        return df
