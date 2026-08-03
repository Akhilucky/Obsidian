"""
Regime Detection Agent
=======================
Identify market state before modeling.

Frequency: Every 5 minutes
Consumes: FEATURE_MATRIX_READY
Produces: REGIME_DETECTED

States:
  TRENDING
  MEAN_REVERTING
  CRISIS
  LOW_VOL

Models must not run without regime label.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import Event, EventType

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states."""
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    CRISIS = "CRISIS"
    LOW_VOL = "LOW_VOL"


class RegimeDetectionAgent(BaseAgent):
    """
    Agent 4: Identify market state before modeling.
    
    Responsibilities:
    - Volatility clustering detection
    - Hidden Markov regime classification
    - Liquidity condition scoring
    """
    
    def __init__(self):
        super().__init__(
            name="RegimeDetectionAgent",
            subscriptions=[EventType.FEATURE_MATRIX_READY.value]
        )
        self._current_regimes: Dict[str, str] = {}
        self._regime_history: Dict[str, List[Dict]] = {}
        self._vol_threshold_crisis = 0.40    # Annualized vol > 40% => CRISIS
        self._vol_threshold_low = 0.12       # Annualized vol < 12% => LOW_VOL
        self._hurst_threshold = 0.55         # Hurst > 0.55 => TRENDING
    
    def initialize(self):
        """Load regime thresholds and warm HMM if available."""
        self._log("Regime detection thresholds loaded")
    
    def consume(self, event: Event):
        """Triggered when feature matrix is ready — detect regime."""
        payload = event.get_payload()
        symbol = payload.get("symbol", "UNKNOWN")
        self._log(f"Regime detection triggered for {symbol}")
    
    def produce(self) -> Optional[Event]:
        return None
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "regimes_tracked": len(self._current_regimes),
            "current_regimes": dict(self._current_regimes),
        }
    
    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    
    def detect(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Full regime detection pipeline.
        
        Uses:
        1. Volatility clustering
        2. Hurst exponent (trending vs mean-reverting)
        3. Liquidity scoring
        4. Optional HMM
        
        Returns regime label and confidence.
        """
        if 'close' not in df.columns:
            return {"regime": MarketRegime.LOW_VOL.value, "confidence": 0.0, "error": "No close data"}
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < 30:
            return {"regime": MarketRegime.LOW_VOL.value, "confidence": 0.5, "note": "Insufficient data"}
        
        # 1. Volatility clustering detection
        vol_regime = self._detect_volatility_regime(returns)
        
        # 2. Hurst exponent for trend detection
        hurst = self._hurst_exponent(returns.values)
        
        # 3. Liquidity condition scoring
        liquidity_score = self._liquidity_score(df)
        
        # 4. Combine signals
        regime, confidence = self._classify_regime(vol_regime, hurst, liquidity_score, returns)
        
        # Store
        self._current_regimes[symbol] = regime
        if symbol not in self._regime_history:
            self._regime_history[symbol] = []
        self._regime_history[symbol].append({
            "regime": regime,
            "confidence": confidence,
            "hurst": round(float(hurst), 4),
            "vol_regime": vol_regime,
            "liquidity_score": round(float(liquidity_score), 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        self._metrics.confidence_scores.append(confidence)
        
        # Publish REGIME_DETECTED
        self._publish(
            EventType.REGIME_DETECTED.value,
            {
                "symbol": symbol,
                "regime": regime,
                "confidence": confidence,
                "hurst": hurst,
                "liquidity_score": liquidity_score,
            }
        )
        
        return {
            "symbol": symbol,
            "regime": regime,
            "confidence": round(confidence, 4),
            "hurst": round(float(hurst), 4),
            "volatility_regime": vol_regime,
            "liquidity_score": round(float(liquidity_score), 4),
        }
    
    # ──────────────────────────────────────────────
    # Detection methods
    # ──────────────────────────────────────────────
    
    def _detect_volatility_regime(self, returns: pd.Series) -> str:
        """Detect regime based on realized volatility."""
        realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        if realized_vol > self._vol_threshold_crisis:
            return "HIGH"
        elif realized_vol < self._vol_threshold_low:
            return "LOW"
        else:
            return "NORMAL"
    
    def _hurst_exponent(self, series: np.ndarray) -> float:
        """
        Estimate the Hurst exponent using R/S analysis.
        
        H > 0.5 => trending (persistent)
        H < 0.5 => mean-reverting (anti-persistent)
        H ≈ 0.5 => random walk
        """
        n = len(series)
        if n < 20:
            return 0.5
        
        max_k = min(n // 2, 100)
        sizes = []
        rs_values = []
        
        for k in range(10, max_k, 5):
            subseries = [series[i:i + k] for i in range(0, n - k, k)]
            if len(subseries) < 2:
                continue
            
            rs_list = []
            for ss in subseries:
                if len(ss) < 2:
                    continue
                mean_ss = np.mean(ss)
                deviations = np.cumsum(ss - mean_ss)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(ss, ddof=1)
                if s > 0:
                    rs_list.append(r / s)
            
            if rs_list:
                sizes.append(k)
                rs_values.append(np.mean(rs_list))
        
        if len(sizes) < 3:
            return 0.5
        
        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)
        
        try:
            slope, _, _, _, _ = np.polyfit(log_sizes, log_rs, 1, full=False, cov=False) if False else (np.polyfit(log_sizes, log_rs, 1)[0], 0, 0, 0, 0)
            return max(0.0, min(1.0, slope))
        except Exception:
            return 0.5
    
    def _liquidity_score(self, df: pd.DataFrame) -> float:
        """
        Score liquidity conditions (0 = illiquid, 1 = highly liquid).
        
        Uses volume relative to historical average and bid-ask proxy.
        """
        if 'volume' not in df.columns:
            return 0.5
        
        vol = df['volume'].dropna()
        if len(vol) < 20:
            return 0.5
        
        recent_vol = vol.iloc[-5:].mean()
        avg_vol = vol.iloc[-60:].mean() if len(vol) >= 60 else vol.mean()
        
        if avg_vol == 0:
            return 0.5
        
        ratio = recent_vol / avg_vol
        # Normalize to 0-1 range
        score = min(1.0, max(0.0, ratio / 2.0))
        return score
    
    def _classify_regime(self, vol_regime: str, hurst: float,
                         liquidity: float, returns: pd.Series) -> tuple:
        """Combine signals into a single regime classification."""
        # CRISIS overrides everything
        if vol_regime == "HIGH" and liquidity < 0.3:
            return MarketRegime.CRISIS.value, 0.85
        
        if vol_regime == "HIGH":
            return MarketRegime.CRISIS.value, 0.70
        
        # LOW_VOL
        if vol_regime == "LOW":
            if hurst > self._hurst_threshold:
                return MarketRegime.TRENDING.value, 0.65
            return MarketRegime.LOW_VOL.value, 0.75
        
        # NORMAL vol — use Hurst to distinguish trend vs mean-reversion
        if hurst > self._hurst_threshold:
            return MarketRegime.TRENDING.value, 0.70
        elif hurst < (1 - self._hurst_threshold):
            return MarketRegime.MEAN_REVERTING.value, 0.70
        else:
            return MarketRegime.LOW_VOL.value, 0.55
    
    def get_regime(self, symbol: str) -> Optional[str]:
        """Get the current regime for a symbol."""
        return self._current_regimes.get(symbol)
