"""
Modeling Agent
===============
Run predictive stack (ML + statistical models).

Frequency: Every 15 minutes
Consumes: REGIME_DETECTED
Produces: MODEL_SIGNAL

Output:
{
  "signal": "LONG",
  "confidence": 0.67,
  "horizon": "5D"
}

Models must not run without regime label.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import Event, EventType

try:
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        RandomForestRegressor, GradientBoostingRegressor
    )
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Integration with core signal generator
try:
    from core.signal_generator import AlphaTableGenerator, generate_signals as core_generate_signals
    CORE_SIGNALS_AVAILABLE = True
except ImportError:
    CORE_SIGNALS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class ModelingAgent(BaseAgent):
    """
    Agent 5: Run predictive stack (ML + statistical models).
    
    Responsibilities:
    - Train/update ensemble models
    - Walk-forward validation
    - Model confidence scoring
    - Feature attribution (SHAP)
    """
    
    def __init__(self):
        super().__init__(
            name="ModelingAgent",
            subscriptions=[EventType.REGIME_DETECTED.value]
        )
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._model_scores: Dict[str, Dict] = {}
        self._last_signals: Dict[str, Dict] = {}
        self._regime_required = True  # Models must not run without regime label
    
    def initialize(self):
        """Initialize model ensemble."""
        if SKLEARN_AVAILABLE:
            self._models = {
                "rf_classifier": RandomForestClassifier(
                    n_estimators=100, max_depth=6, random_state=42
                ),
                "gb_classifier": GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
                ),
                "rf_regressor": RandomForestRegressor(
                    n_estimators=100, max_depth=6, random_state=42
                ),
                "gb_regressor": GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
                ),
            }
            self._log(f"Initialized {len(self._models)} models (sklearn available)")
        else:
            self._log("sklearn not available — using statistical fallback", level="warning")
        
        # Wire up core signal generator
        if CORE_SIGNALS_AVAILABLE:
            try:
                self._alpha_generator = AlphaTableGenerator()
                self._log("Core AlphaTableGenerator connected (multi-factor alpha signals)")
            except Exception as e:
                self._alpha_generator = None
                self._log(f"Core signal generator init failed: {e}", level="warning")
        else:
            self._alpha_generator = None
    
    def consume(self, event: Event):
        """React to REGIME_DETECTED — ready to model."""
        payload = event.get_payload()
        symbol = payload.get("symbol", "UNKNOWN")
        regime = payload.get("regime", None)
        
        if self._regime_required and regime is None:
            self._log(f"Regime not available for {symbol} — model BLOCKED", level="warning")
            return
        
        self._log(f"Model ready to run for {symbol} (regime={regime})")
    
    def produce(self) -> Optional[Event]:
        return None
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if SKLEARN_AVAILABLE else "degraded",
            "sklearn_available": SKLEARN_AVAILABLE,
            "models_loaded": list(self._models.keys()),
            "symbols_with_signals": list(self._last_signals.keys()),
        }
    
    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    
    def predict(self, feature_df: pd.DataFrame, symbol: str,
                regime: str, horizon: str = "5D",
                target_col: str = "returns_5d") -> Dict[str, Any]:
        """
        Generate prediction using ensemble.
        
        Args:
            feature_df: Feature matrix from FeatureEngineeringAgent
            symbol: Ticker symbol
            regime: Current market regime (REQUIRED)
            horizon: Prediction horizon
            target_col: Target variable column name
        
        Returns:
            Signal dict with direction, confidence, horizon
        """
        if self._regime_required and not regime:
            return {"error": "Regime label required before modeling"}
        
        # Prepare data
        df = feature_df.dropna()
        if target_col not in df.columns:
            # Create target from close if not present
            if 'close' in df.columns:
                period = int(horizon.replace("D", ""))
                df[target_col] = df['close'].pct_change(period).shift(-period)
                df = df.dropna()
            else:
                return {"error": f"Target column {target_col} not found"}
        
        # Separate features and target
        exclude_cols = [target_col, 'symbol', 'source', 'open', 'high', 'low', 'close',
                        'volume', 'adj_close', 'date']
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32']]
        
        if len(feature_cols) < 3:
            return {"error": "Insufficient features for modeling"}
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        if len(X) < 50:
            return {"error": "Insufficient data for modeling"}
        
        # Walk-forward validation
        result = self._walk_forward_predict(X, y, feature_cols, symbol, regime)
        result["symbol"] = symbol
        result["horizon"] = horizon
        result["regime"] = regime
        result["timestamp"] = datetime.now(datetime.timezone.utc).isoformat()
        
        self._last_signals[symbol] = result
        self._metrics.confidence_scores.append(result.get("confidence", 0))
        
        # Publish MODEL_SIGNAL
        self._publish(
            EventType.MODEL_SIGNAL.value,
            {
                "symbol": symbol,
                "signal": result.get("signal"),
                "confidence": result.get("confidence"),
                "horizon": horizon,
                "regime": regime,
            }
        )
        
        return result
    
    # ──────────────────────────────────────────────
    # Walk-forward validation
    # ──────────────────────────────────────────────
    
    def _walk_forward_predict(self, X: np.ndarray, y: np.ndarray,
                               feature_cols: List[str], symbol: str,
                               regime: str) -> Dict[str, Any]:
        """Walk-forward cross-validation and final prediction."""
        if not SKLEARN_AVAILABLE:
            return self._statistical_fallback(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scalers[symbol] = scaler
        
        # Binary classification target (up/down)
        y_class = (y > 0).astype(int)
        
        # Walk-forward splits
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_class[train_idx], y_class[val_idx]
            
            try:
                model = self._models.get("gb_classifier")
                if model:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    scores.append(accuracy_score(y_val, preds))
            except Exception as e:
                self._log(f"Walk-forward fold failed: {e}", level="warning")
        
        # Final prediction (use last observation)
        last_X = X_scaled[-1:].reshape(1, -1)
        
        # Ensemble predictions
        ensemble_probs = []
        for name, model in self._models.items():
            if "classifier" in name:
                try:
                    model.fit(X_scaled[:-1], y_class[:-1])
                    prob = model.predict_proba(last_X)[0]
                    ensemble_probs.append(prob[1])  # Probability of UP
                except Exception:
                    pass
        
        if not ensemble_probs:
            return self._statistical_fallback(y)
        
        avg_prob = np.mean(ensemble_probs)
        
        # Determine signal
        if avg_prob > 0.6:
            signal = SignalDirection.LONG.value
        elif avg_prob < 0.4:
            signal = SignalDirection.SHORT.value
        else:
            signal = SignalDirection.NEUTRAL.value
        
        confidence = abs(avg_prob - 0.5) * 2  # Scale to 0-1
        
        # Feature attribution (simple importance)
        feature_importance = {}
        try:
            gb = self._models.get("gb_classifier")
            if gb and hasattr(gb, 'feature_importances_'):
                for i, name in enumerate(feature_cols):
                    feature_importance[name] = round(float(gb.feature_importances_[i]), 4)
        except Exception:
            pass
        
        wf_score = np.mean(scores) if scores else 0.5
        
        self._model_scores[symbol] = {
            "walk_forward_accuracy": round(float(wf_score), 4),
            "ensemble_prob": round(float(avg_prob), 4),
            "n_models": len(ensemble_probs),
        }
        
        return {
            "signal": signal,
            "confidence": round(float(confidence), 4),
            "ensemble_prob": round(float(avg_prob), 4),
            "walk_forward_accuracy": round(float(wf_score), 4),
            "feature_importance": dict(sorted(feature_importance.items(),
                                              key=lambda x: x[1], reverse=True)[:10]),
        }
    
    def _statistical_fallback(self, y: np.ndarray) -> Dict[str, Any]:
        """Simple statistical prediction when sklearn is not available."""
        recent = y[-20:]
        mean_return = np.mean(recent)
        
        if mean_return > 0.01:
            signal = SignalDirection.LONG.value
        elif mean_return < -0.01:
            signal = SignalDirection.SHORT.value
        else:
            signal = SignalDirection.NEUTRAL.value
        
        return {
            "signal": signal,
            "confidence": round(min(abs(mean_return) * 10, 1.0), 4),
            "method": "statistical_fallback",
        }
    
    def generate_alpha_signals(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate multi-factor alpha signals using core.signal_generator.
        
        Returns individual signal components (momentum, mean_reversion,
        value, quality, volatility, etc.) alongside the ensemble prediction.
        """
        if not CORE_SIGNALS_AVAILABLE:
            return {"error": "Core signal generator not available"}
        
        try:
            signals = core_generate_signals(df, symbol)
            return {
                name: {
                    "value": round(float(s.value), 4),
                    "confidence": round(float(s.confidence), 4),
                    "type": s.signal_type.value,
                }
                for name, s in signals.items()
            }
        except Exception as e:
            self._log(f"Alpha signal generation failed: {e}", level="warning")
            return {"error": str(e)}
    
    def generate_alpha_table(self, data: Dict[str, pd.DataFrame],
                             symbols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Generate full alpha table using core AlphaTableGenerator.
        
        Returns DataFrame with combined alpha scores per symbol.
        """
        if not self._alpha_generator:
            self._log("Alpha table generator not available", level="warning")
            return None
        
        try:
            alpha_table = self._alpha_generator.generate_alpha_table(data, symbols)
            self._log(f"Generated alpha table: {len(alpha_table)} symbols")
            return alpha_table
        except Exception as e:
            self._log(f"Alpha table generation failed: {e}", level="error")
            return None
