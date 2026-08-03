"""
Monitoring Agent
=================
Detect drift and degradation post-deployment.

Frequency: Daily
Consumes: MODEL_SIGNAL, SCENARIO_RESULT, ALERT_GENERATED
Produces: DRIFT_DETECTED, ALERT_GENERATED

Responsibilities:
- Model decay detection
- Live vs backtest divergence
- Feature predictive power tracking
- Alert generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import deque
from enum import Enum

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import Event, EventType

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DriftType(Enum):
    MODEL_DECAY = "MODEL_DECAY"
    FEATURE_DRIFT = "FEATURE_DRIFT"
    PERFORMANCE_DIVERGENCE = "PERFORMANCE_DIVERGENCE"
    DATA_QUALITY_DROP = "DATA_QUALITY_DROP"


# Thresholds
MODEL_DECAY_THRESHOLD = 0.15        # Accuracy drop > 15% → alert
FEATURE_DRIFT_Z_THRESHOLD = 3.0     # Feature z-score > 3 → drift
PERFORMANCE_DIVERGENCE_PCT = 0.20   # Live vs backtest > 20% divergence


class MonitoringAgent(BaseAgent):
    """
    Agent 9: Detect drift and degradation post-deployment.
    
    Responsibilities:
    - Model decay detection
    - Live vs backtest divergence
    - Feature predictive power tracking
    - Alert generation
    """
    
    def __init__(self):
        super().__init__(
            name="MonitoringAgent",
            subscriptions=[
                EventType.MODEL_SIGNAL.value,
                EventType.SCENARIO_RESULT.value,
            ]
        )
        self._signal_history: deque = deque(maxlen=500)
        self._performance_log: Dict[str, List[Dict]] = {}
        self._feature_baselines: Dict[str, Dict[str, float]] = {}
        self._alerts: List[Dict] = []
        self._backtest_benchmarks: Dict[str, float] = {}
    
    def initialize(self):
        """Load baseline metrics and benchmarks."""
        self._log("Monitoring baselines loaded")
    
    def consume(self, event: Event):
        """Track signals and scenarios for drift detection."""
        payload = event.get_payload()
        
        if event.event_type == EventType.MODEL_SIGNAL.value:
            self._signal_history.append({
                "symbol": payload.get("symbol"),
                "signal": payload.get("signal"),
                "confidence": payload.get("confidence"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self._check_model_decay(payload)
        
        elif event.event_type == EventType.SCENARIO_RESULT.value:
            resilience = payload.get("resilience_score", 0)
            if resilience < 0.3:
                self._generate_alert(
                    AlertSeverity.WARNING,
                    f"Low portfolio resilience: {resilience:.2f}",
                    {"resilience_score": resilience}
                )
    
    def produce(self) -> Optional[Event]:
        return None
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "signals_tracked": len(self._signal_history),
            "active_alerts": len([a for a in self._alerts if not a.get("resolved")]),
            "total_alerts": len(self._alerts),
        }
    
    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    
    def check_model_decay(self, symbol: str,
                          live_accuracy: float,
                          backtest_accuracy: Optional[float] = None) -> Dict[str, Any]:
        """
        Compare live model performance against backtest baseline.
        """
        baseline = backtest_accuracy or self._backtest_benchmarks.get(symbol, 0.55)
        decay = baseline - live_accuracy
        
        result = {
            "symbol": symbol,
            "live_accuracy": round(live_accuracy, 4),
            "backtest_accuracy": round(baseline, 4),
            "decay": round(decay, 4),
            "decayed": decay > MODEL_DECAY_THRESHOLD,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if result["decayed"]:
            self._generate_alert(
                AlertSeverity.CRITICAL,
                f"Model decay for {symbol}: {decay:.1%} drop",
                result
            )
            self._publish(
                EventType.DRIFT_DETECTED.value,
                {
                    "symbol": symbol,
                    "drift_type": DriftType.MODEL_DECAY.value,
                    "decay": decay,
                }
            )
        
        return result
    
    def check_feature_drift(self, current_features: pd.DataFrame,
                            baseline_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect feature distribution drifts using KS test and z-scores.
        """
        drifted_features = []
        
        for col in current_features.select_dtypes(include=[np.number]).columns:
            current = current_features[col].dropna()
            
            if col in self._feature_baselines:
                base_mean = self._feature_baselines[col].get("mean", current.mean())
                base_std = self._feature_baselines[col].get("std", current.std())
            elif baseline_features is not None and col in baseline_features.columns:
                base = baseline_features[col].dropna()
                base_mean = base.mean()
                base_std = base.std()
            else:
                continue
            
            if base_std == 0:
                continue
            
            current_mean = current.mean()
            z_score = abs(current_mean - base_mean) / base_std
            
            if z_score > FEATURE_DRIFT_Z_THRESHOLD:
                drifted_features.append({
                    "feature": col,
                    "z_score": round(float(z_score), 2),
                    "current_mean": round(float(current_mean), 4),
                    "baseline_mean": round(float(base_mean), 4),
                })
        
        has_drift = len(drifted_features) > 0
        
        if has_drift:
            self._publish(
                EventType.DRIFT_DETECTED.value,
                {
                    "drift_type": DriftType.FEATURE_DRIFT.value,
                    "drifted_features": len(drifted_features),
                    "features": [d["feature"] for d in drifted_features],
                }
            )
        
        return {
            "drifted": has_drift,
            "drifted_count": len(drifted_features),
            "details": drifted_features,
        }
    
    def check_performance_divergence(self, symbol: str,
                                      live_returns: pd.Series,
                                      backtest_returns: pd.Series) -> Dict[str, Any]:
        """
        Compare live performance vs backtest expectations.
        """
        live_sharpe = self._sharpe(live_returns)
        bt_sharpe = self._sharpe(backtest_returns)
        
        if bt_sharpe == 0:
            divergence = 0
        else:
            divergence = abs(live_sharpe - bt_sharpe) / abs(bt_sharpe)
        
        result = {
            "symbol": symbol,
            "live_sharpe": round(float(live_sharpe), 4),
            "backtest_sharpe": round(float(bt_sharpe), 4),
            "divergence_pct": round(float(divergence), 4),
            "diverged": divergence > PERFORMANCE_DIVERGENCE_PCT,
        }
        
        if result["diverged"]:
            self._generate_alert(
                AlertSeverity.WARNING,
                f"Performance divergence for {symbol}: {divergence:.1%}",
                result
            )
            self._publish(
                EventType.DRIFT_DETECTED.value,
                {
                    "symbol": symbol,
                    "drift_type": DriftType.PERFORMANCE_DIVERGENCE.value,
                    "divergence_pct": divergence,
                }
            )
        
        return result
    
    def set_baseline(self, feature_name: str, mean: float, std: float):
        """Set feature baseline for drift detection."""
        self._feature_baselines[feature_name] = {"mean": mean, "std": std}
    
    def set_backtest_benchmark(self, symbol: str, accuracy: float):
        """Set backtest accuracy benchmark for a symbol."""
        self._backtest_benchmarks[symbol] = accuracy
    
    def get_alerts(self, severity: Optional[str] = None,
                   unresolved_only: bool = True) -> List[Dict]:
        """Get alerts, optionally filtered."""
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        if unresolved_only:
            alerts = [a for a in alerts if not a.get("resolved")]
        return alerts
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self._alerts:
            if alert.get("id") == alert_id:
                alert["resolved"] = True
                alert["resolved_at"] = datetime.now(timezone.utc).isoformat()
                break
    
    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────
    
    def _check_model_decay(self, signal_payload: Dict):
        """Quick check on incoming signal confidence trend."""
        recent = list(self._signal_history)[-50:]
        if len(recent) < 10:
            return
        
        confidences = [s.get("confidence", 0) for s in recent if s.get("confidence")]
        if not confidences:
            return
        
        # Check if confidence is trending down
        first_half = np.mean(confidences[:len(confidences)//2])
        second_half = np.mean(confidences[len(confidences)//2:])
        
        if first_half > 0 and (first_half - second_half) / first_half > MODEL_DECAY_THRESHOLD:
            self._generate_alert(
                AlertSeverity.WARNING,
                f"Model confidence declining: {first_half:.2f} → {second_half:.2f}",
                {"first_half_avg": first_half, "second_half_avg": second_half}
            )
    
    def _generate_alert(self, severity: AlertSeverity, message: str,
                        details: Dict[str, Any]):
        """Generate and store an alert."""
        import uuid
        alert = {
            "id": str(uuid.uuid4())[:8],
            "severity": severity.value,
            "message": message,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved": False,
        }
        self._alerts.append(alert)
        self._log(f"ALERT [{severity.value}]: {message}", 
                  level="warning" if severity != AlertSeverity.CRITICAL else "error")
        
        self._publish(
            EventType.ALERT_GENERATED.value,
            {
                "severity": severity.value,
                "message": message,
            }
        )
    
    def _sharpe(self, returns: pd.Series, rf: float = 0.0) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess = returns - rf / 252
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(252))
