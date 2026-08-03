"""
Data Quality Agent
===================
Ensure institutional-grade data integrity.

Frequency: Realtime
Consumes: DATA_INGESTED
Produces: DATA_VALIDATED

Output:
{
  "event": "DATA_VALIDATED",
  "confidence": 0.94,
  "anomalies": 2
}
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

logger = logging.getLogger(__name__)

# Thresholds
MAX_MISSING_PCT = 0.05        # 5% max missing data
OUTLIER_Z_THRESHOLD = 4.0     # z-score for outlier rejection
MIN_CONFIDENCE = 0.50         # Below this we reject the data


class DataQualityAgent(BaseAgent):
    """
    Agent 2: Ensure institutional-grade data integrity.
    
    Responsibilities:
    - Missing data detection
    - Outlier rejection
    - Source reconciliation
    - Confidence scoring
    """
    
    def __init__(self):
        super().__init__(
            name="DataQualityAgent",
            subscriptions=[EventType.DATA_INGESTED.value]
        )
        self._validation_results: Dict[str, Dict] = {}
    
    def initialize(self):
        """Load validation rules and thresholds."""
        self._log("Quality rules loaded")
    
    def consume(self, event: Event):
        """Validate incoming data referenced by DATA_INGESTED event."""
        payload = event.get_payload()
        symbol = payload.get("symbol", "UNKNOWN")
        rows = payload.get("rows", 0)
        columns = payload.get("columns", [])
        source = payload.get("source", "unknown")
        
        self._log(f"Validating data for {symbol} ({rows} rows, source={source})")
        
        # Run quality checks (on metadata — real impl would access the dataframe)
        checks = {
            "has_rows": rows > 0,
            "has_ohlc": all(c in columns for c in ['open', 'high', 'low', 'close']),
            "has_volume": 'volume' in columns,
            "source_known": source in ["yahoo", "openbb", "fred", "coingecko"],
        }
        
        anomalies = sum(1 for v in checks.values() if not v)
        confidence = 1.0 - (anomalies / max(len(checks), 1))
        
        result = {
            "symbol": symbol,
            "checks": checks,
            "anomalies": anomalies,
            "confidence": round(confidence, 4),
            "source": source,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._validation_results[symbol] = result
        self._metrics.confidence_scores.append(confidence)
        
        # Publish validation result
        self._publish(
            EventType.DATA_VALIDATED.value,
            {
                "symbol": symbol,
                "confidence": confidence,
                "anomalies": anomalies,
                "checks": checks,
            }
        )
        
        if confidence < MIN_CONFIDENCE:
            self._log(f"Data for {symbol} REJECTED (confidence={confidence})", level="warning")
    
    def produce(self) -> Optional[Event]:
        """Produce is handled inside consume via _publish."""
        return None
    
    def health_check(self) -> Dict[str, Any]:
        recent_conf = self._metrics.confidence_scores[-20:] if self._metrics.confidence_scores else [1.0]
        avg_confidence = np.mean(recent_conf)
        return {
            "status": "healthy" if avg_confidence > MIN_CONFIDENCE else "degraded",
            "avg_confidence": round(float(avg_confidence), 4),
            "symbols_validated": len(self._validation_results),
        }
    
    # ──────────────────────────────────────────────
    # Deep validation (used when actual dataframe is available)
    # ──────────────────────────────────────────────
    
    def validate(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Alias for validate_dataframe — used by orchestrator."""
        return self.validate_dataframe(df, symbol)
    
    def validate_dataframe(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Full validation suite on an actual DataFrame.
        
        Returns dict with confidence score and anomaly details.
        """
        issues = []
        
        # 1. Missing data detection
        missing_pct = df.isnull().mean()
        for col, pct in missing_pct.items():
            if pct > MAX_MISSING_PCT:
                issues.append({
                    "type": "MISSING_DATA",
                    "column": col,
                    "missing_pct": round(float(pct), 4),
                })
        
        # 2. Outlier rejection (z-score on returns if close exists)
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            if len(returns) > 10:
                z_scores = np.abs((returns - returns.mean()) / returns.std())
                outliers = z_scores[z_scores > OUTLIER_Z_THRESHOLD]
                if len(outliers) > 0:
                    issues.append({
                        "type": "OUTLIER",
                        "count": int(len(outliers)),
                        "max_z": round(float(z_scores.max()), 2),
                    })
        
        # 3. Monotonicity check on dates
        if isinstance(df.index, pd.DatetimeIndex):
            if not df.index.is_monotonic_increasing:
                issues.append({"type": "DATE_ORDER", "detail": "Index not sorted"})
        
        # 4. Duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append({"type": "DUPLICATES", "count": int(dup_count)})
        
        # 5. OHLC sanity: high >= low, high >= open, high >= close
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            bad_rows = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close'])
            ).sum()
            if bad_rows > 0:
                issues.append({"type": "OHLC_INVALID", "bad_rows": int(bad_rows)})
        
        anomalies = len(issues)
        max_anomalies = 5
        confidence = max(0.0, 1.0 - (anomalies / max_anomalies))
        
        result = {
            "symbol": symbol,
            "confidence": round(confidence, 4),
            "anomalies": anomalies,
            "issues": issues,
            "rows": len(df),
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        self._validation_results[symbol] = result
        self._metrics.confidence_scores.append(confidence)
        
        # Publish
        self._publish(
            EventType.DATA_VALIDATED.value,
            {
                "symbol": symbol,
                "confidence": confidence,
                "anomalies": anomalies,
            }
        )
        
        return result
    
    def reconcile_sources(self, dfs: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """
        Source reconciliation — compare data from multiple providers.
        
        Returns reconciliation report.
        """
        if len(dfs) < 2:
            return {"reconciled": True, "note": "Single source, nothing to reconcile"}
        
        sources = list(dfs.keys())
        close_cols = {}
        for src, df in dfs.items():
            if 'close' in df.columns:
                close_cols[src] = df['close']
        
        if len(close_cols) < 2:
            return {"reconciled": True, "note": "Insufficient close data for reconciliation"}
        
        # Compare pairwise correlation
        src_names = list(close_cols.keys())
        correlations = {}
        for i in range(len(src_names)):
            for j in range(i + 1, len(src_names)):
                a = close_cols[src_names[i]]
                b = close_cols[src_names[j]]
                common = pd.concat([a, b], axis=1).dropna()
                if len(common) > 5:
                    corr = common.iloc[:, 0].corr(common.iloc[:, 1])
                    correlations[f"{src_names[i]}_vs_{src_names[j]}"] = round(float(corr), 4)
        
        all_high = all(c > 0.99 for c in correlations.values()) if correlations else True
        
        return {
            "symbol": symbol,
            "reconciled": all_high,
            "correlations": correlations,
            "sources": src_names,
        }
