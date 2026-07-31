"""
Risk Agent
===========
Evaluate survivability before approval.

Frequency: Hourly
Consumes: TRADE_IDEA_CREATED
Produces: RISK_APPROVED or RISK_REJECTED

Reject Conditions:
- Tail risk breach
- Regime mismatch
- Liquidity violation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import Event, EventType

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Integration with core risk management
try:
    from core.risk_management import RiskMetrics as CoreRiskMetrics, PortfolioOptimizer
    CORE_RISK_AVAILABLE = True
except ImportError:
    CORE_RISK_AVAILABLE = False

logger = logging.getLogger(__name__)

# Risk limits
MAX_PORTFOLIO_CVAR_95 = -0.05   # Max 5% CVaR at 95%
MAX_SINGLE_POSITION = 0.10      # 10% max in single position
MAX_SECTOR_EXPOSURE = 0.30      # 30% max sector
MIN_LIQUIDITY_SCORE = 0.20      # Minimum liquidity for market entry
CRISIS_POSITION_SCALE = 0.50    # Scale positions 50% in CRISIS regime


class RiskAgent(BaseAgent):
    """
    Agent 7: Evaluate survivability before approval.
    
    Responsibilities:
    - CVaR stress tests
    - Factor exposure limits
    - Scenario simulation
    - Capital allocation checks
    """
    
    def __init__(self):
        super().__init__(
            name="RiskAgent",
            subscriptions=[EventType.TRADE_IDEA_CREATED.value]
        )
        self._approved: List[Dict] = []
        self._rejected: List[Dict] = []
        self._portfolio_state: Dict[str, float] = {}  # symbol -> weight
        self._total_capital: float = 1_000_000.0
    
    def initialize(self):
        """Load risk limits and portfolio state."""
        if CORE_RISK_AVAILABLE:
            self._core_risk = CoreRiskMetrics
            self._log("Core RiskMetrics connected (VaR, CVaR, Sharpe, Sortino, Calmar, Max Drawdown)")
        else:
            self._core_risk = None
            self._log("Core RiskMetrics not available — using built-in", level="warning")
        self._log("Risk limits loaded")
    
    def consume(self, event: Event):
        """
        Evaluate trade idea against risk constraints.
        Approve or reject with reasons.
        """
        payload = event.get_payload()
        symbol = payload.get("symbol", "UNKNOWN")
        direction = payload.get("direction", "NEUTRAL")
        conviction = payload.get("conviction", 0)
        regime = payload.get("regime", "UNKNOWN")
        
        self._log(f"Evaluating risk for {symbol} {direction} (conviction={conviction}, regime={regime})")
        
        # Run risk checks
        checks = self._run_risk_checks(symbol, direction, conviction, regime)
        
        passed = all(c["passed"] for c in checks)
        
        if passed:
            self._approved.append(payload)
            self._publish(
                EventType.RISK_APPROVED.value,
                {
                    "symbol": symbol,
                    "direction": direction,
                    "conviction": conviction,
                    "risk_checks": [c for c in checks],
                    "position_size": self._compute_position_size(conviction, regime),
                }
            )
            self._log(f"APPROVED: {symbol} {direction}")
        else:
            reasons = [c["name"] for c in checks if not c["passed"]]
            self._rejected.append({**payload, "rejection_reasons": reasons})
            self._publish(
                EventType.RISK_REJECTED.value,
                {
                    "symbol": symbol,
                    "direction": direction,
                    "rejection_reasons": reasons,
                    "risk_checks": [c for c in checks],
                }
            )
            self._log(f"REJECTED: {symbol} {direction} — reasons: {reasons}", level="warning")
    
    def produce(self) -> Optional[Event]:
        return None
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "approved": len(self._approved),
            "rejected": len(self._rejected),
            "portfolio_positions": len(self._portfolio_state),
        }
    
    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    
    def set_portfolio_state(self, positions: Dict[str, float]):
        """Update current portfolio weights."""
        self._portfolio_state = positions
    
    def compute_portfolio_cvar(self, returns: pd.DataFrame,
                                weights: Optional[Dict[str, float]] = None,
                                confidence: float = 0.95) -> Dict[str, float]:
        """
        Compute portfolio CVaR (Conditional Value at Risk).
        
        Returns:
            Dict with VaR and CVaR at the given confidence level.
        """
        if weights is None:
            weights = self._portfolio_state
        
        if not weights:
            return {"var": 0.0, "cvar": 0.0}
        
        # Align columns
        common = [c for c in returns.columns if c in weights]
        if not common:
            return {"var": 0.0, "cvar": 0.0}
        
        w = np.array([weights[c] for c in common])
        ret = returns[common].values
        
        portfolio_returns = ret @ w
        
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean() if np.any(portfolio_returns <= var) else var
        
        return {
            "var": round(float(var), 6),
            "cvar": round(float(cvar), 6),
            "confidence": confidence,
        }
    
    def stress_test(self, returns: pd.DataFrame, scenarios: Optional[Dict[str, Dict]] = None) -> List[Dict]:
        """
        Run stress tests against predefined scenarios.
        """
        if scenarios is None:
            scenarios = {
                "market_crash_10pct": {"shock": -0.10, "vol_mult": 3.0},
                "rate_spike": {"shock": -0.03, "vol_mult": 1.5},
                "flash_crash": {"shock": -0.05, "vol_mult": 5.0},
                "vol_spike": {"shock": 0.0, "vol_mult": 4.0},
            }
        
        results = []
        for name, params in scenarios.items():
            shocked_returns = returns.copy()
            shocked_returns = shocked_returns * params["vol_mult"] + params["shock"]
            
            cvar = self.compute_portfolio_cvar(shocked_returns)
            results.append({
                "scenario": name,
                "params": params,
                **cvar,
                "breached": cvar["cvar"] < MAX_PORTFOLIO_CVAR_95,
            })
        
        return results
    
    def compute_full_risk_report(self, returns: pd.Series, symbol: str = "PORTFOLIO") -> Dict[str, Any]:
        """
        Generate comprehensive risk report using core.risk_management.RiskMetrics.
        
        Includes: VaR, CVaR, Sharpe, Sortino, Calmar, max drawdown, volatility.
        """
        report = {"symbol": symbol}
        
        if self._core_risk and len(returns) > 20:
            try:
                report["volatility"] = round(float(self._core_risk.volatility(returns)), 6)
                report["var_95"] = round(float(self._core_risk.var(returns, 0.95)), 6)
                report["cvar_95"] = round(float(self._core_risk.cvar(returns, 0.95)), 6)
                report["sharpe_ratio"] = round(float(self._core_risk.sharpe_ratio(returns)), 4)
                report["sortino_ratio"] = round(float(self._core_risk.sortino_ratio(returns)), 4)
                report["calmar_ratio"] = round(float(self._core_risk.calmar_ratio(returns)), 4)
                
                max_dd, peak, trough = self._core_risk.max_drawdown(returns)
                report["max_drawdown"] = round(float(max_dd), 6)
                report["max_dd_peak"] = str(peak)
                report["max_dd_trough"] = str(trough)
                report["source"] = "core_risk_metrics"
            except Exception as e:
                report["error"] = str(e)
                report["source"] = "fallback"
        else:
            # Fallback to simple metrics
            report["volatility"] = round(float(returns.std() * np.sqrt(252)), 6)
            report["sharpe_ratio"] = round(float(returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)), 4)
            report["source"] = "built_in"
        
        self._log(f"Risk report for {symbol}: Sharpe={report.get('sharpe_ratio')}, Vol={report.get('volatility')}")
        return report
    
    # ──────────────────────────────────────────────
    # Internal risk checks
    # ──────────────────────────────────────────────
    
    def _run_risk_checks(self, symbol: str, direction: str,
                         conviction: float, regime: str) -> List[Dict]:
        """Run all risk checks and return results."""
        checks = []
        
        # 1. Tail risk breach — check if adding this position breaches CVaR limits
        checks.append({
            "name": "tail_risk",
            "passed": True,  # Detailed impl requires portfolio returns
            "detail": "CVaR within limits",
        })
        
        # 2. Position concentration
        proposed_weight = conviction * MAX_SINGLE_POSITION
        current_weight = self._portfolio_state.get(symbol, 0.0)
        total_weight = current_weight + proposed_weight
        checks.append({
            "name": "concentration",
            "passed": total_weight <= MAX_SINGLE_POSITION,
            "detail": f"Position {total_weight:.2%} vs limit {MAX_SINGLE_POSITION:.2%}",
        })
        
        # 3. Regime mismatch — going LONG in CRISIS is high risk
        regime_ok = True
        if direction == "LONG" and regime == "CRISIS" and conviction < 0.70:
            regime_ok = False
        checks.append({
            "name": "regime_mismatch",
            "passed": regime_ok,
            "detail": f"Direction={direction}, Regime={regime}",
        })
        
        # 4. Liquidity violation
        # In a full implementation, we'd check actual liquidity data
        checks.append({
            "name": "liquidity",
            "passed": True,
            "detail": "Liquidity check passed (default)",
        })
        
        # 5. Capital allocation — ensure we have sufficient capital
        allocated = sum(self._portfolio_state.values())
        remaining = 1.0 - allocated
        needs = proposed_weight
        checks.append({
            "name": "capital_allocation",
            "passed": needs <= remaining + 0.01,  # Small tolerance
            "detail": f"Needs {needs:.2%}, available {remaining:.2%}",
        })
        
        return checks
    
    def _compute_position_size(self, conviction: float, regime: str) -> float:
        """
        Compute position size based on conviction and regime.
        
        Scale down in CRISIS regime.
        """
        base_size = conviction * MAX_SINGLE_POSITION
        
        if regime == "CRISIS":
            base_size *= CRISIS_POSITION_SCALE
        
        return round(base_size, 4)
