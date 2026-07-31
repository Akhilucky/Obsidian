"""
Lifecycle Agent
================
Manage strategy maturity like software releases.

Frequency: On demand / after stage criteria met
Consumes: RISK_APPROVED, DRIFT_DETECTED, ALERT_GENERATED
Produces: STAGE_TRANSITION

Stages:
  RESEARCH → VALIDATED → SHADOW → ACTIVE → RETIRED

No strategy may skip stages.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from agents.base_agent import BaseAgent
from core.event_bus import Event, EventType

logger = logging.getLogger(__name__)


class StrategyStage(Enum):
    """Strategy lifecycle stages — order matters, no skipping."""
    RESEARCH = "RESEARCH"
    VALIDATED = "VALIDATED"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


# Ordered stages for transition validation
STAGE_ORDER = [
    StrategyStage.RESEARCH,
    StrategyStage.VALIDATED,
    StrategyStage.SHADOW,
    StrategyStage.ACTIVE,
    StrategyStage.RETIRED,
]

# Criteria for stage transitions
TRANSITION_CRITERIA = {
    "RESEARCH_to_VALIDATED": {
        "min_backtest_sharpe": 0.5,
        "min_backtest_days": 252,
        "max_drawdown": -0.25,
    },
    "VALIDATED_to_SHADOW": {
        "min_walk_forward_accuracy": 0.52,
        "risk_approval_required": True,
    },
    "SHADOW_to_ACTIVE": {
        "min_shadow_days": 30,
        "live_vs_backtest_correlation": 0.70,
        "no_critical_alerts": True,
    },
    "ACTIVE_to_RETIRED": {
        "model_decay_detected": True,
        "or_manual_override": True,
    },
}


class LifecycleAgent(BaseAgent):
    """
    Agent 10: Manage strategy maturity like software releases.
    
    Responsibilities:
    - Track strategy stage
    - Enforce stage transitions (no skipping)
    - Evaluate promotion criteria
    - Handle retirement
    """
    
    def __init__(self):
        super().__init__(
            name="LifecycleAgent",
            subscriptions=[
                EventType.RISK_APPROVED.value,
                EventType.DRIFT_DETECTED.value,
                EventType.ALERT_GENERATED.value,
            ]
        )
        self._strategies: Dict[str, Dict] = {}  # strategy_id -> state
        self._transition_log: List[Dict] = []
    
    def initialize(self):
        """Load existing strategy states."""
        self._log("Lifecycle manager initialized")
    
    def consume(self, event: Event):
        """React to events that may trigger transitions."""
        payload = event.get_payload()
        
        if event.event_type == EventType.RISK_APPROVED.value:
            # A risk-approved strategy may be eligible for promotion
            symbol = payload.get("symbol", "")
            self._log(f"Risk approval received for {symbol}")
        
        elif event.event_type == EventType.DRIFT_DETECTED.value:
            # Drift may trigger retirement
            drift_type = payload.get("drift_type", "")
            symbol = payload.get("symbol", "")
            self._log(f"Drift detected ({drift_type}) for {symbol} — evaluating retirement")
            self._evaluate_retirement(symbol, drift_type)
        
        elif event.event_type == EventType.ALERT_GENERATED.value:
            severity = payload.get("severity", "")
            if severity == "CRITICAL":
                self._log("Critical alert received — checking active strategies", level="warning")
    
    def produce(self) -> Optional[Event]:
        return None
    
    def health_check(self) -> Dict[str, Any]:
        stage_counts = {}
        for s in self._strategies.values():
            stage = s.get("stage", "UNKNOWN")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        return {
            "status": "healthy",
            "strategies_tracked": len(self._strategies),
            "stage_counts": stage_counts,
            "transitions_logged": len(self._transition_log),
        }
    
    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    
    def register_strategy(self, strategy_id: str, name: str,
                          metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Register a new strategy in RESEARCH stage."""
        if strategy_id in self._strategies:
            return {"error": f"Strategy {strategy_id} already registered"}
        
        state = {
            "strategy_id": strategy_id,
            "name": name,
            "stage": StrategyStage.RESEARCH.value,
            "registered_at": datetime.now(datetime.timezone.utc).isoformat(),
            "stage_entered_at": datetime.now(datetime.timezone.utc).isoformat(),
            "metadata": metadata or {},
            "history": [{
                "stage": StrategyStage.RESEARCH.value,
                "entered_at": datetime.now(datetime.timezone.utc).isoformat(),
                "reason": "Initial registration",
            }],
        }
        self._strategies[strategy_id] = state
        self._log(f"Strategy registered: {strategy_id} ({name}) → RESEARCH")
        return state
    
    def promote(self, strategy_id: str, evidence: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Promote a strategy to the next stage.
        
        No stage may be skipped.
        """
        if strategy_id not in self._strategies:
            return {"error": f"Strategy {strategy_id} not found"}
        
        state = self._strategies[strategy_id]
        current_stage = StrategyStage(state["stage"])
        
        # Find next stage
        current_idx = STAGE_ORDER.index(current_stage)
        if current_idx >= len(STAGE_ORDER) - 1:
            return {"error": f"Strategy {strategy_id} is already at final stage ({current_stage.value})"}
        
        next_stage = STAGE_ORDER[current_idx + 1]
        
        # Validate transition criteria
        criteria_key = f"{current_stage.value}_to_{next_stage.value}"
        criteria = TRANSITION_CRITERIA.get(criteria_key, {})
        
        validation = self._validate_criteria(criteria, evidence or {})
        
        if not validation["passed"]:
            return {
                "error": "Promotion criteria not met",
                "criteria": criteria,
                "validation": validation,
                "current_stage": current_stage.value,
                "target_stage": next_stage.value,
            }
        
        # Execute transition
        old_stage = current_stage.value
        state["stage"] = next_stage.value
        state["stage_entered_at"] = datetime.now(datetime.timezone.utc).isoformat()
        state["history"].append({
            "stage": next_stage.value,
            "entered_at": datetime.now(datetime.timezone.utc).isoformat(),
            "reason": f"Promoted from {old_stage}",
            "evidence": evidence,
        })
        
        transition = {
            "strategy_id": strategy_id,
            "from_stage": old_stage,
            "to_stage": next_stage.value,
            "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
            "evidence": evidence,
        }
        self._transition_log.append(transition)
        
        self._log(f"Strategy {strategy_id}: {old_stage} → {next_stage.value}")
        
        # Publish STAGE_TRANSITION
        self._publish(
            EventType.STAGE_TRANSITION.value,
            transition
        )
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "new_stage": next_stage.value,
            "transition": transition,
        }
    
    def retire(self, strategy_id: str, reason: str = "Manual retirement") -> Dict[str, Any]:
        """Force-retire a strategy (moves to RETIRED regardless of current stage)."""
        if strategy_id not in self._strategies:
            return {"error": f"Strategy {strategy_id} not found"}
        
        state = self._strategies[strategy_id]
        old_stage = state["stage"]
        
        state["stage"] = StrategyStage.RETIRED.value
        state["stage_entered_at"] = datetime.now(datetime.timezone.utc).isoformat()
        state["history"].append({
            "stage": StrategyStage.RETIRED.value,
            "entered_at": datetime.now(datetime.timezone.utc).isoformat(),
            "reason": reason,
        })
        
        transition = {
            "strategy_id": strategy_id,
            "from_stage": old_stage,
            "to_stage": StrategyStage.RETIRED.value,
            "reason": reason,
            "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._transition_log.append(transition)
        
        self._log(f"Strategy {strategy_id} RETIRED from {old_stage}: {reason}")
        
        self._publish(
            EventType.STAGE_TRANSITION.value,
            transition
        )
        
        return {"success": True, **transition}
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Get current state of a strategy."""
        return self._strategies.get(strategy_id)
    
    def get_strategies_by_stage(self, stage: str) -> List[Dict]:
        """Get all strategies at a given stage."""
        return [s for s in self._strategies.values() if s["stage"] == stage]
    
    def get_all_strategies(self) -> Dict[str, Dict]:
        """Get all tracked strategies."""
        return dict(self._strategies)
    
    def get_transition_history(self, strategy_id: Optional[str] = None) -> List[Dict]:
        """Get transition history, optionally for a specific strategy."""
        if strategy_id:
            return [t for t in self._transition_log if t["strategy_id"] == strategy_id]
        return list(self._transition_log)
    
    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────
    
    def _validate_criteria(self, criteria: Dict, evidence: Dict) -> Dict[str, Any]:
        """Validate transition criteria against provided evidence."""
        if not criteria:
            return {"passed": True, "checks": []}
        
        checks = []
        all_passed = True
        
        for key, required in criteria.items():
            if key.startswith("or_"):
                continue  # 'or' conditions are alternatives
            
            provided = evidence.get(key)
            
            if isinstance(required, bool):
                passed = bool(provided) == required
            elif isinstance(required, (int, float)):
                if provided is None:
                    passed = False
                elif key.startswith("max_"):
                    passed = provided <= required
                elif key.startswith("min_"):
                    passed = provided >= required
                else:
                    passed = provided >= required
            else:
                passed = provided is not None
            
            checks.append({
                "criterion": key,
                "required": required,
                "provided": provided,
                "passed": passed,
            })
            
            if not passed:
                all_passed = False
        
        return {"passed": all_passed, "checks": checks}
    
    def _evaluate_retirement(self, symbol: str, drift_type: str):
        """Check if any ACTIVE strategy for this symbol should be retired."""
        for sid, state in self._strategies.items():
            if state["stage"] == StrategyStage.ACTIVE.value:
                meta = state.get("metadata", {})
                if meta.get("symbol") == symbol or meta.get("symbols", []):
                    if drift_type in ["MODEL_DECAY", "PERFORMANCE_DIVERGENCE"]:
                        self._log(
                            f"Evaluating retirement for {sid} due to {drift_type}",
                            level="warning"
                        )
                        # In production, this would check more criteria
                        # For now, flag for review
                        self._publish(
                            EventType.ALERT_GENERATED.value,
                            {
                                "severity": "WARNING",
                                "message": f"Strategy {sid} flagged for review: {drift_type}",
                            }
                        )
