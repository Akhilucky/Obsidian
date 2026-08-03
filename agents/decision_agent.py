"""
Decision Agent
===============
Convert predictions into structured trade ideas.

Frequency: After MODEL_SIGNAL
Consumes: MODEL_SIGNAL
Produces: TRADE_IDEA_CREATED

This is where analytics becomes actionable intelligence.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import Event, EventType

logger = logging.getLogger(__name__)

# Minimum confidence to generate a trade idea
MIN_CONVICTION = 0.30
MAX_IDEAS_PER_CYCLE = 20


@dataclass
class TradeIdea:
    """Structured trade idea."""
    idea_id: str
    symbol: str
    direction: str          # LONG, SHORT
    conviction: float       # 0-1 scale
    confidence: float       # Model confidence
    regime: str
    horizon: str
    explanation: str
    risk_context: Dict[str, Any] = field(default_factory=dict)
    ranking: int = 0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "conviction": self.conviction,
            "confidence": self.confidence,
            "regime": self.regime,
            "horizon": self.horizon,
            "explanation": self.explanation,
            "risk_context": self.risk_context,
            "ranking": self.ranking,
            "timestamp": self.timestamp,
        }


class DecisionAgent(BaseAgent):
    """
    Agent 6: Convert predictions into structured trade ideas.
    
    Responsibilities:
    - Combine model output + risk context
    - Rank opportunities
    - Attach explanation
    - Produce conviction score
    """
    
    def __init__(self):
        super().__init__(
            name="DecisionAgent",
            subscriptions=[EventType.MODEL_SIGNAL.value]
        )
        self._pending_signals: List[Dict] = []
        self._trade_ideas: List[TradeIdea] = []
        self._risk_context: Dict[str, Dict] = {}
    
    def initialize(self):
        """Load decision rules and ranking weights."""
        self._log("Decision rules loaded")
    
    def consume(self, event: Event):
        """
        Collect model signals and generate trade ideas.
        """
        payload = event.get_payload()
        symbol = payload.get("symbol", "UNKNOWN")
        signal = payload.get("signal", "NEUTRAL")
        confidence = payload.get("confidence", 0)
        horizon = payload.get("horizon", "5D")
        regime = payload.get("regime", "UNKNOWN")
        
        if signal == "NEUTRAL":
            self._log(f"Skipping {symbol} — NEUTRAL signal")
            return
        
        if confidence < MIN_CONVICTION:
            self._log(f"Skipping {symbol} — low confidence {confidence}")
            return
        
        # Build trade idea
        idea = self._build_trade_idea(symbol, signal, confidence, regime, horizon)
        self._trade_ideas.append(idea)
        self._log(f"Trade idea created: {symbol} {signal} (conviction={idea.conviction})")
        
        # Publish TRADE_IDEA_CREATED
        self._publish(
            EventType.TRADE_IDEA_CREATED.value,
            idea.to_dict()
        )
    
    def produce(self) -> Optional[Event]:
        return None
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "ideas_generated": len(self._trade_ideas),
            "pending_signals": len(self._pending_signals),
        }
    
    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    
    def evaluate(self, symbol: str, signal: str, confidence: float,
                 regime: str, horizon: str = "5D") -> Dict[str, Any]:
        """
        Evaluate a model signal and generate a trade idea.
        
        Called by the orchestrator for direct pipeline invocation.
        Also publishes TRADE_IDEA_CREATED event.
        
        Returns:
            Dict with direction, conviction, explanation, etc.
        """
        if signal == "NEUTRAL":
            return {"direction": "NEUTRAL", "conviction": 0.0, "explanation": "Neutral signal"}
        
        if confidence < MIN_CONVICTION:
            return {"direction": signal, "conviction": confidence, "explanation": "Below minimum conviction"}
        
        idea = self._build_trade_idea(symbol, signal, confidence, regime, horizon)
        self._trade_ideas.append(idea)
        self._log(f"Trade idea created: {symbol} {signal} (conviction={idea.conviction})")
        
        # Publish TRADE_IDEA_CREATED
        self._publish(
            EventType.TRADE_IDEA_CREATED.value,
            idea.to_dict()
        )
        
        return idea.to_dict()
    
    def set_risk_context(self, symbol: str, context: Dict[str, Any]):
        """Provide risk context for decision-making (from Risk Agent events)."""
        self._risk_context[symbol] = context
    
    def rank_ideas(self) -> List[TradeIdea]:
        """Rank current trade ideas by conviction score."""
        ranked = sorted(self._trade_ideas, key=lambda x: x.conviction, reverse=True)
        for i, idea in enumerate(ranked):
            idea.ranking = i + 1
        return ranked[:MAX_IDEAS_PER_CYCLE]
    
    def get_top_ideas(self, n: int = 5) -> List[Dict]:
        """Get top-N trade ideas."""
        ranked = self.rank_ideas()
        return [idea.to_dict() for idea in ranked[:n]]
    
    def clear_ideas(self):
        """Reset trade ideas for next cycle."""
        self._trade_ideas.clear()
        self._pending_signals.clear()
    
    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────
    
    def _build_trade_idea(self, symbol: str, signal: str,
                          confidence: float, regime: str,
                          horizon: str) -> TradeIdea:
        """Build a structured trade idea with conviction and explanation."""
        import uuid
        
        # Conviction = f(confidence, regime alignment, risk context)
        conviction = self._compute_conviction(signal, confidence, regime, symbol)
        
        # Generate explanation
        explanation = self._generate_explanation(symbol, signal, confidence, regime, conviction)
        
        # Get risk context if available
        risk_ctx = self._risk_context.get(symbol, {})
        
        return TradeIdea(
            idea_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction=signal,
            conviction=round(conviction, 4),
            confidence=round(confidence, 4),
            regime=regime,
            horizon=horizon,
            explanation=explanation,
            risk_context=risk_ctx,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def _compute_conviction(self, signal: str, confidence: float,
                            regime: str, symbol: str) -> float:
        """
        Compute conviction score combining:
        - Model confidence
        - Regime alignment
        - Risk context
        """
        base_conviction = confidence
        
        # Regime alignment bonus
        regime_alignment = {
            ("LONG", "TRENDING"): 0.15,
            ("SHORT", "TRENDING"): 0.05,
            ("LONG", "MEAN_REVERTING"): 0.05,
            ("SHORT", "MEAN_REVERTING"): 0.10,
            ("LONG", "LOW_VOL"): 0.10,
            ("SHORT", "LOW_VOL"): 0.05,
            ("LONG", "CRISIS"): -0.15,
            ("SHORT", "CRISIS"): 0.10,
        }
        alignment_bonus = regime_alignment.get((signal, regime), 0.0)
        
        # Risk context adjustment
        risk_ctx = self._risk_context.get(symbol, {})
        risk_penalty = 0.0
        if risk_ctx.get("tail_risk_breach", False):
            risk_penalty = -0.20
        if risk_ctx.get("liquidity_warning", False):
            risk_penalty -= 0.10
        
        conviction = max(0.0, min(1.0, base_conviction + alignment_bonus + risk_penalty))
        return conviction
    
    def _generate_explanation(self, symbol: str, signal: str,
                              confidence: float, regime: str,
                              conviction: float) -> str:
        """Generate human-readable explanation for the trade idea."""
        regime_desc = {
            "TRENDING": "trending market conditions",
            "MEAN_REVERTING": "mean-reverting conditions",
            "CRISIS": "crisis/high volatility conditions",
            "LOW_VOL": "low volatility environment",
        }
        
        regime_text = regime_desc.get(regime, "current market conditions")
        
        return (
            f"{signal} {symbol} with {conviction:.0%} conviction. "
            f"Model confidence: {confidence:.0%}. "
            f"Regime: {regime_text}. "
            f"{'Risk context applied.' if symbol in self._risk_context else 'No risk overlay.'}"
        )
