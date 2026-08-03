"""
Base Agent - Abstract base for all Obsidian Quant Platform agents
================================================================
Every agent must implement:
  - initialize()
  - consume(event)
  - produce()
  - health_check()

Every agent exposes:
  - /metrics
  - /logs
  - /health
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.event_bus import EventBus, Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Observable metrics every agent must expose."""
    events_consumed: int = 0
    events_produced: int = 0
    errors: int = 0
    last_run: Optional[str] = None
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    confidence_scores: List[float] = field(default_factory=list)
    drift_stats: Dict[str, float] = field(default_factory=dict)
    data_reliability: float = 1.0
    custom: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Anti-pattern enforcement:
    - No agent may call another agent directly (communicate via EventBus).
    - No hidden database writes (all output goes through produce()).
    - No synchronous blocking workflows.
    - No UI logic inside agents.
    """
    
    def __init__(self, name: str, subscriptions: Optional[List[str]] = None):
        self.name = name
        self.bus = EventBus()
        self._metrics = AgentMetrics()
        self._logs: deque = deque(maxlen=1000)
        self._is_initialized = False
        self._subscriptions = subscriptions or []
        
        # Auto-subscribe to event types
        for event_type in self._subscriptions:
            self.bus.subscribe(event_type, self._safe_consume)
    
    # ──────────────────────────────────────────────
    # Required interface (agents.md spec)
    # ──────────────────────────────────────────────
    
    @abstractmethod
    def initialize(self):
        """Set up agent state, load models, warm caches."""
        ...
    
    @abstractmethod
    def consume(self, event: Event):
        """Process an incoming event. Must be idempotent."""
        ...
    
    @abstractmethod
    def produce(self) -> Optional[Event]:
        """Generate output event(s). Called after consume or on schedule."""
        ...
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return agent health status."""
        ...
    
    # ──────────────────────────────────────────────
    # Observability endpoints (/metrics, /logs, /health)
    # ──────────────────────────────────────────────
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Expose /metrics."""
        return {
            "agent": self.name,
            "events_consumed": self._metrics.events_consumed,
            "events_produced": self._metrics.events_produced,
            "errors": self._metrics.errors,
            "last_run": self._metrics.last_run,
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 2),
            "confidence_scores": self._metrics.confidence_scores[-10:],
            "drift_stats": self._metrics.drift_stats,
            "data_reliability": self._metrics.data_reliability,
            "custom": self._metrics.custom,
        }
    
    @property
    def logs(self) -> List[str]:
        """Expose /logs."""
        return list(self._logs)
    
    @property
    def health(self) -> Dict[str, Any]:
        """Expose /health."""
        check = self.health_check()
        check["agent"] = self.name
        check["initialized"] = self._is_initialized
        check["timestamp"] = datetime.now(timezone.utc).isoformat()
        return check
    
    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────
    
    def _safe_consume(self, event: Event):
        """Wrapper that tracks metrics and handles errors."""
        start = time.time()
        try:
            self.consume(event)
            self._metrics.events_consumed += 1
        except Exception as e:
            self._metrics.errors += 1
            self._log(f"ERROR consuming {event.event_type}: {e}", level="error")
            # Publish error event
            err_event = Event.create(
                EventType.AGENT_ERROR.value,
                self.name,
                {"error": str(e), "source_event": event.event_type}
            )
            self.bus.publish(err_event)
        finally:
            elapsed_ms = (time.time() - start) * 1000
            total = self._metrics.total_latency_ms + elapsed_ms
            count = self._metrics.events_consumed or 1
            self._metrics.total_latency_ms = total
            self._metrics.avg_latency_ms = total / count
            self._metrics.last_run = datetime.now(timezone.utc).isoformat()
    
    def _publish(self, event_type: str, payload: Dict[str, Any]):
        """Helper to publish an event and track metrics."""
        event = Event.create(event_type, self.name, payload)
        self.bus.publish(event)
        self._metrics.events_produced += 1
        self._log(f"Published {event_type}")
        return event
    
    def _log(self, message: str, level: str = "info"):
        """Internal structured logging."""
        ts = datetime.now(timezone.utc).isoformat()
        entry = f"[{ts}] [{self.name}] [{level.upper()}] {message}"
        self._logs.append(entry)
        getattr(logger, level, logger.info)(entry)
    
    def start(self):
        """Initialize the agent."""
        self._log("Initializing...")
        self.initialize()
        self._is_initialized = True
        self._log("Initialized successfully")
