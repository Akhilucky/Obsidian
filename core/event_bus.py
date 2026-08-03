"""
Event Bus - Inter-Agent Communication Layer
=============================================
Immutable JSON event messages, idempotent consumption, no shared state.

Rules:
- JSON only
- Immutable messages
- No shared state
- Idempotent consumption required
"""

import json
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from copy import deepcopy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """All event types in the agent pipeline."""
    # Data pipeline
    DATA_INGESTED = "DATA_INGESTED"
    DATA_VALIDATED = "DATA_VALIDATED"
    
    # Feature pipeline
    FEATURE_MATRIX_READY = "FEATURE_MATRIX_READY"
    
    # Regime
    REGIME_DETECTED = "REGIME_DETECTED"
    
    # Modeling
    MODEL_SIGNAL = "MODEL_SIGNAL"
    
    # Decision
    TRADE_IDEA_CREATED = "TRADE_IDEA_CREATED"
    
    # Risk
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"
    
    # Scenario
    SCENARIO_RESULT = "SCENARIO_RESULT"
    
    # Monitoring
    DRIFT_DETECTED = "DRIFT_DETECTED"
    ALERT_GENERATED = "ALERT_GENERATED"
    
    # Lifecycle
    STAGE_TRANSITION = "STAGE_TRANSITION"
    
    # System
    AGENT_HEALTH = "AGENT_HEALTH"
    AGENT_ERROR = "AGENT_ERROR"


@dataclass(frozen=True)
class Event:
    """
    Immutable event message passed between agents.
    
    Once created, an Event cannot be modified.
    All fields are frozen (dataclass frozen=True).
    """
    event_id: str
    event_type: str
    source_agent: str
    timestamp: str
    payload: tuple  # frozen dict not possible; use tuple of sorted items
    
    @staticmethod
    def create(event_type: str, source_agent: str, payload: Dict[str, Any]) -> 'Event':
        """Factory method to create a new immutable event."""
        return Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source_agent=source_agent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=tuple(sorted(payload.items())) if isinstance(payload, dict) else ()
        )
    
    def get_payload(self) -> Dict[str, Any]:
        """Reconstruct payload dict from frozen tuple."""
        return dict(self.payload)
    
    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_agent": self.source_agent,
            "timestamp": self.timestamp,
            "payload": self.get_payload()
        }, default=str, indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> 'Event':
        """Deserialize event from JSON string."""
        data = json.loads(json_str)
        return Event(
            event_id=data["event_id"],
            event_type=data["event_type"],
            source_agent=data["source_agent"],
            timestamp=data["timestamp"],
            payload=tuple(sorted(data.get("payload", {}).items()))
        )


class EventBus:
    """
    Central event bus for agent-to-agent communication.
    
    - Agents subscribe to event types
    - Events are dispatched to all subscribers
    - Thread-safe
    - Tracks processed event IDs for idempotency
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern — one bus per process."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_log: List[Event] = []
        self._processed_ids: set = set()
        self._bus_lock = threading.Lock()
        self._initialized = True
        logger.info("EventBus initialized (singleton)")
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe a callback to an event type."""
        with self._bus_lock:
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type}: {callback}")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe a callback from an event type."""
        with self._bus_lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
    
    def publish(self, event: Event):
        """
        Publish an event to all subscribers.
        
        Idempotency: if the event_id was already processed, skip.
        """
        with self._bus_lock:
            if event.event_id in self._processed_ids:
                logger.warning(f"Duplicate event {event.event_id} — skipped")
                return
            self._processed_ids.add(event.event_id)
            self._event_log.append(event)
        
        subscribers = self._subscribers.get(event.event_type, [])
        for callback in subscribers:
            try:
                # Pass a deep copy so consumers can't mutate state
                callback(deepcopy(event))
            except Exception as e:
                logger.error(f"Error in subscriber {callback} for {event.event_type}: {e}")
    
    def get_event_log(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Retrieve recent events, optionally filtered by type."""
        with self._bus_lock:
            if event_type:
                filtered = [e for e in self._event_log if e.event_type == event_type]
            else:
                filtered = list(self._event_log)
            return filtered[-limit:]
    
    def clear(self):
        """Reset the bus (useful for testing)."""
        with self._bus_lock:
            self._subscribers.clear()
            self._event_log.clear()
            self._processed_ids.clear()
            logger.info("EventBus cleared")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Bus statistics."""
        with self._bus_lock:
            type_counts = defaultdict(int)
            for e in self._event_log:
                type_counts[e.event_type] += 1
            return {
                "total_events": len(self._event_log),
                "unique_processed": len(self._processed_ids),
                "subscriber_count": sum(len(v) for v in self._subscribers.values()),
                "events_by_type": dict(type_counts)
            }
