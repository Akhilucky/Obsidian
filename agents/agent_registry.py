"""
Agent Registry
===============
Central registry for all agents in the Aegis Quant Platform.

To add a new agent:
1. Create file in /agents
2. Implement: initialize(), consume(event), produce(), health_check()
3. Register in this file

Usage:
    from agents.agent_registry import AgentRegistry
    registry = AgentRegistry()
    registry.start_all()
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Any, Dict, List, Optional, Type
from datetime import datetime, timezone

from agents.base_agent import BaseAgent
from core.event_bus import EventBus

logger = logging.getLogger(__name__)

# Scheduling frequencies (seconds)
SCHEDULE = {
    "DataIngestionAgent": 0,         # Realtime (event-driven)
    "DataQualityAgent": 0,           # Realtime (event-driven)
    "RegimeDetectionAgent": 300,     # 5 min
    "ModelingAgent": 900,            # 15 min
    "RiskAgent": 3600,               # Hourly
    "MonitoringAgent": 86400,        # Daily
    # Decision, Scenario, Lifecycle are event-driven
}


class AgentRegistry:
    """
    Central registry and lifecycle manager for all agents.
    
    Responsibilities:
    - Register/deregister agents
    - Initialize all agents
    - Health check aggregation
    - Metrics aggregation
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._bus = EventBus()
        self._started = False
    
    def register(self, agent: BaseAgent):
        """Register an agent instance."""
        if agent.name in self._agents:
            logger.warning(f"Agent {agent.name} already registered — replacing")
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def deregister(self, name: str):
        """Remove an agent from the registry."""
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Deregistered agent: {name}")
    
    def get(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(name)
    
    def start_all(self):
        """Initialize all registered agents."""
        logger.info(f"Starting {len(self._agents)} agents...")
        for name, agent in self._agents.items():
            try:
                agent.start()
                logger.info(f"  ✓ {name}")
            except Exception as e:
                logger.error(f"  ✗ {name}: {e}")
        self._started = True
        logger.info("All agents started")
    
    def health_check_all(self) -> Dict[str, Any]:
        """Aggregate health from all agents."""
        results = {}
        for name, agent in self._agents.items():
            try:
                results[name] = agent.health
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        
        all_healthy = all(
            r.get("status") == "healthy" for r in results.values()
        )
        
        return {
            "system_status": "healthy" if all_healthy else "degraded",
            "agents": results,
            "agent_count": len(self._agents),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def metrics_all(self) -> Dict[str, Any]:
        """Aggregate metrics from all agents."""
        results = {}
        for name, agent in self._agents.items():
            results[name] = agent.metrics
        return {
            "agents": results,
            "bus_stats": self._bus.stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def logs_all(self, last_n: int = 50) -> Dict[str, List[str]]:
        """Collect recent logs from all agents."""
        results = {}
        for name, agent in self._agents.items():
            results[name] = agent.logs[-last_n:]
        return results
    
    @property
    def agents(self) -> Dict[str, BaseAgent]:
        return dict(self._agents)
    
    @property
    def agent_names(self) -> List[str]:
        return list(self._agents.keys())


def create_default_registry() -> AgentRegistry:
    """
    Create a registry with all default agents pre-registered.
    
    This is the standard way to bootstrap the multi-agent system.
    
    Usage:
        registry = create_default_registry()
        registry.start_all()
    """
    from agents.data_agent import DataIngestionAgent
    from agents.quality_agent import DataQualityAgent
    from agents.feature_agent import FeatureEngineeringAgent
    from agents.regime_agent import RegimeDetectionAgent
    from agents.model_agent import ModelingAgent
    from agents.decision_agent import DecisionAgent
    from agents.risk_agent import RiskAgent
    from agents.scenario_agent import ScenarioAgent
    from agents.monitor_agent import MonitoringAgent
    from agents.lifecycle_agent import LifecycleAgent
    
    registry = AgentRegistry()
    
    # Register in pipeline order
    registry.register(DataIngestionAgent())
    registry.register(DataQualityAgent())
    registry.register(FeatureEngineeringAgent())
    registry.register(RegimeDetectionAgent())
    registry.register(ModelingAgent())
    registry.register(DecisionAgent())
    registry.register(RiskAgent())
    registry.register(ScenarioAgent())
    registry.register(MonitoringAgent())
    registry.register(LifecycleAgent())
    
    return registry
