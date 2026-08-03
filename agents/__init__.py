"""
Agents Module - Obsidian Quant Platform
======================================
Cooperative multi-agent pipeline for:
  Data → Quality → Feature → Regime → Model → Decision → Risk → Execution Sim → Monitoring

Each agent:
- Implements: initialize(), consume(event), produce(), health_check()
- Exposes: /metrics, /logs, /health
- Communicates via EventBus (JSON, immutable, idempotent)

Usage:
    from agents.agent_registry import create_default_registry
    registry = create_default_registry()
    registry.start_all()
"""

from agents.base_agent import BaseAgent
from agents.agent_registry import AgentRegistry, create_default_registry
from agents.orchestrator import AgentOrchestrator

__all__ = [
    'BaseAgent',
    'AgentRegistry',
    'create_default_registry',
    'AgentOrchestrator',
]
