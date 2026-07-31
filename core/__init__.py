"""
Core Module - Bloomberg Terminal
================================
Package initialization and exports.
"""

from pathlib import Path

# Module paths
CORE_DIR = Path(__file__).parent
PROJECT_DIR = CORE_DIR.parent
DATA_DIR = PROJECT_DIR / "data_warehouse"
FEATURE_DIR = PROJECT_DIR / "feature_store"
SIGNALS_DIR = PROJECT_DIR / "signals"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
FEATURE_DIR.mkdir(exist_ok=True)
SIGNALS_DIR.mkdir(exist_ok=True)

# Lazy imports to avoid circular dependencies
def get_data_pipeline():
    from .data_ingest import DataIngestPipeline
    return DataIngestPipeline()

def get_feature_store():
    from .feature_store import FeatureStore
    return FeatureStore()

def get_signal_generator():
    from .signal_generator import AlphaTableGenerator
    return AlphaTableGenerator()

def get_portfolio_optimizer(returns):
    from .risk_management import PortfolioOptimizer
    return PortfolioOptimizer(returns)

def get_event_bus():
    from .event_bus import EventBus
    return EventBus()

__all__ = [
    'get_data_pipeline',
    'get_feature_store', 
    'get_signal_generator',
    'get_portfolio_optimizer',
    'get_event_bus',
    'DATA_DIR',
    'FEATURE_DIR',
    'SIGNALS_DIR'
]
