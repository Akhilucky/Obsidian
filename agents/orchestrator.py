"""
Agent Orchestrator - Continuous Pipeline Runner
=================================================
Runs the full agent pipeline as a continuously reasoning system:
  Data → Quality → Feature → Regime → Model → Decision → Risk → Execution Sim → Monitoring

Usage:
    from agents.orchestrator import AgentOrchestrator
    orch = AgentOrchestrator()
    orch.run_pipeline(["AAPL", "MSFT", "GOOGL"])

    # Or continuous mode:
    orch.run_continuous(symbols=["AAPL"], interval_seconds=300)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from agents.agent_registry import AgentRegistry, create_default_registry
from core.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a single pipeline run for a symbol."""
    symbol: str
    timestamp: str
    success: bool
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "success": self.success,
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
            "data": {k: str(v)[:200] for k, v in self.data.items()},
            "duration_ms": round(self.duration_ms, 2),
        }


class AgentOrchestrator:
    """
    Orchestrates the full agent pipeline end-to-end.
    
    This is the primary entry point for running the cooperative
    multi-agent system as a continuously reasoning financial engine.
    """
    
    def __init__(self, registry: Optional[AgentRegistry] = None):
        self._registry = registry or create_default_registry()
        self._bus = EventBus()
        self._results: List[PipelineResult] = []
        self._running = False
        self._started = False
    
    @property
    def registry(self) -> AgentRegistry:
        """Public access to the agent registry."""
        return self._registry
        
    def start(self):
        """Initialize all agents in the registry."""
        if not self._started:
            self._registry.start_all()
            self._started = True
            logger.info("Orchestrator ready — all agents initialized")
    
    def run_pipeline(self, symbols: List[str], source: str = "yahoo",
                     period: str = "1y", horizon: str = "5D") -> Dict[str, PipelineResult]:
        """
        Run the full pipeline for a list of symbols.
        
        Pipeline:
            1. DataIngestionAgent  → ingest raw data
            2. DataQualityAgent    → validate + confidence score
            3. FeatureEngAgent     → compute feature matrix
            4. RegimeDetectionAgent → identify market regime
            5. ModelingAgent       → generate signal
            6. DecisionAgent       → create trade idea
            7. RiskAgent           → approve/reject
            8. ScenarioAgent       → stress test
            9. MonitoringAgent     → check drift
            10. LifecycleAgent     → manage strategy stage
        
        Returns:
            Dict[symbol -> PipelineResult]
        """
        self.start()
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        max_workers = min(len(symbols), 8)  # Cap at 8 parallel workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._run_single, symbol, source, period, horizon): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=60)
                except Exception as e:
                    logger.error(f"Pipeline failed for {symbol}: {e}")
                    result = PipelineResult(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        success=False,
                        stages_failed=[f"ThreadPoolError: {str(e)}"],
                    )
                results[symbol] = result
                self._results.append(result)
        
        return results
    
    def run_continuous(self, symbols: List[str], interval_seconds: int = 300,
                       max_iterations: Optional[int] = None, **kwargs):
        """
        Run the pipeline in a continuous loop.
        
        Args:
            symbols: Universe of symbols
            interval_seconds: Sleep between runs
            max_iterations: Stop after N iterations (None = infinite)
        """
        self.start()
        self._running = True
        iteration = 0
        
        logger.info(f"Starting continuous pipeline: {symbols} every {interval_seconds}s")
        
        try:
            while self._running:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Pipeline iteration #{iteration} — {datetime.now(timezone.utc).isoformat()}")
                logger.info(f"{'='*60}")
                
                self.run_pipeline(symbols, **kwargs)
                
                # Print health summary
                health = self._registry.health_check_all()
                logger.info(f"System status: {health['system_status']}")
                
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break
                
                logger.info(f"Sleeping {interval_seconds}s until next run...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user (Ctrl+C)")
        finally:
            self._running = False
    
    def stop(self):
        """Stop the continuous pipeline."""
        self._running = False
        logger.info("Orchestrator stop requested")
    
    # ──────────────────────────────────────────────
    # Single symbol pipeline
    # ──────────────────────────────────────────────
    
    def _run_single(self, symbol: str, source: str, period: str,
                    horizon: str) -> PipelineResult:
        """Execute the full pipeline for a single symbol."""
        start_ts = time.time()
        result = PipelineResult(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=False,
        )
        
        try:
            # ── Stage 1: Data Ingestion ──
            data_agent = self._registry.get("DataIngestionAgent")
            data = data_agent.ingest([symbol], source=source, period=period)
            
            if symbol not in data or data[symbol].empty:
                result.stages_failed.append("DataIngestion")
                return self._finalize(result, start_ts)
            
            df = data[symbol]
            result.stages_completed.append("DataIngestion")
            result.data["rows_ingested"] = len(df)
            
            # ── Stage 2: Data Quality ──
            quality_agent = self._registry.get("DataQualityAgent")
            quality_result = quality_agent.validate(df, symbol)
            confidence = quality_result.get("confidence", 0)
            result.stages_completed.append("DataQuality")
            result.data["quality_confidence"] = confidence
            result.data["anomalies"] = quality_result.get("anomalies", 0)
            
            if confidence < 0.3:
                logger.warning(f"{symbol}: Data quality too low ({confidence}) — aborting")
                result.stages_failed.append("DataQuality_LowConfidence")
                return self._finalize(result, start_ts)
            
            # ── Stage 3: Feature Engineering ──
            feature_agent = self._registry.get("FeatureEngineeringAgent")
            feature_df = feature_agent.compute(df, symbol)
            result.stages_completed.append("FeatureEngineering")
            result.data["features_computed"] = len([c for c in feature_df.columns if c not in df.columns])
            
            # ── Stage 4: Regime Detection ──
            regime_agent = self._registry.get("RegimeDetectionAgent")
            regime_result = regime_agent.detect(df, symbol)
            regime = regime_result.get("regime", "LOW_VOL")
            result.stages_completed.append("RegimeDetection")
            result.data["regime"] = regime
            result.data["hurst_exponent"] = regime_result.get("hurst_exponent")
            
            # ── Stage 5: Feature Scaling (regime-sensitive) ──
            feature_df = feature_agent.compute(df, symbol, regime=regime)
            result.stages_completed.append("FeatureScaling")
            
            # ── Stage 6: Modeling ──
            model_agent = self._registry.get("ModelingAgent")
            prediction = model_agent.predict(feature_df, symbol, regime=regime, horizon=horizon)
            
            if "error" in prediction:
                logger.warning(f"{symbol}: Model error — {prediction['error']}")
                result.stages_failed.append("Modeling")
                result.data["model_error"] = prediction["error"]
                return self._finalize(result, start_ts)
            
            result.stages_completed.append("Modeling")
            result.data["signal"] = prediction.get("signal")
            result.data["confidence"] = prediction.get("confidence")
            result.data["ensemble_prob"] = prediction.get("ensemble_prob")
            
            # ── Stage 7: Decision ──
            decision_agent = self._registry.get("DecisionAgent")
            trade_idea = decision_agent.evaluate(
                symbol=symbol,
                signal=prediction.get("signal", "NEUTRAL"),
                confidence=prediction.get("confidence", 0),
                regime=regime,
            )
            result.stages_completed.append("Decision")
            result.data["conviction"] = trade_idea.get("conviction")
            result.data["direction"] = trade_idea.get("direction")
            result.data["explanation"] = trade_idea.get("explanation")
            
            # ── Stage 8: Risk Evaluation ──
            # Risk agent consumes TRADE_IDEA_CREATED events automatically
            # but we can also check the approval state directly
            risk_agent = self._registry.get("RiskAgent")
            result.stages_completed.append("RiskEvaluation")
            result.data["risk_approved"] = len(risk_agent._approved)
            result.data["risk_rejected"] = len(risk_agent._rejected)
            
            # ── Stage 9: Scenario Stress Test ──
            scenario_agent = self._registry.get("ScenarioAgent")
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                scenario_result = scenario_agent.run_scenarios(returns, symbol)
                result.stages_completed.append("ScenarioStressTest")
                result.data["resilience_score"] = scenario_result.get("resilience_score")
            
            # ── Stage 10: Monitoring ──
            monitor_agent = self._registry.get("MonitoringAgent")
            result.stages_completed.append("Monitoring")
            result.data["active_alerts"] = len(monitor_agent._active_alerts) if hasattr(monitor_agent, '_active_alerts') else 0
            
            # ── Stage 11: Lifecycle ──
            lifecycle_agent = self._registry.get("LifecycleAgent")
            lifecycle_agent.register_strategy(
                f"{symbol}_{horizon}",
                {"symbol": symbol, "horizon": horizon, "regime": regime}
            )
            result.stages_completed.append("Lifecycle")
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Pipeline error for {symbol}: {e}")
            result.stages_failed.append(f"Error: {str(e)}")
        
        return self._finalize(result, start_ts)
    
    def _finalize(self, result: PipelineResult, start_ts: float) -> PipelineResult:
        """Finalize pipeline result with timing."""
        result.duration_ms = (time.time() - start_ts) * 1000
        status = "SUCCESS" if result.success else "PARTIAL"
        logger.info(
            f"[{result.symbol}] {status} in {result.duration_ms:.0f}ms — "
            f"stages: {len(result.stages_completed)} completed, "
            f"{len(result.stages_failed)} failed"
        )
        return result
    
    # ──────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────
    
    @property
    def last_results(self) -> List[Dict]:
        """Get the last pipeline results as dicts."""
        return [r.to_dict() for r in self._results[-50:]]
    
    def summary(self) -> Dict[str, Any]:
        """Get orchestrator summary."""
        health = self._registry.health_check_all() if self._started else {}
        recent = self._results[-20:]
        
        return {
            "started": self._started,
            "running": self._running,
            "total_runs": len(self._results),
            "recent_success_rate": (
                sum(1 for r in recent if r.success) / len(recent) 
                if recent else 0
            ),
            "avg_duration_ms": (
                np.mean([r.duration_ms for r in recent]) 
                if recent else 0
            ),
            "system_health": health.get("system_status", "unknown"),
            "agent_count": health.get("agent_count", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def pipeline_diagram(self) -> str:
        """ASCII diagram of the pipeline with current status."""
        stages = [
            ("Data Ingestion", "DataIngestionAgent"),
            ("Data Quality", "DataQualityAgent"),
            ("Feature Engineering", "FeatureEngineeringAgent"),
            ("Regime Detection", "RegimeDetectionAgent"),
            ("Modeling", "ModelingAgent"),
            ("Decision", "DecisionAgent"),
            ("Risk Evaluation", "RiskAgent"),
            ("Scenario Sim", "ScenarioAgent"),
            ("Monitoring", "MonitoringAgent"),
            ("Lifecycle", "LifecycleAgent"),
        ]
        
        lines = ["Pipeline Status", "=" * 50]
        for label, agent_name in stages:
            agent = self._registry.get(agent_name)
            if agent:
                health = agent.health_check()
                status = health.get("status", "unknown")
                icon = "OK" if status == "healthy" else ("!!" if status == "degraded" else "XX")
            else:
                icon = "--"
            lines.append(f"  [{icon}] {label:<25} ({agent_name})")
            if label != "Lifecycle":
                lines.append(f"         |")
                lines.append(f"         v")
        
        lines.append("=" * 50)
        return "\n".join(lines)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    """Run the pipeline from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Obsidian Quant Agent Pipeline")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL"],
                        help="Symbols to analyze")
    parser.add_argument("--source", default="yahoo", help="Data source")
    parser.add_argument("--period", default="1y", help="Data period")
    parser.add_argument("--horizon", default="5D", help="Prediction horizon")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Continuous interval (seconds)")
    parser.add_argument("--iterations", type=int, default=None, help="Max iterations")
    parser.add_argument("--diagram", action="store_true", help="Print pipeline diagram")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    orch = AgentOrchestrator()
    
    if args.diagram:
        orch.start()
        print(orch.pipeline_diagram())
        return
    
    if args.continuous:
        orch.run_continuous(
            symbols=args.symbols,
            interval_seconds=args.interval,
            max_iterations=args.iterations,
            source=args.source,
            period=args.period,
            horizon=args.horizon,
        )
    else:
        results = orch.run_pipeline(
            symbols=args.symbols,
            source=args.source,
            period=args.period,
            horizon=args.horizon,
        )
        
        print("\n" + orch.pipeline_diagram())
        print("\nResults:")
        print("-" * 50)
        for symbol, result in results.items():
            d = result.to_dict()
            print(f"\n{symbol}:")
            print(f"  Success: {d['success']}")
            print(f"  Duration: {d['duration_ms']:.0f}ms")
            print(f"  Stages: {', '.join(d['stages_completed'])}")
            if d['stages_failed']:
                print(f"  Failed: {', '.join(d['stages_failed'])}")
            for key in ['signal', 'confidence', 'regime', 'conviction', 'direction', 'resilience_score']:
                if key in d['data']:
                    print(f"  {key}: {d['data'][key]}")
        
        print(f"\nSummary: {json.dumps(orch.summary(), indent=2)}")


if __name__ == "__main__":
    main()
