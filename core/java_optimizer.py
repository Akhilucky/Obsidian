"""
Java portfolio optimizer integration.

Invokes PortfolioOptimizer (java/PortfolioOptimizer.java, pure-Java
mean-variance optimization) as a subprocess. Falls back to a numpy
implementation if Java is unavailable.

Build: javac -d java/build java/PortfolioOptimizer.java
"""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_JAVA_DIR = Path(__file__).parent.parent / "java"
_CLASSES_DIR = _JAVA_DIR / "build"

JAVA_AVAILABLE = shutil.which("java") is not None and (_CLASSES_DIR / "PortfolioOptimizer.class").exists()


def java_available() -> bool:
    """Whether the Java optimizer is usable."""
    return JAVA_AVAILABLE


def optimize(
    tickers: List[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    timeout: float = 30.0,
) -> Optional[Dict]:
    """
    Run the Java mean-variance optimizer.

    Args:
        tickers: Asset identifiers
        expected_returns: Array of mean returns
        cov_matrix: Symmetric covariance matrix

    Returns:
        {"min_variance": {ticker: weight}, "max_sharpe": {ticker: weight},
         "sharpe": float} or None if Java is unavailable / fails.
    """
    if not JAVA_AVAILABLE:
        return None

    n = len(tickers)
    if len(expected_returns) != n or cov_matrix.shape != (n, n):
        return None

    lines = [str(n), ",".join(tickers),
             ",".join(f"{x:.10f}" for x in expected_returns)]
    for row in cov_matrix:
        lines.append(",".join(f"{x:.10f}" for x in row))
    payload = "\n".join(lines) + "\n"

    try:
        result = subprocess.run(
            ["java", "-cp", str(_CLASSES_DIR), "PortfolioOptimizer"],
            input=payload, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def optimize_numpy(
    tickers: List[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> Dict:
    """
    Pure-numpy fallback with identical closed-form math to the Java optimizer.
    """
    inv = np.linalg.inv(cov_matrix)
    n = len(tickers)

    w_min = inv @ np.ones(n)
    w_min = w_min / w_min.sum()

    denom = expected_returns @ inv @ expected_returns
    w_sharpe = None
    sharpe = 0.0
    if abs(denom) > 1e-12:
        w_sharpe = inv @ expected_returns
        w_sharpe = w_sharpe / w_sharpe.sum()
        ret = w_sharpe @ expected_returns
        var = w_sharpe @ cov_matrix @ w_sharpe
        sharpe = ret / np.sqrt(max(var, 1e-12))

    return {
        "min_variance": {t: float(w) for t, w in zip(tickers, w_min)},
        "max_sharpe": {t: float(w) for t, w in zip(tickers, w_sharpe)}
        if w_sharpe is not None else {},
        "sharpe": float(sharpe),
    }


def optimize_portfolio(
    tickers: List[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    prefer_java: bool = True,
) -> Dict:
    """
    Portfolio optimization with Java-first execution and numpy fallback.
    """
    if prefer_java:
        result = optimize(tickers, expected_returns, cov_matrix)
        if result is not None:
            result["engine"] = "java"
            return result
    result = optimize_numpy(tickers, expected_returns, cov_matrix)
    result["engine"] = "numpy"
    return result
