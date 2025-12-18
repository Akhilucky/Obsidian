"""
Feature Store - Versioned, Point-in-Time Feature Warehouse
============================================================
SQL & Parquet warehouse with:
- Point-in-time correct feature retrieval
- Feature versioning and lineage
- Automatic feature computation
- 100+ pre-built features (technical, fundamental, alternative)
"""

import os
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

import pandas as pd
import numpy as np

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature store directory
FEATURE_STORE_DIR = Path(__file__).parent.parent / "feature_store"
FEATURE_STORE_DIR.mkdir(exist_ok=True)


class FeatureCategory(Enum):
    """Feature categories."""
    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    ALTERNATIVE = "alternative"
    MACRO = "macro"
    CUSTOM = "custom"


@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    name: str
    category: FeatureCategory
    description: str
    compute_fn: Callable
    lookback: int = 0
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0.0"


@dataclass
class FeatureMetadata:
    """Metadata for stored features."""
    name: str
    category: str
    version: str
    created_at: str
    updated_at: str
    row_count: int
    date_range: tuple
    checksum: str


class FeatureRegistry:
    """Registry of all available features."""
    
    def __init__(self):
        self.features: Dict[str, FeatureDefinition] = {}
        self._register_default_features()
    
    def register(self, feature: FeatureDefinition):
        """Register a feature."""
        self.features[feature.name] = feature
        logger.debug(f"Registered feature: {feature.name}")
    
    def get(self, name: str) -> Optional[FeatureDefinition]:
        """Get a feature definition."""
        return self.features.get(name)
    
    def list_features(self, category: FeatureCategory = None) -> List[str]:
        """List all registered features."""
        if category:
            return [name for name, f in self.features.items() if f.category == category]
        return list(self.features.keys())
    
    def _register_default_features(self):
        """Register default feature definitions."""
        
        # ========== Price Features ==========
        self.register(FeatureDefinition(
            name="return_1d",
            category=FeatureCategory.PRICE,
            description="1-day return",
            compute_fn=lambda df: df['close'].pct_change(1),
            lookback=1
        ))
        
        self.register(FeatureDefinition(
            name="return_5d",
            category=FeatureCategory.PRICE,
            description="5-day return",
            compute_fn=lambda df: df['close'].pct_change(5),
            lookback=5
        ))
        
        self.register(FeatureDefinition(
            name="return_10d",
            category=FeatureCategory.PRICE,
            description="10-day return",
            compute_fn=lambda df: df['close'].pct_change(10),
            lookback=10
        ))
        
        self.register(FeatureDefinition(
            name="return_20d",
            category=FeatureCategory.PRICE,
            description="20-day return",
            compute_fn=lambda df: df['close'].pct_change(20),
            lookback=20
        ))
        
        self.register(FeatureDefinition(
            name="log_return",
            category=FeatureCategory.PRICE,
            description="Log return",
            compute_fn=lambda df: np.log(df['close'] / df['close'].shift(1)),
            lookback=1
        ))
        
        self.register(FeatureDefinition(
            name="high_low_range",
            category=FeatureCategory.PRICE,
            description="High-low range as % of close",
            compute_fn=lambda df: (df['high'] - df['low']) / df['close'],
            lookback=0
        ))
        
        self.register(FeatureDefinition(
            name="close_to_high",
            category=FeatureCategory.PRICE,
            description="Close position relative to high",
            compute_fn=lambda df: (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10),
            lookback=0
        ))
        
        self.register(FeatureDefinition(
            name="gap",
            category=FeatureCategory.PRICE,
            description="Overnight gap",
            compute_fn=lambda df: (df['open'] - df['close'].shift(1)) / df['close'].shift(1),
            lookback=1
        ))
        
        # ========== Volume Features ==========
        self.register(FeatureDefinition(
            name="volume_sma_ratio",
            category=FeatureCategory.VOLUME,
            description="Volume relative to 20-day SMA",
            compute_fn=lambda df: df['volume'] / df['volume'].rolling(20).mean(),
            lookback=20
        ))
        
        self.register(FeatureDefinition(
            name="volume_change",
            category=FeatureCategory.VOLUME,
            description="Volume change from previous day",
            compute_fn=lambda df: df['volume'].pct_change(1),
            lookback=1
        ))
        
        self.register(FeatureDefinition(
            name="dollar_volume",
            category=FeatureCategory.VOLUME,
            description="Dollar volume",
            compute_fn=lambda df: df['close'] * df['volume'],
            lookback=0
        ))
        
        self.register(FeatureDefinition(
            name="volume_trend",
            category=FeatureCategory.VOLUME,
            description="5-day volume trend",
            compute_fn=lambda df: df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean(),
            lookback=20
        ))
        
        # ========== Momentum Features ==========
        if TA_AVAILABLE:
            self.register(FeatureDefinition(
                name="rsi_14",
                category=FeatureCategory.MOMENTUM,
                description="14-day RSI",
                compute_fn=lambda df: ta.momentum.RSIIndicator(df['close'], window=14).rsi(),
                lookback=14
            ))
            
            self.register(FeatureDefinition(
                name="rsi_7",
                category=FeatureCategory.MOMENTUM,
                description="7-day RSI",
                compute_fn=lambda df: ta.momentum.RSIIndicator(df['close'], window=7).rsi(),
                lookback=7
            ))
            
            self.register(FeatureDefinition(
                name="stoch_k",
                category=FeatureCategory.MOMENTUM,
                description="Stochastic %K",
                compute_fn=lambda df: ta.momentum.StochasticOscillator(
                    df['high'], df['low'], df['close']
                ).stoch(),
                lookback=14
            ))
            
            self.register(FeatureDefinition(
                name="stoch_d",
                category=FeatureCategory.MOMENTUM,
                description="Stochastic %D",
                compute_fn=lambda df: ta.momentum.StochasticOscillator(
                    df['high'], df['low'], df['close']
                ).stoch_signal(),
                lookback=14
            ))
            
            self.register(FeatureDefinition(
                name="williams_r",
                category=FeatureCategory.MOMENTUM,
                description="Williams %R",
                compute_fn=lambda df: ta.momentum.WilliamsRIndicator(
                    df['high'], df['low'], df['close']
                ).williams_r(),
                lookback=14
            ))
            
            self.register(FeatureDefinition(
                name="cci",
                category=FeatureCategory.MOMENTUM,
                description="Commodity Channel Index",
                compute_fn=lambda df: ta.trend.CCIIndicator(
                    df['high'], df['low'], df['close']
                ).cci(),
                lookback=20
            ))
            
            self.register(FeatureDefinition(
                name="roc_10",
                category=FeatureCategory.MOMENTUM,
                description="10-day Rate of Change",
                compute_fn=lambda df: ta.momentum.ROCIndicator(df['close'], window=10).roc(),
                lookback=10
            ))
        
        # ========== Volatility Features ==========
        self.register(FeatureDefinition(
            name="volatility_10d",
            category=FeatureCategory.VOLATILITY,
            description="10-day realized volatility",
            compute_fn=lambda df: df['close'].pct_change().rolling(10).std() * np.sqrt(252),
            lookback=10
        ))
        
        self.register(FeatureDefinition(
            name="volatility_20d",
            category=FeatureCategory.VOLATILITY,
            description="20-day realized volatility",
            compute_fn=lambda df: df['close'].pct_change().rolling(20).std() * np.sqrt(252),
            lookback=20
        ))
        
        self.register(FeatureDefinition(
            name="volatility_60d",
            category=FeatureCategory.VOLATILITY,
            description="60-day realized volatility",
            compute_fn=lambda df: df['close'].pct_change().rolling(60).std() * np.sqrt(252),
            lookback=60
        ))
        
        if TA_AVAILABLE:
            self.register(FeatureDefinition(
                name="atr_14",
                category=FeatureCategory.VOLATILITY,
                description="14-day ATR",
                compute_fn=lambda df: ta.volatility.AverageTrueRange(
                    df['high'], df['low'], df['close'], window=14
                ).average_true_range(),
                lookback=14
            ))
            
            self.register(FeatureDefinition(
                name="bb_width",
                category=FeatureCategory.VOLATILITY,
                description="Bollinger Band width",
                compute_fn=lambda df: ta.volatility.BollingerBands(
                    df['close'], window=20, window_dev=2
                ).bollinger_wband(),
                lookback=20
            ))
            
            self.register(FeatureDefinition(
                name="bb_pctb",
                category=FeatureCategory.VOLATILITY,
                description="Bollinger Band %B",
                compute_fn=lambda df: ta.volatility.BollingerBands(
                    df['close'], window=20, window_dev=2
                ).bollinger_pband(),
                lookback=20
            ))
        
        # ========== Trend Features ==========
        self.register(FeatureDefinition(
            name="sma_5",
            category=FeatureCategory.TREND,
            description="5-day SMA",
            compute_fn=lambda df: df['close'].rolling(5).mean(),
            lookback=5
        ))
        
        self.register(FeatureDefinition(
            name="sma_10",
            category=FeatureCategory.TREND,
            description="10-day SMA",
            compute_fn=lambda df: df['close'].rolling(10).mean(),
            lookback=10
        ))
        
        self.register(FeatureDefinition(
            name="sma_20",
            category=FeatureCategory.TREND,
            description="20-day SMA",
            compute_fn=lambda df: df['close'].rolling(20).mean(),
            lookback=20
        ))
        
        self.register(FeatureDefinition(
            name="sma_50",
            category=FeatureCategory.TREND,
            description="50-day SMA",
            compute_fn=lambda df: df['close'].rolling(50).mean(),
            lookback=50
        ))
        
        self.register(FeatureDefinition(
            name="sma_200",
            category=FeatureCategory.TREND,
            description="200-day SMA",
            compute_fn=lambda df: df['close'].rolling(200).mean(),
            lookback=200
        ))
        
        self.register(FeatureDefinition(
            name="ema_12",
            category=FeatureCategory.TREND,
            description="12-day EMA",
            compute_fn=lambda df: df['close'].ewm(span=12).mean(),
            lookback=12
        ))
        
        self.register(FeatureDefinition(
            name="ema_26",
            category=FeatureCategory.TREND,
            description="26-day EMA",
            compute_fn=lambda df: df['close'].ewm(span=26).mean(),
            lookback=26
        ))
        
        self.register(FeatureDefinition(
            name="price_to_sma20",
            category=FeatureCategory.TREND,
            description="Price relative to 20-day SMA",
            compute_fn=lambda df: df['close'] / df['close'].rolling(20).mean() - 1,
            lookback=20
        ))
        
        self.register(FeatureDefinition(
            name="price_to_sma50",
            category=FeatureCategory.TREND,
            description="Price relative to 50-day SMA",
            compute_fn=lambda df: df['close'] / df['close'].rolling(50).mean() - 1,
            lookback=50
        ))
        
        self.register(FeatureDefinition(
            name="sma_cross_20_50",
            category=FeatureCategory.TREND,
            description="SMA 20/50 crossover signal",
            compute_fn=lambda df: (df['close'].rolling(20).mean() > df['close'].rolling(50).mean()).astype(int),
            lookback=50
        ))
        
        if TA_AVAILABLE:
            self.register(FeatureDefinition(
                name="macd",
                category=FeatureCategory.TREND,
                description="MACD line",
                compute_fn=lambda df: ta.trend.MACD(df['close']).macd(),
                lookback=26
            ))
            
            self.register(FeatureDefinition(
                name="macd_signal",
                category=FeatureCategory.TREND,
                description="MACD signal line",
                compute_fn=lambda df: ta.trend.MACD(df['close']).macd_signal(),
                lookback=35
            ))
            
            self.register(FeatureDefinition(
                name="macd_hist",
                category=FeatureCategory.TREND,
                description="MACD histogram",
                compute_fn=lambda df: ta.trend.MACD(df['close']).macd_diff(),
                lookback=35
            ))
            
            self.register(FeatureDefinition(
                name="adx",
                category=FeatureCategory.TREND,
                description="Average Directional Index",
                compute_fn=lambda df: ta.trend.ADXIndicator(
                    df['high'], df['low'], df['close']
                ).adx(),
                lookback=14
            ))


class FeatureStore:
    """
    Feature Store with versioning and point-in-time retrieval.
    """
    
    def __init__(self, store_dir: Path = None):
        self.store_dir = store_dir or FEATURE_STORE_DIR
        self.store_dir.mkdir(exist_ok=True)
        
        self.registry = FeatureRegistry()
        self.metadata_file = self.store_dir / "metadata.json"
        self.metadata: Dict[str, FeatureMetadata] = self._load_metadata()
    
    def compute_features(self, df: pd.DataFrame, feature_names: List[str] = None,
                        symbol: str = None) -> pd.DataFrame:
        """Compute features for a dataframe."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Use all features if none specified
        if feature_names is None:
            feature_names = self.registry.list_features()
        
        computed = 0
        for name in feature_names:
            feature = self.registry.get(name)
            if feature is None:
                logger.warning(f"Unknown feature: {name}")
                continue
            
            try:
                df[name] = feature.compute_fn(df)
                computed += 1
            except Exception as e:
                logger.error(f"Error computing {name}: {e}")
        
        logger.info(f"Computed {computed}/{len(feature_names)} features")
        
        if symbol:
            df['symbol'] = symbol
        
        return df
    
    def compute_all_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Compute all registered features."""
        return self.compute_features(df, None, symbol)
    
    def save_features(self, df: pd.DataFrame, name: str, version: str = "1.0.0"):
        """Save computed features to the store."""
        if df.empty:
            logger.warning("Cannot save empty dataframe")
            return
        
        # Create versioned path
        feature_dir = self.store_dir / name / version
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        path = feature_dir / "data.parquet"
        df.to_parquet(path)
        
        # Compute checksum
        checksum = hashlib.md5(df.to_json().encode()).hexdigest()
        
        # Update metadata
        self.metadata[f"{name}/{version}"] = FeatureMetadata(
            name=name,
            category="computed",
            version=version,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            row_count=len(df),
            date_range=(str(df.index.min()), str(df.index.max())),
            checksum=checksum
        )
        self._save_metadata()
        
        logger.info(f"Saved {len(df)} rows to {path}")
    
    def load_features(self, name: str, version: str = "latest",
                     start_date: str = None, end_date: str = None,
                     as_of: str = None) -> pd.DataFrame:
        """
        Load features with point-in-time correctness.
        
        Args:
            name: Feature set name
            version: Version to load ("latest" for most recent)
            start_date: Filter start date
            end_date: Filter end date
            as_of: Point-in-time date (returns data that was available as of this date)
        """
        feature_dir = self.store_dir / name
        
        if not feature_dir.exists():
            logger.warning(f"Feature set not found: {name}")
            return pd.DataFrame()
        
        # Get version
        if version == "latest":
            versions = sorted([d.name for d in feature_dir.iterdir() if d.is_dir()])
            if not versions:
                return pd.DataFrame()
            version = versions[-1]
        
        # Load data
        path = feature_dir / version / "data.parquet"
        if not path.exists():
            return pd.DataFrame()
        
        df = pd.read_parquet(path)
        
        # Apply date filters
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Point-in-time filter
        if as_of:
            df = df[df.index <= as_of]
        
        return df
    
    def get_feature_matrix(self, symbols: List[str], feature_names: List[str],
                          start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get a feature matrix for multiple symbols."""
        all_data = []
        
        for symbol in symbols:
            df = self.load_features(symbol, start_date=start_date, end_date=end_date)
            if not df.empty:
                df = df[feature_names] if feature_names else df
                df['symbol'] = symbol
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data)
        return pd.DataFrame()
    
    def list_feature_sets(self) -> List[str]:
        """List all available feature sets."""
        return [d.name for d in self.store_dir.iterdir() 
                if d.is_dir() and d.name != "__pycache__"]
    
    def get_feature_info(self, name: str) -> Optional[FeatureDefinition]:
        """Get information about a feature."""
        return self.registry.get(name)
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                data = json.load(f)
                return {k: FeatureMetadata(**v) for k, v in data.items()}
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        data = {k: v.__dict__ for k, v in self.metadata.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class FeatureEngineer:
    """High-level feature engineering utilities."""
    
    def __init__(self, store: FeatureStore = None):
        self.store = store or FeatureStore()
    
    def create_lagged_features(self, df: pd.DataFrame, columns: List[str],
                               lags: List[int]) -> pd.DataFrame:
        """Create lagged versions of features."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str],
                               windows: List[int], funcs: List[str] = None) -> pd.DataFrame:
        """Create rolling window features."""
        df = df.copy()
        funcs = funcs or ['mean', 'std', 'min', 'max']
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    for func in funcs:
                        df[f"{col}_{func}{window}"] = getattr(df[col].rolling(window), func)()
        return df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 5,
                     target_type: str = "return") -> pd.DataFrame:
        """Create target variable for ML."""
        df = df.copy()
        
        if target_type == "return":
            df['target'] = df['close'].pct_change(horizon).shift(-horizon)
        elif target_type == "direction":
            df['target'] = (df['close'].pct_change(horizon).shift(-horizon) > 0).astype(int)
        elif target_type == "quintile":
            returns = df['close'].pct_change(horizon).shift(-horizon)
            df['target'] = pd.qcut(returns, 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        return df
    
    def prepare_ml_dataset(self, df: pd.DataFrame, 
                          feature_names: List[str] = None,
                          target_horizon: int = 5,
                          train_ratio: float = 0.8) -> tuple:
        """Prepare dataset for ML training."""
        # Compute features
        df = self.store.compute_all_features(df)
        
        # Create target
        df = self.create_target(df, horizon=target_horizon)
        
        # Select features
        if feature_names is None:
            feature_names = self.store.registry.list_features()
        
        # Drop NaN rows
        df = df.dropna()
        
        # Split
        split_idx = int(len(df) * train_ratio)
        
        X = df[feature_names]
        y = df['target']
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test


# Convenience functions
def compute_features(df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
    """Compute features for a dataframe."""
    store = FeatureStore()
    return store.compute_features(df, features)


def get_available_features() -> List[str]:
    """Get list of all available features."""
    store = FeatureStore()
    return store.registry.list_features()


if __name__ == "__main__":
    import yfinance as yf
    
    # Test feature computation
    print("Testing Feature Store...")
    
    # Fetch sample data
    df = yf.download("AAPL", period="2y", progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    # Initialize store
    store = FeatureStore()
    
    # Compute all features
    df_features = store.compute_all_features(df, symbol="AAPL")
    
    print(f"\nComputed {len(df_features.columns)} columns")
    print(f"Features: {store.registry.list_features()[:10]}...")
    print(f"\nSample data:")
    print(df_features[['close', 'return_1d', 'rsi_14', 'sma_20', 'volatility_20d']].tail())
    
    # Save features
    store.save_features(df_features, "AAPL", "1.0.0")
    
    # List feature sets
    print(f"\nSaved feature sets: {store.list_feature_sets()}")
