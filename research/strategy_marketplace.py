"""
Strategy Marketplace & Sharing Hub
====================================
Share, discover, and collaborate on trading strategies.
Features version control, leaderboards, and community ratings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib
import os
import shutil
import warnings
warnings.filterwarnings('ignore')


class StrategyCategory(Enum):
    """Strategy categories"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MACHINE_LEARNING = "machine_learning"
    OPTIONS = "options"
    CRYPTO = "crypto"
    FACTOR = "factor"
    MULTI_ASSET = "multi_asset"
    OTHER = "other"


class StrategyStatus(Enum):
    """Publication status"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    UNDER_REVIEW = "under_review"


@dataclass
class StrategyMetadata:
    """Metadata for a shared strategy"""
    strategy_id: str
    name: str
    author: str
    description: str
    category: StrategyCategory
    tags: List[str]
    version: str
    created_at: datetime
    updated_at: datetime
    status: StrategyStatus = StrategyStatus.DRAFT
    
    # Performance metrics (from backtests)
    sharpe_ratio: Optional[float] = None
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    
    # Community metrics
    downloads: int = 0
    stars: int = 0
    forks: int = 0
    reviews: List[Dict] = field(default_factory=list)
    
    # Code references
    main_file: str = ""
    dependencies: List[str] = field(default_factory=list)
    min_capital: float = 10000
    
    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'author': self.author,
            'description': self.description,
            'category': self.category.value,
            'tags': self.tags,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.value,
            'sharpe_ratio': self.sharpe_ratio,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'downloads': self.downloads,
            'stars': self.stars,
            'forks': self.forks,
            'reviews': self.reviews,
            'main_file': self.main_file,
            'dependencies': self.dependencies,
            'min_capital': self.min_capital
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyMetadata':
        return cls(
            strategy_id=data['strategy_id'],
            name=data['name'],
            author=data['author'],
            description=data['description'],
            category=StrategyCategory(data['category']),
            tags=data['tags'],
            version=data['version'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            status=StrategyStatus(data['status']),
            sharpe_ratio=data.get('sharpe_ratio'),
            total_return=data.get('total_return'),
            max_drawdown=data.get('max_drawdown'),
            win_rate=data.get('win_rate'),
            downloads=data.get('downloads', 0),
            stars=data.get('stars', 0),
            forks=data.get('forks', 0),
            reviews=data.get('reviews', []),
            main_file=data.get('main_file', ''),
            dependencies=data.get('dependencies', []),
            min_capital=data.get('min_capital', 10000)
        )


@dataclass
class StrategyVersion:
    """Version of a strategy"""
    version: str
    commit_hash: str
    message: str
    timestamp: datetime
    changes: List[str]
    author: str


class StrategyRepository:
    """
    Local repository for strategy management
    ========================================
    Manages strategy files, versions, and exports.
    """
    
    def __init__(self, base_path: str = "strategies"):
        self.base_path = base_path
        self._ensure_structure()
    
    def _ensure_structure(self):
        """Create repository structure"""
        dirs = [
            self.base_path,
            os.path.join(self.base_path, 'published'),
            os.path.join(self.base_path, 'drafts'),
            os.path.join(self.base_path, 'imported'),
            os.path.join(self.base_path, 'backups')
        ]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
    
    def create_strategy(self, name: str, author: str, category: StrategyCategory,
                       description: str = "", tags: List[str] = None) -> StrategyMetadata:
        """Create a new strategy"""
        strategy_id = hashlib.md5(f"{name}_{author}_{datetime.now()}".encode()).hexdigest()[:12]
        
        metadata = StrategyMetadata(
            strategy_id=strategy_id,
            name=name,
            author=author,
            description=description,
            category=category,
            tags=tags or [],
            version="0.1.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=StrategyStatus.DRAFT
        )
        
        # Create strategy directory
        strategy_dir = os.path.join(self.base_path, 'drafts', strategy_id)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Save metadata
        self._save_metadata(strategy_id, metadata, 'drafts')
        
        # Create template files
        self._create_template_files(strategy_dir, name)
        
        return metadata
    
    def _create_template_files(self, strategy_dir: str, name: str):
        """Create template strategy files"""
        # Main strategy file
        main_content = f'''"""
{name}
{'=' * len(name)}
Auto-generated strategy template.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class Strategy:
    """
    Strategy implementation
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {{}}
        self.name = "{name}"
        
    def initialize(self):
        """Initialize strategy parameters"""
        # Add your initialization logic here
        pass
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals
        
        Parameters:
        -----------
        data : OHLCV DataFrame
        
        Returns:
        --------
        Series with signals: 1 (buy), -1 (sell), 0 (hold)
        """
        signals = pd.Series(0, index=data.index)
        
        # Add your signal logic here
        # Example: Simple moving average crossover
        sma_fast = data['close'].rolling(20).mean()
        sma_slow = data['close'].rolling(50).mean()
        
        signals[sma_fast > sma_slow] = 1
        signals[sma_fast < sma_slow] = -1
        
        return signals
    
    def on_data(self, current_bar: pd.Series, portfolio: Dict) -> Dict[str, Any]:
        """
        Called on each new data bar
        
        Returns:
        --------
        Action dict: {{'action': 'buy'/'sell'/'hold', 'size': float}}
        """
        return {{'action': 'hold', 'size': 0}}
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters for optimization"""
        return {{
            'sma_fast': {{'type': 'int', 'min': 5, 'max': 50, 'default': 20}},
            'sma_slow': {{'type': 'int', 'min': 20, 'max': 200, 'default': 50}}
        }}
'''
        
        with open(os.path.join(strategy_dir, 'strategy.py'), 'w') as f:
            f.write(main_content)
        
        # Config file
        config = {
            'name': name,
            'version': '0.1.0',
            'parameters': {
                'sma_fast': 20,
                'sma_slow': 50
            },
            'risk_management': {
                'stop_loss': 0.05,
                'take_profit': 0.10,
                'max_position_size': 0.25
            },
            'data_requirements': {
                'min_history': 100,
                'frequency': 'daily'
            }
        }
        
        with open(os.path.join(strategy_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # README
        readme = f"""# {name}

## Description
Add your strategy description here.

## Parameters
- `sma_fast`: Fast moving average period (default: 20)
- `sma_slow`: Slow moving average period (default: 50)

## Usage
```python
from strategy import Strategy

strategy = Strategy()
signals = strategy.generate_signals(data)
```

## Backtest Results
Add your backtest results here.

## Changelog
- v0.1.0: Initial version
"""
        
        with open(os.path.join(strategy_dir, 'README.md'), 'w') as f:
            f.write(readme)
    
    def _save_metadata(self, strategy_id: str, metadata: StrategyMetadata, folder: str):
        """Save strategy metadata"""
        meta_path = os.path.join(self.base_path, folder, strategy_id, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _load_metadata(self, strategy_id: str, folder: str) -> Optional[StrategyMetadata]:
        """Load strategy metadata"""
        meta_path = os.path.join(self.base_path, folder, strategy_id, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return StrategyMetadata.from_dict(json.load(f))
        return None
    
    def list_strategies(self, status: Optional[StrategyStatus] = None) -> List[StrategyMetadata]:
        """List all strategies"""
        strategies = []
        
        for folder in ['published', 'drafts', 'imported']:
            folder_path = os.path.join(self.base_path, folder)
            if os.path.exists(folder_path):
                for strategy_id in os.listdir(folder_path):
                    metadata = self._load_metadata(strategy_id, folder)
                    if metadata:
                        if status is None or metadata.status == status:
                            strategies.append(metadata)
        
        return strategies
    
    def publish_strategy(self, strategy_id: str, 
                        performance_metrics: Optional[Dict] = None) -> bool:
        """Publish a draft strategy"""
        # Find the strategy
        draft_path = os.path.join(self.base_path, 'drafts', strategy_id)
        if not os.path.exists(draft_path):
            return False
        
        # Load and update metadata
        metadata = self._load_metadata(strategy_id, 'drafts')
        if metadata is None:
            return False
        
        metadata.status = StrategyStatus.PUBLISHED
        metadata.updated_at = datetime.now()
        
        if performance_metrics:
            metadata.sharpe_ratio = performance_metrics.get('sharpe_ratio')
            metadata.total_return = performance_metrics.get('total_return')
            metadata.max_drawdown = performance_metrics.get('max_drawdown')
            metadata.win_rate = performance_metrics.get('win_rate')
        
        # Move to published folder
        published_path = os.path.join(self.base_path, 'published', strategy_id)
        shutil.copytree(draft_path, published_path)
        
        # Save updated metadata
        self._save_metadata(strategy_id, metadata, 'published')
        
        return True
    
    def export_strategy(self, strategy_id: str, export_path: str) -> bool:
        """Export a strategy as a zip package"""
        # Find strategy
        for folder in ['published', 'drafts', 'imported']:
            strategy_path = os.path.join(self.base_path, folder, strategy_id)
            if os.path.exists(strategy_path):
                shutil.make_archive(
                    os.path.join(export_path, strategy_id),
                    'zip',
                    strategy_path
                )
                return True
        return False
    
    def import_strategy(self, zip_path: str) -> Optional[StrategyMetadata]:
        """Import a strategy from zip package"""
        import zipfile
        
        # Extract strategy ID from filename
        filename = os.path.basename(zip_path).replace('.zip', '')
        
        import_path = os.path.join(self.base_path, 'imported', filename)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(import_path)
        
        # Load metadata
        return self._load_metadata(filename, 'imported')
    
    def update_version(self, strategy_id: str, new_version: str,
                      changes: List[str], message: str = "") -> bool:
        """Update strategy version"""
        for folder in ['published', 'drafts']:
            metadata = self._load_metadata(strategy_id, folder)
            if metadata:
                # Create backup
                backup_name = f"{strategy_id}_v{metadata.version}"
                strategy_path = os.path.join(self.base_path, folder, strategy_id)
                backup_path = os.path.join(self.base_path, 'backups', backup_name)
                shutil.copytree(strategy_path, backup_path)
                
                # Update version
                metadata.version = new_version
                metadata.updated_at = datetime.now()
                self._save_metadata(strategy_id, metadata, folder)
                
                return True
        return False


class StrategyMarketplace:
    """
    Strategy Marketplace
    =====================
    Discover, rate, and download community strategies.
    """
    
    def __init__(self, repository: StrategyRepository):
        self.repository = repository
        self.leaderboard_cache: Optional[pd.DataFrame] = None
        self.cache_time: Optional[datetime] = None
    
    def search(self, query: Optional[str] = None,
              category: Optional[StrategyCategory] = None,
              tags: Optional[List[str]] = None,
              min_sharpe: Optional[float] = None,
              sort_by: str = 'stars') -> List[StrategyMetadata]:
        """
        Search for strategies
        
        Parameters:
        -----------
        query : Text search in name/description
        category : Filter by category
        tags : Filter by tags
        min_sharpe : Minimum Sharpe ratio
        sort_by : 'stars', 'downloads', 'sharpe_ratio', 'created_at'
        """
        strategies = self.repository.list_strategies(StrategyStatus.PUBLISHED)
        
        # Text search
        if query:
            query = query.lower()
            strategies = [s for s in strategies 
                         if query in s.name.lower() or query in s.description.lower()]
        
        # Category filter
        if category:
            strategies = [s for s in strategies if s.category == category]
        
        # Tags filter
        if tags:
            strategies = [s for s in strategies 
                         if any(t in s.tags for t in tags)]
        
        # Performance filter
        if min_sharpe is not None:
            strategies = [s for s in strategies 
                         if s.sharpe_ratio and s.sharpe_ratio >= min_sharpe]
        
        # Sort
        if sort_by == 'stars':
            strategies.sort(key=lambda x: x.stars, reverse=True)
        elif sort_by == 'downloads':
            strategies.sort(key=lambda x: x.downloads, reverse=True)
        elif sort_by == 'sharpe_ratio':
            strategies.sort(key=lambda x: x.sharpe_ratio or 0, reverse=True)
        elif sort_by == 'created_at':
            strategies.sort(key=lambda x: x.created_at, reverse=True)
        
        return strategies
    
    def get_featured(self, limit: int = 10) -> List[StrategyMetadata]:
        """Get featured strategies (high quality + popular)"""
        strategies = self.repository.list_strategies(StrategyStatus.PUBLISHED)
        
        # Score based on multiple factors
        def score(s):
            sharpe_score = (s.sharpe_ratio or 0) * 10
            popularity_score = s.stars + s.downloads * 0.1
            recency_score = 1 / (1 + (datetime.now() - s.updated_at).days)
            return sharpe_score + popularity_score + recency_score * 5
        
        strategies.sort(key=score, reverse=True)
        return strategies[:limit]
    
    def get_leaderboard(self, metric: str = 'sharpe_ratio',
                       category: Optional[StrategyCategory] = None,
                       timeframe: str = 'all') -> pd.DataFrame:
        """
        Get strategy leaderboard
        
        Parameters:
        -----------
        metric : Ranking metric ('sharpe_ratio', 'total_return', 'win_rate')
        category : Filter by category
        timeframe : 'all', '1m', '3m', '6m', '1y'
        """
        strategies = self.repository.list_strategies(StrategyStatus.PUBLISHED)
        
        if category:
            strategies = [s for s in strategies if s.category == category]
        
        data = []
        for s in strategies:
            data.append({
                'rank': 0,
                'name': s.name,
                'author': s.author,
                'category': s.category.value,
                'sharpe_ratio': s.sharpe_ratio or 0,
                'total_return': (s.total_return or 0) * 100,
                'max_drawdown': (s.max_drawdown or 0) * 100,
                'win_rate': (s.win_rate or 0) * 100,
                'stars': s.stars,
                'downloads': s.downloads
            })
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            df = df.sort_values(metric, ascending=False)
            df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def star_strategy(self, strategy_id: str, user_id: str) -> bool:
        """Star a strategy"""
        for folder in ['published']:
            metadata = self.repository._load_metadata(strategy_id, folder)
            if metadata:
                metadata.stars += 1
                self.repository._save_metadata(strategy_id, metadata, folder)
                return True
        return False
    
    def add_review(self, strategy_id: str, user_id: str,
                  rating: int, comment: str) -> bool:
        """Add a review to a strategy"""
        if rating < 1 or rating > 5:
            return False
        
        for folder in ['published']:
            metadata = self.repository._load_metadata(strategy_id, folder)
            if metadata:
                review = {
                    'user_id': user_id,
                    'rating': rating,
                    'comment': comment,
                    'timestamp': datetime.now().isoformat()
                }
                metadata.reviews.append(review)
                self.repository._save_metadata(strategy_id, metadata, folder)
                return True
        return False
    
    def download_strategy(self, strategy_id: str, 
                         destination: str) -> bool:
        """Download a strategy"""
        success = self.repository.export_strategy(strategy_id, destination)
        
        if success:
            # Increment download count
            for folder in ['published']:
                metadata = self.repository._load_metadata(strategy_id, folder)
                if metadata:
                    metadata.downloads += 1
                    self.repository._save_metadata(strategy_id, metadata, folder)
        
        return success
    
    def get_recommendations(self, user_strategies: List[str],
                           limit: int = 5) -> List[StrategyMetadata]:
        """Get strategy recommendations based on user's existing strategies"""
        # Get user's strategy categories and tags
        user_categories = set()
        user_tags = set()
        
        for strategy_id in user_strategies:
            for folder in ['published', 'drafts', 'imported']:
                metadata = self.repository._load_metadata(strategy_id, folder)
                if metadata:
                    user_categories.add(metadata.category)
                    user_tags.update(metadata.tags)
        
        # Find similar strategies
        all_strategies = self.repository.list_strategies(StrategyStatus.PUBLISHED)
        
        recommendations = []
        for s in all_strategies:
            if s.strategy_id in user_strategies:
                continue
            
            score = 0
            if s.category in user_categories:
                score += 2
            score += len(set(s.tags) & user_tags)
            score += (s.sharpe_ratio or 0) * 0.5
            
            if score > 0:
                recommendations.append((s, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in recommendations[:limit]]


class StrategyCollaboration:
    """
    Collaboration tools for strategy development
    """
    
    def __init__(self, repository: StrategyRepository):
        self.repository = repository
    
    def fork_strategy(self, strategy_id: str, new_author: str) -> Optional[StrategyMetadata]:
        """Fork a strategy to create your own version"""
        # Find original
        original = None
        for folder in ['published', 'imported']:
            original = self.repository._load_metadata(strategy_id, folder)
            if original:
                break
        
        if not original:
            return None
        
        # Create fork
        forked = self.repository.create_strategy(
            name=f"{original.name} (Fork)",
            author=new_author,
            category=original.category,
            description=f"Forked from {original.name} by {original.author}",
            tags=original.tags + ['fork']
        )
        
        # Increment fork count on original
        original.forks += 1
        self.repository._save_metadata(
            strategy_id, original, 
            'published' if original.status == StrategyStatus.PUBLISHED else 'drafts'
        )
        
        return forked
    
    def compare_strategies(self, strategy_ids: List[str]) -> pd.DataFrame:
        """Compare multiple strategies"""
        data = []
        
        for strategy_id in strategy_ids:
            for folder in ['published', 'drafts', 'imported']:
                metadata = self.repository._load_metadata(strategy_id, folder)
                if metadata:
                    data.append({
                        'name': metadata.name,
                        'category': metadata.category.value,
                        'sharpe': metadata.sharpe_ratio,
                        'return': f"{(metadata.total_return or 0)*100:.1f}%",
                        'drawdown': f"{(metadata.max_drawdown or 0)*100:.1f}%",
                        'win_rate': f"{(metadata.win_rate or 0)*100:.1f}%",
                        'version': metadata.version,
                        'stars': metadata.stars
                    })
                    break
        
        return pd.DataFrame(data)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STRATEGY MARKETPLACE DEMO")
    print("=" * 60)
    
    # Initialize repository and marketplace
    repo = StrategyRepository("demo_strategies")
    marketplace = StrategyMarketplace(repo)
    collab = StrategyCollaboration(repo)
    
    # Create some strategies
    print("\n1. Creating strategies...")
    
    momentum_strategy = repo.create_strategy(
        name="Dual Momentum Pro",
        author="QuantDev",
        category=StrategyCategory.MOMENTUM,
        description="Advanced dual momentum strategy with regime detection",
        tags=["momentum", "systematic", "equity"]
    )
    print(f"   Created: {momentum_strategy.name} (ID: {momentum_strategy.strategy_id})")
    
    ml_strategy = repo.create_strategy(
        name="ML Ensemble Alpha",
        author="DataScientist",
        category=StrategyCategory.MACHINE_LEARNING,
        description="Ensemble of LSTM and XGBoost for price prediction",
        tags=["ml", "deep-learning", "prediction"]
    )
    print(f"   Created: {ml_strategy.name} (ID: {ml_strategy.strategy_id})")
    
    # Publish with metrics
    print("\n2. Publishing strategies...")
    
    repo.publish_strategy(momentum_strategy.strategy_id, {
        'sharpe_ratio': 1.85,
        'total_return': 0.245,
        'max_drawdown': -0.12,
        'win_rate': 0.58
    })
    
    repo.publish_strategy(ml_strategy.strategy_id, {
        'sharpe_ratio': 2.1,
        'total_return': 0.32,
        'max_drawdown': -0.15,
        'win_rate': 0.62
    })
    
    # Search marketplace
    print("\n3. Searching marketplace...")
    results = marketplace.search(min_sharpe=1.5, sort_by='sharpe_ratio')
    for s in results:
        print(f"   - {s.name}: Sharpe={s.sharpe_ratio:.2f}, Return={s.total_return*100:.1f}%")
    
    # Get leaderboard
    print("\n4. Strategy Leaderboard:")
    leaderboard = marketplace.get_leaderboard()
    if len(leaderboard) > 0:
        print(leaderboard.to_string(index=False))
    
    # Fork a strategy
    print("\n5. Forking strategy...")
    forked = collab.fork_strategy(ml_strategy.strategy_id, "NewTrader")
    if forked:
        print(f"   Forked: {forked.name}")
    
    # Compare strategies
    print("\n6. Strategy Comparison:")
    comparison = collab.compare_strategies([
        momentum_strategy.strategy_id,
        ml_strategy.strategy_id
    ])
    print(comparison.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Demo complete! Check 'demo_strategies' folder for files.")
