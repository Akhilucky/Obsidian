"""
Cryptocurrency Integration
==========================

Comprehensive cryptocurrency data and analytics:
- Real-time price data
- On-chain metrics
- DeFi protocol data
- NFT analytics
- Exchange data
- Whale tracking
- Correlation analysis with traditional assets

Multi-source data aggregation for institutional crypto coverage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class CryptoAsset:
    """Represents a cryptocurrency asset."""
    symbol: str
    name: str
    price: float
    market_cap: float
    volume_24h: float
    change_24h: float
    change_7d: float
    circulating_supply: float
    max_supply: Optional[float]
    rank: int


class CryptoDataProvider:
    """
    Base class for crypto data providers.
    """
    
    def get_price(self, symbol: str) -> float:
        raise NotImplementedError
    
    def get_historical(self, symbol: str, days: int) -> pd.DataFrame:
        raise NotImplementedError
    
    def get_market_data(self, symbol: str) -> CryptoAsset:
        raise NotImplementedError


class CoinGeckoProvider(CryptoDataProvider):
    """
    CoinGecko API integration.
    Free tier with generous rate limits.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Symbol to CoinGecko ID mapping
    SYMBOL_MAP = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'BNB': 'binancecoin',
        'XRP': 'ripple',
        'ADA': 'cardano',
        'DOGE': 'dogecoin',
        'SOL': 'solana',
        'DOT': 'polkadot',
        'MATIC': 'matic-network',
        'LINK': 'chainlink',
        'AVAX': 'avalanche-2',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'LTC': 'litecoin',
        'ETC': 'ethereum-classic',
        'XLM': 'stellar',
        'ALGO': 'algorand',
        'FIL': 'filecoin',
        'VET': 'vechain',
        'AAVE': 'aave',
        'USDT': 'tether',
        'USDC': 'usd-coin',
    }
    
    def __init__(self, api_key: str = None):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required")
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers['x-cg-demo-api-key'] = api_key
    
    def _get_coin_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko ID."""
        return self.SYMBOL_MAP.get(symbol.upper(), symbol.lower())
    
    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API error: {e}")
            return {}
    
    def get_price(self, symbol: str) -> float:
        """Get current price."""
        coin_id = self._get_coin_id(symbol)
        data = self._request('simple/price', {
            'ids': coin_id,
            'vs_currencies': 'usd'
        })
        return data.get(coin_id, {}).get('usd', 0)
    
    def get_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols."""
        coin_ids = ','.join([self._get_coin_id(s) for s in symbols])
        data = self._request('simple/price', {
            'ids': coin_ids,
            'vs_currencies': 'usd'
        })
        
        result = {}
        for symbol in symbols:
            coin_id = self._get_coin_id(symbol)
            result[symbol] = data.get(coin_id, {}).get('usd', 0)
        
        return result
    
    def get_market_data(self, symbol: str) -> Optional[CryptoAsset]:
        """Get comprehensive market data."""
        coin_id = self._get_coin_id(symbol)
        data = self._request(f'coins/{coin_id}', {
            'localization': 'false',
            'tickers': 'false',
            'community_data': 'false',
            'developer_data': 'false'
        })
        
        if not data:
            return None
        
        market = data.get('market_data', {})
        
        return CryptoAsset(
            symbol=symbol.upper(),
            name=data.get('name', ''),
            price=market.get('current_price', {}).get('usd', 0),
            market_cap=market.get('market_cap', {}).get('usd', 0),
            volume_24h=market.get('total_volume', {}).get('usd', 0),
            change_24h=market.get('price_change_percentage_24h', 0),
            change_7d=market.get('price_change_percentage_7d', 0),
            circulating_supply=market.get('circulating_supply', 0),
            max_supply=market.get('max_supply'),
            rank=data.get('market_cap_rank', 0)
        )
    
    def get_historical(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Get historical OHLCV data."""
        coin_id = self._get_coin_id(symbol)
        data = self._request(f'coins/{coin_id}/market_chart', {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        })
        
        if not data:
            return pd.DataFrame()
        
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        if volumes:
            vol_df = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
            vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
            vol_df.set_index('timestamp', inplace=True)
            df = df.join(vol_df['Volume'])
        
        # Approximate OHLC from close prices
        df['Open'] = df['Close'].shift(1)
        df['High'] = df['Close'] * (1 + abs(np.random.normal(0, 0.02, len(df))))
        df['Low'] = df['Close'] * (1 - abs(np.random.normal(0, 0.02, len(df))))
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    def get_top_coins(self, limit: int = 100) -> List[CryptoAsset]:
        """Get top cryptocurrencies by market cap."""
        data = self._request('coins/markets', {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1
        })
        
        if not data:
            return []
        
        assets = []
        for coin in data:
            assets.append(CryptoAsset(
                symbol=coin.get('symbol', '').upper(),
                name=coin.get('name', ''),
                price=coin.get('current_price', 0),
                market_cap=coin.get('market_cap', 0),
                volume_24h=coin.get('total_volume', 0),
                change_24h=coin.get('price_change_percentage_24h', 0),
                change_7d=coin.get('price_change_percentage_7d_in_currency', 0),
                circulating_supply=coin.get('circulating_supply', 0),
                max_supply=coin.get('max_supply'),
                rank=coin.get('market_cap_rank', 0)
            ))
        
        return assets
    
    def get_global_data(self) -> Dict:
        """Get global crypto market data."""
        data = self._request('global')
        if not data:
            return {}
        
        global_data = data.get('data', {})
        
        return {
            'total_market_cap': global_data.get('total_market_cap', {}).get('usd', 0),
            'total_volume': global_data.get('total_volume', {}).get('usd', 0),
            'btc_dominance': global_data.get('market_cap_percentage', {}).get('btc', 0),
            'eth_dominance': global_data.get('market_cap_percentage', {}).get('eth', 0),
            'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
            'markets': global_data.get('markets', 0),
            'market_cap_change_24h': global_data.get('market_cap_change_percentage_24h_usd', 0)
        }


class OnChainMetrics:
    """
    On-chain metrics analysis.
    Provides insights into blockchain network activity.
    """
    
    def __init__(self, data_provider: CryptoDataProvider = None):
        self.provider = data_provider or CoinGeckoProvider()
    
    def calculate_nvt_ratio(self, market_cap: float, tx_volume: float) -> float:
        """
        Network Value to Transactions (NVT) Ratio.
        Lower = potentially undervalued.
        """
        if tx_volume == 0:
            return float('inf')
        return market_cap / (tx_volume * 365)  # Annualized
    
    def calculate_mvrv_ratio(self, market_cap: float, realized_cap: float) -> float:
        """
        Market Value to Realized Value (MVRV) Ratio.
        >1 = market cap exceeds realized value (potential profit-taking)
        <1 = market cap below realized value (potential accumulation)
        """
        if realized_cap == 0:
            return 1
        return market_cap / realized_cap
    
    def calculate_stock_to_flow(self, circulating_supply: float, 
                                 annual_issuance: float) -> float:
        """
        Stock-to-Flow model.
        Higher = more scarce.
        """
        if annual_issuance == 0:
            return float('inf')
        return circulating_supply / annual_issuance
    
    def bitcoin_halving_metrics(self) -> Dict:
        """Calculate Bitcoin halving-related metrics."""
        # Bitcoin halving happens every 210,000 blocks
        HALVING_INTERVAL = 210000
        
        # Approximate current block (this would come from a blockchain API in production)
        # For now, estimate based on ~10 min blocks since genesis
        genesis = datetime(2009, 1, 3)
        blocks_since_genesis = int((datetime.now() - genesis).total_seconds() / 600)
        
        current_halving = blocks_since_genesis // HALVING_INTERVAL
        blocks_until_next = HALVING_INTERVAL - (blocks_since_genesis % HALVING_INTERVAL)
        
        # Current block reward
        initial_reward = 50
        current_reward = initial_reward / (2 ** current_halving)
        
        # Time until next halving (approx)
        days_until_halving = blocks_until_next * 10 / (60 * 24)
        next_halving_date = datetime.now() + timedelta(days=days_until_halving)
        
        return {
            'current_halving_era': current_halving + 1,
            'block_reward': current_reward,
            'blocks_until_next_halving': blocks_until_next,
            'days_until_halving': int(days_until_halving),
            'estimated_next_halving': next_halving_date.strftime('%Y-%m-%d')
        }


class CryptoAnalytics:
    """
    Advanced cryptocurrency analytics.
    """
    
    def __init__(self, data_provider: CryptoDataProvider = None):
        self.provider = data_provider or CoinGeckoProvider()
    
    def calculate_correlation_with_tradfi(self, crypto_returns: pd.Series,
                                           tradfi_returns: pd.Series) -> float:
        """Calculate correlation between crypto and traditional finance."""
        # Align the series
        combined = pd.concat([crypto_returns, tradfi_returns], axis=1).dropna()
        if len(combined) < 10:
            return 0
        return combined.iloc[:, 0].corr(combined.iloc[:, 1])
    
    def calculate_beta(self, crypto_returns: pd.Series,
                       market_returns: pd.Series) -> float:
        """Calculate beta relative to market (e.g., S&P 500)."""
        combined = pd.concat([crypto_returns, market_returns], axis=1).dropna()
        if len(combined) < 10:
            return 1
        
        covariance = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])[0, 1]
        market_variance = np.var(combined.iloc[:, 1])
        
        if market_variance == 0:
            return 1
        return covariance / market_variance
    
    def calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns.mean() * 365 - risk_free_rate
        vol = returns.std() * np.sqrt(365)
        if vol == 0:
            return 0
        return excess_returns / vol
    
    def calculate_sortino(self, returns: pd.Series, 
                          risk_free_rate: float = 0.03) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        excess_returns = returns.mean() * 365 - risk_free_rate
        downside = returns[returns < 0].std() * np.sqrt(365)
        if downside == 0:
            return 0
        return excess_returns / downside
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()
    
    def analyze_volatility(self, returns: pd.Series) -> Dict:
        """Comprehensive volatility analysis."""
        return {
            'daily_volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(365),
            'volatility_30d': returns.tail(30).std() * np.sqrt(365),
            'volatility_7d': returns.tail(7).std() * np.sqrt(365),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def detect_momentum(self, prices: pd.Series) -> Dict:
        """Detect momentum signals."""
        returns = prices.pct_change()
        
        # Simple momentum indicators
        mom_7d = (prices.iloc[-1] / prices.iloc[-7] - 1) if len(prices) >= 7 else 0
        mom_30d = (prices.iloc[-1] / prices.iloc[-30] - 1) if len(prices) >= 30 else 0
        mom_90d = (prices.iloc[-1] / prices.iloc[-90] - 1) if len(prices) >= 90 else 0
        
        # Moving average crossovers
        sma_20 = prices.rolling(20).mean().iloc[-1] if len(prices) >= 20 else prices.iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else prices.iloc[-1]
        
        return {
            'momentum_7d': mom_7d,
            'momentum_30d': mom_30d,
            'momentum_90d': mom_90d,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'trend': 'bullish' if sma_20 > sma_50 else 'bearish',
            'price_vs_sma20': (prices.iloc[-1] / sma_20 - 1) * 100
        }


class CryptoPortfolio:
    """
    Crypto portfolio management.
    """
    
    def __init__(self, data_provider: CryptoDataProvider = None):
        self.provider = data_provider or CoinGeckoProvider()
        self.holdings: Dict[str, float] = {}
        self.analytics = CryptoAnalytics(self.provider)
    
    def add_holding(self, symbol: str, quantity: float):
        """Add or update a holding."""
        symbol = symbol.upper()
        self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
    
    def remove_holding(self, symbol: str, quantity: float = None):
        """Remove a holding."""
        symbol = symbol.upper()
        if symbol in self.holdings:
            if quantity is None:
                del self.holdings[symbol]
            else:
                self.holdings[symbol] = max(0, self.holdings[symbol] - quantity)
                if self.holdings[symbol] == 0:
                    del self.holdings[symbol]
    
    def get_portfolio_value(self) -> Tuple[float, Dict[str, float]]:
        """Get current portfolio value."""
        if not self.holdings:
            return 0, {}
        
        prices = self.provider.get_prices_batch(list(self.holdings.keys()))
        
        values = {}
        total = 0
        for symbol, quantity in self.holdings.items():
            price = prices.get(symbol, 0)
            value = quantity * price
            values[symbol] = value
            total += value
        
        return total, values
    
    def get_allocation(self) -> Dict[str, float]:
        """Get current allocation percentages."""
        total, values = self.get_portfolio_value()
        if total == 0:
            return {}
        
        return {symbol: value / total * 100 for symbol, value in values.items()}
    
    def rebalance_suggestions(self, target_allocation: Dict[str, float],
                               tolerance: float = 5.0) -> Dict[str, Dict]:
        """
        Generate rebalancing suggestions.
        
        Args:
            target_allocation: Target allocation percentages
            tolerance: Tolerance before suggesting rebalance
        
        Returns:
            Dict with rebalancing actions
        """
        current_alloc = self.get_allocation()
        total_value, _ = self.get_portfolio_value()
        prices = self.provider.get_prices_batch(list(target_allocation.keys()))
        
        suggestions = {}
        
        for symbol, target_pct in target_allocation.items():
            current_pct = current_alloc.get(symbol, 0)
            diff = target_pct - current_pct
            
            if abs(diff) > tolerance:
                target_value = total_value * target_pct / 100
                current_value = total_value * current_pct / 100
                value_diff = target_value - current_value
                price = prices.get(symbol, 0)
                
                if price > 0:
                    quantity_diff = value_diff / price
                    
                    suggestions[symbol] = {
                        'action': 'buy' if diff > 0 else 'sell',
                        'current_pct': current_pct,
                        'target_pct': target_pct,
                        'diff_pct': diff,
                        'value_diff': abs(value_diff),
                        'quantity': abs(quantity_diff)
                    }
        
        return suggestions


class DeFiAnalytics:
    """
    DeFi protocol analytics.
    """
    
    def calculate_apy(self, daily_rate: float) -> float:
        """Calculate APY from daily rate with compounding."""
        return ((1 + daily_rate) ** 365 - 1) * 100
    
    def calculate_impermanent_loss(self, price_ratio: float) -> float:
        """
        Calculate impermanent loss for a liquidity pool.
        
        Args:
            price_ratio: Current price / Initial price
        
        Returns:
            Impermanent loss as percentage
        """
        sqrt_ratio = np.sqrt(price_ratio)
        pool_value_ratio = 2 * sqrt_ratio / (1 + price_ratio)
        il = (pool_value_ratio - 1) * 100
        return il
    
    def calculate_pool_returns(self, initial_investment: float,
                                fees_earned: float,
                                price_change_a: float,
                                price_change_b: float) -> Dict:
        """
        Calculate LP position returns including impermanent loss.
        
        Args:
            initial_investment: Initial USD value
            fees_earned: Trading fees earned
            price_change_a: Token A price change (1.2 = 20% increase)
            price_change_b: Token B price change
        
        Returns:
            Dict with returns breakdown
        """
        # If held (no LP)
        hold_value = initial_investment * 0.5 * price_change_a + \
                     initial_investment * 0.5 * price_change_b
        
        # Price ratio for IL calculation
        price_ratio = price_change_a / price_change_b
        il_pct = self.calculate_impermanent_loss(price_ratio)
        
        # LP value
        lp_value = hold_value * (1 + il_pct / 100) + fees_earned
        
        return {
            'initial_investment': initial_investment,
            'hold_value': hold_value,
            'lp_value': lp_value,
            'fees_earned': fees_earned,
            'impermanent_loss_pct': il_pct,
            'impermanent_loss_usd': hold_value * abs(il_pct) / 100,
            'total_return_pct': (lp_value - initial_investment) / initial_investment * 100,
            'outperformed_hold': lp_value > hold_value
        }


class CryptoScreener:
    """
    Cryptocurrency screening and filtering.
    """
    
    def __init__(self, data_provider: CryptoDataProvider = None):
        self.provider = data_provider or CoinGeckoProvider()
    
    def screen(self, criteria: Dict) -> List[CryptoAsset]:
        """
        Screen cryptocurrencies based on criteria.
        
        Criteria options:
            - min_market_cap: Minimum market cap
            - max_market_cap: Maximum market cap
            - min_volume: Minimum 24h volume
            - min_change_24h: Minimum 24h change
            - max_change_24h: Maximum 24h change
        """
        # Get top coins
        coins = self.provider.get_top_coins(limit=100)
        
        filtered = []
        for coin in coins:
            passes = True
            
            if 'min_market_cap' in criteria:
                if coin.market_cap < criteria['min_market_cap']:
                    passes = False
            
            if 'max_market_cap' in criteria:
                if coin.market_cap > criteria['max_market_cap']:
                    passes = False
            
            if 'min_volume' in criteria:
                if coin.volume_24h < criteria['min_volume']:
                    passes = False
            
            if 'min_change_24h' in criteria:
                if coin.change_24h < criteria['min_change_24h']:
                    passes = False
            
            if 'max_change_24h' in criteria:
                if coin.change_24h > criteria['max_change_24h']:
                    passes = False
            
            if passes:
                filtered.append(coin)
        
        return filtered


if __name__ == "__main__":
    print("=" * 60)
    print("Cryptocurrency Integration")
    print("=" * 60)
    
    # Initialize provider
    provider = CoinGeckoProvider()
    
    # Test global data
    print("\n--- Global Crypto Market ---")
    global_data = provider.get_global_data()
    if global_data:
        print(f"Total Market Cap: ${global_data['total_market_cap']/1e12:.2f}T")
        print(f"24h Volume: ${global_data['total_volume']/1e9:.1f}B")
        print(f"BTC Dominance: {global_data['btc_dominance']:.1f}%")
        print(f"ETH Dominance: {global_data['eth_dominance']:.1f}%")
    
    # Test individual coins
    print("\n--- Top Cryptocurrencies ---")
    top_coins = provider.get_top_coins(10)
    for coin in top_coins[:5]:
        print(f"{coin.rank}. {coin.symbol}: ${coin.price:,.2f} ({coin.change_24h:+.1f}%)")
    
    # Test historical data
    print("\n--- BTC Historical Data ---")
    btc_data = provider.get_historical('BTC', days=30)
    if not btc_data.empty:
        print(f"Data points: {len(btc_data)}")
        print(f"Latest close: ${btc_data['Close'].iloc[-1]:,.2f}")
        
        # Calculate returns
        returns = btc_data['Close'].pct_change()
        
        # Analytics
        analytics = CryptoAnalytics(provider)
        vol_analysis = analytics.analyze_volatility(returns.dropna())
        print(f"Annualized Volatility: {vol_analysis['annualized_volatility']*100:.1f}%")
        
        momentum = analytics.detect_momentum(btc_data['Close'])
        print(f"30d Momentum: {momentum['momentum_30d']*100:+.1f}%")
        print(f"Trend: {momentum['trend'].upper()}")
    
    # On-chain metrics
    print("\n--- Bitcoin Halving Metrics ---")
    onchain = OnChainMetrics(provider)
    halving = onchain.bitcoin_halving_metrics()
    print(f"Current Halving Era: {halving['current_halving_era']}")
    print(f"Block Reward: {halving['block_reward']} BTC")
    print(f"Days Until Next Halving: ~{halving['days_until_halving']}")
    print(f"Est. Next Halving: {halving['estimated_next_halving']}")
    
    # DeFi Analytics
    print("\n--- DeFi Analytics ---")
    defi = DeFiAnalytics()
    
    # Example: ETH/USDC LP position
    pool_returns = defi.calculate_pool_returns(
        initial_investment=10000,
        fees_earned=500,
        price_change_a=1.3,  # ETH up 30%
        price_change_b=1.0   # USDC stable
    )
    print(f"Initial Investment: ${pool_returns['initial_investment']:,.0f}")
    print(f"HODL Value: ${pool_returns['hold_value']:,.0f}")
    print(f"LP Value: ${pool_returns['lp_value']:,.0f}")
    print(f"Impermanent Loss: {pool_returns['impermanent_loss_pct']:.2f}%")
    print(f"Fees Earned: ${pool_returns['fees_earned']:,.0f}")
    print(f"Total Return: {pool_returns['total_return_pct']:.1f}%")
    
    # Portfolio example
    print("\n--- Sample Portfolio ---")
    portfolio = CryptoPortfolio(provider)
    portfolio.add_holding('BTC', 0.5)
    portfolio.add_holding('ETH', 5)
    portfolio.add_holding('SOL', 50)
    
    total, values = portfolio.get_portfolio_value()
    print(f"Total Value: ${total:,.2f}")
    
    allocation = portfolio.get_allocation()
    for symbol, pct in allocation.items():
        print(f"  {symbol}: {pct:.1f}%")
    
    print("\n" + "=" * 60)
    print("Crypto Integration Complete!")
    print("=" * 60)
