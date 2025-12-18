"""
Market Making Trading Strategies
=================================
High-frequency market making strategies including bid-ask spread capture,
order book analysis, and micro-trading algorithms.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import heapq
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "limit"
    MARKET = "market"
    IOC = "ioc"  # Immediate or Cancel


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partial"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Represents a single order."""
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0


@dataclass
class Quote:
    """Market quote with bid/ask."""
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp: datetime
    
    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid_price) * 10000


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    quantity: float
    order_count: int = 1


@dataclass
class OrderBook:
    """Full order book representation."""
    symbol: str
    bids: List[OrderBookLevel]  # Sorted descending by price
    asks: List[OrderBookLevel]  # Sorted ascending by price
    timestamp: datetime
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return float('inf')
    
    def get_vwap(self, side: OrderSide, quantity: float) -> float:
        """Calculate VWAP to fill a given quantity."""
        levels = self.asks if side == OrderSide.BUY else self.bids
        remaining = quantity
        total_cost = 0.0
        total_qty = 0.0
        
        for level in levels:
            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            total_qty += fill_qty
            remaining -= fill_qty
            if remaining <= 0:
                break
        
        return total_cost / total_qty if total_qty > 0 else 0.0
    
    def get_market_impact(self, side: OrderSide, quantity: float) -> float:
        """Estimate market impact of an order."""
        vwap = self.get_vwap(side, quantity)
        return abs(vwap - self.mid_price) / self.mid_price if self.mid_price > 0 else 0.0


@dataclass
class MarketMakingConfig:
    """Configuration for market making strategies."""
    # Spread parameters
    min_spread_bps: float = 5.0
    target_spread_bps: float = 10.0
    max_spread_bps: float = 50.0
    
    # Position limits
    max_position: float = 1000.0
    position_limit_pct: float = 0.8  # Start reducing size at this % of max
    
    # Order sizing
    base_order_size: float = 100.0
    min_order_size: float = 10.0
    size_increment: float = 10.0
    
    # Risk parameters
    max_loss_per_trade: float = 50.0
    daily_loss_limit: float = 1000.0
    inventory_skew_factor: float = 0.5
    
    # Timing
    quote_lifetime_ms: int = 1000
    min_time_between_trades_ms: int = 100
    
    # Market conditions
    volatility_adjustment: bool = True
    volume_adjustment: bool = True


class InventoryManager:
    """Manages position inventory and risk."""
    
    def __init__(self, config: MarketMakingConfig):
        self.config = config
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.trade_count = 0
        self.daily_pnl = 0.0
    
    def update_position(self, side: OrderSide, quantity: float, price: float):
        """Update position after a fill."""
        if side == OrderSide.BUY:
            # Update average entry for longs
            if self.position >= 0:
                total_cost = self.position * self.avg_entry_price + quantity * price
                self.position += quantity
                self.avg_entry_price = total_cost / self.position if self.position > 0 else 0
            else:
                # Closing short position
                pnl = (self.avg_entry_price - price) * min(quantity, abs(self.position))
                self.realized_pnl += pnl
                self.daily_pnl += pnl
                self.position += quantity
                if self.position > 0:
                    self.avg_entry_price = price
        else:
            # Sell
            if self.position <= 0:
                total_cost = abs(self.position) * self.avg_entry_price + quantity * price
                self.position -= quantity
                self.avg_entry_price = total_cost / abs(self.position) if self.position < 0 else 0
            else:
                # Closing long position
                pnl = (price - self.avg_entry_price) * min(quantity, self.position)
                self.realized_pnl += pnl
                self.daily_pnl += pnl
                self.position -= quantity
                if self.position < 0:
                    self.avg_entry_price = price
        
        self.trade_count += 1
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.position > 0:
            return (current_price - self.avg_entry_price) * self.position
        elif self.position < 0:
            return (self.avg_entry_price - current_price) * abs(self.position)
        return 0.0
    
    def get_inventory_skew(self) -> float:
        """
        Calculate inventory skew for quote adjustment.
        Returns value between -1 (max short) and 1 (max long).
        """
        return self.position / self.config.max_position
    
    def can_trade(self, side: OrderSide, quantity: float) -> bool:
        """Check if a trade is allowed given current inventory."""
        new_position = self.position + (quantity if side == OrderSide.BUY else -quantity)
        
        # Check position limits
        if abs(new_position) > self.config.max_position:
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.config.daily_loss_limit:
            return False
        
        return True
    
    def get_adjusted_size(self, side: OrderSide) -> float:
        """Get position-adjusted order size."""
        base_size = self.config.base_order_size
        skew = self.get_inventory_skew()
        
        # Reduce size when approaching limits
        position_ratio = abs(self.position) / self.config.max_position
        if position_ratio > self.config.position_limit_pct:
            reduction = (position_ratio - self.config.position_limit_pct) / (1 - self.config.position_limit_pct)
            base_size *= (1 - reduction * 0.8)
        
        # Skew size based on inventory
        if side == OrderSide.BUY:
            # Reduce buy size if long, increase if short
            adjustment = 1 - skew * self.config.inventory_skew_factor
        else:
            # Reduce sell size if short, increase if long
            adjustment = 1 + skew * self.config.inventory_skew_factor
        
        adjusted_size = base_size * adjustment
        return max(adjusted_size, self.config.min_order_size)
    
    def reset_daily(self):
        """Reset daily P&L tracking."""
        self.daily_pnl = 0.0


class MarketMakingStrategy(ABC):
    """Base class for market making strategies."""
    
    def __init__(self, config: MarketMakingConfig = None):
        self.config = config or MarketMakingConfig()
        self.inventory = InventoryManager(self.config)
        self.active_orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.quote_history: Deque[Quote] = deque(maxlen=1000)
        self.trade_history: List[Dict] = []
    
    @abstractmethod
    def generate_quotes(self, order_book: OrderBook, **kwargs) -> Tuple[Optional[Quote], Optional[Quote]]:
        """Generate bid and ask quotes."""
        pass
    
    def submit_order(self, order: Order) -> bool:
        """Submit an order (simulation)."""
        if not self.inventory.can_trade(order.side, order.quantity):
            return False
        
        self.active_orders[order.order_id] = order
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if order_id in self.active_orders:
            self.active_orders[order_id].status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
            return True
        return False
    
    def cancel_all_orders(self):
        """Cancel all active orders."""
        for order_id in list(self.active_orders.keys()):
            self.cancel_order(order_id)
    
    def on_fill(self, order_id: str, fill_qty: float, fill_price: float):
        """Handle order fill."""
        if order_id not in self.active_orders:
            return
        
        order = self.active_orders[order_id]
        order.filled_quantity += fill_qty
        order.avg_fill_price = (
            (order.avg_fill_price * (order.filled_quantity - fill_qty) + fill_price * fill_qty)
            / order.filled_quantity
        )
        
        # Update inventory
        self.inventory.update_position(order.side, fill_qty, fill_price)
        
        # Update order status
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            del self.active_orders[order_id]
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Record trade
        self.trade_history.append({
            'order_id': order_id,
            'side': order.side.value,
            'quantity': fill_qty,
            'price': fill_price,
            'timestamp': datetime.now(),
            'position': self.inventory.position,
            'pnl': self.inventory.realized_pnl
        })
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"ORD-{self.order_counter:06d}"


class BasicMarketMaker(MarketMakingStrategy):
    """
    Basic Market Making Strategy
    
    Provides liquidity by posting symmetric quotes around mid price,
    with inventory-based skew adjustments.
    """
    
    def __init__(self, config: MarketMakingConfig = None):
        super().__init__(config)
        self.volatility_window: Deque[float] = deque(maxlen=100)
    
    def generate_quotes(self, 
                        order_book: OrderBook,
                        volatility: float = None,
                        **kwargs) -> Tuple[Optional[Order], Optional[Order]]:
        """
        Generate bid and ask quotes.
        
        Args:
            order_book: Current order book state
            volatility: Current volatility estimate
            
        Returns:
            Tuple of (bid_order, ask_order)
        """
        mid_price = order_book.mid_price
        if mid_price <= 0:
            return None, None
        
        # Calculate spread
        base_spread_bps = self.config.target_spread_bps
        
        # Volatility adjustment
        if self.config.volatility_adjustment and volatility:
            self.volatility_window.append(volatility)
            avg_vol = np.mean(self.volatility_window)
            vol_ratio = volatility / avg_vol if avg_vol > 0 else 1.0
            base_spread_bps *= (0.5 + 0.5 * vol_ratio)
        
        # Clamp spread
        spread_bps = max(self.config.min_spread_bps, 
                        min(base_spread_bps, self.config.max_spread_bps))
        half_spread = mid_price * (spread_bps / 10000) / 2
        
        # Inventory skew
        skew = self.inventory.get_inventory_skew()
        skew_adjustment = mid_price * abs(skew) * (self.config.inventory_skew_factor / 100)
        
        # Calculate prices
        if skew > 0:  # Long inventory, want to sell more
            bid_price = mid_price - half_spread - skew_adjustment
            ask_price = mid_price + half_spread - skew_adjustment
        elif skew < 0:  # Short inventory, want to buy more
            bid_price = mid_price - half_spread + skew_adjustment
            ask_price = mid_price + half_spread + skew_adjustment
        else:
            bid_price = mid_price - half_spread
            ask_price = mid_price + half_spread
        
        # Get adjusted sizes
        bid_size = self.inventory.get_adjusted_size(OrderSide.BUY)
        ask_size = self.inventory.get_adjusted_size(OrderSide.SELL)
        
        # Create orders
        bid_order = Order(
            order_id=self._generate_order_id(),
            symbol=order_book.symbol,
            side=OrderSide.BUY,
            price=round(bid_price, 2),
            quantity=bid_size,
            order_type=OrderType.LIMIT
        )
        
        ask_order = Order(
            order_id=self._generate_order_id(),
            symbol=order_book.symbol,
            side=OrderSide.SELL,
            price=round(ask_price, 2),
            quantity=ask_size,
            order_type=OrderType.LIMIT
        )
        
        return bid_order, ask_order


class OrderBookImbalanceStrategy(MarketMakingStrategy):
    """
    Order Book Imbalance Strategy
    
    Analyzes order book imbalance to predict short-term price moves
    and adjusts quotes accordingly.
    """
    
    def __init__(self, 
                 config: MarketMakingConfig = None,
                 imbalance_threshold: float = 0.3,
                 depth_levels: int = 5):
        super().__init__(config)
        self.imbalance_threshold = imbalance_threshold
        self.depth_levels = depth_levels
        self.imbalance_history: Deque[float] = deque(maxlen=50)
    
    def calculate_imbalance(self, order_book: OrderBook) -> float:
        """
        Calculate order book imbalance.
        
        Returns value between -1 (sell pressure) and 1 (buy pressure).
        """
        bid_volume = sum(level.quantity for level in order_book.bids[:self.depth_levels])
        ask_volume = sum(level.quantity for level in order_book.asks[:self.depth_levels])
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        imbalance = (bid_volume - ask_volume) / total
        return imbalance
    
    def calculate_weighted_imbalance(self, order_book: OrderBook) -> float:
        """
        Calculate price-weighted order book imbalance.
        Levels closer to mid get higher weight.
        """
        mid = order_book.mid_price
        if mid <= 0:
            return 0.0
        
        bid_weighted = 0.0
        ask_weighted = 0.0
        
        for i, level in enumerate(order_book.bids[:self.depth_levels]):
            distance = abs(level.price - mid) / mid
            weight = 1 / (1 + distance * 100)  # Decay with distance
            bid_weighted += level.quantity * weight
        
        for i, level in enumerate(order_book.asks[:self.depth_levels]):
            distance = abs(level.price - mid) / mid
            weight = 1 / (1 + distance * 100)
            ask_weighted += level.quantity * weight
        
        total = bid_weighted + ask_weighted
        if total == 0:
            return 0.0
        
        return (bid_weighted - ask_weighted) / total
    
    def predict_price_direction(self, order_book: OrderBook) -> int:
        """
        Predict short-term price direction based on imbalance.
        
        Returns:
            1 for up, -1 for down, 0 for neutral
        """
        imbalance = self.calculate_weighted_imbalance(order_book)
        self.imbalance_history.append(imbalance)
        
        # Average recent imbalance
        avg_imbalance = np.mean(self.imbalance_history)
        
        if avg_imbalance > self.imbalance_threshold:
            return 1  # Expect price to rise
        elif avg_imbalance < -self.imbalance_threshold:
            return -1  # Expect price to fall
        return 0
    
    def generate_quotes(self,
                        order_book: OrderBook,
                        **kwargs) -> Tuple[Optional[Order], Optional[Order]]:
        """
        Generate quotes based on order book imbalance.
        """
        mid_price = order_book.mid_price
        if mid_price <= 0:
            return None, None
        
        imbalance = self.calculate_weighted_imbalance(order_book)
        direction = self.predict_price_direction(order_book)
        
        # Base spread
        spread_bps = self.config.target_spread_bps
        half_spread = mid_price * (spread_bps / 10000) / 2
        
        # Adjust based on imbalance
        imbalance_adjustment = mid_price * abs(imbalance) * 0.001
        
        if direction > 0:
            # Expect price rise: tighter bid, wider ask
            bid_price = mid_price - half_spread + imbalance_adjustment
            ask_price = mid_price + half_spread + imbalance_adjustment
            # Larger bid size to capture upward move
            bid_size = self.config.base_order_size * (1 + abs(imbalance))
            ask_size = self.config.base_order_size * (1 - abs(imbalance) * 0.5)
        elif direction < 0:
            # Expect price fall: wider bid, tighter ask
            bid_price = mid_price - half_spread - imbalance_adjustment
            ask_price = mid_price + half_spread - imbalance_adjustment
            bid_size = self.config.base_order_size * (1 - abs(imbalance) * 0.5)
            ask_size = self.config.base_order_size * (1 + abs(imbalance))
        else:
            bid_price = mid_price - half_spread
            ask_price = mid_price + half_spread
            bid_size = ask_size = self.config.base_order_size
        
        # Apply inventory skew
        inventory_skew = self.inventory.get_inventory_skew()
        skew_adjustment = mid_price * abs(inventory_skew) * 0.0005
        if inventory_skew > 0:
            bid_price -= skew_adjustment
            ask_price -= skew_adjustment
        elif inventory_skew < 0:
            bid_price += skew_adjustment
            ask_price += skew_adjustment
        
        bid_order = Order(
            order_id=self._generate_order_id(),
            symbol=order_book.symbol,
            side=OrderSide.BUY,
            price=round(bid_price, 2),
            quantity=max(bid_size, self.config.min_order_size),
            order_type=OrderType.LIMIT
        )
        
        ask_order = Order(
            order_id=self._generate_order_id(),
            symbol=order_book.symbol,
            side=OrderSide.SELL,
            price=round(ask_price, 2),
            quantity=max(ask_size, self.config.min_order_size),
            order_type=OrderType.LIMIT
        )
        
        return bid_order, ask_order


class MicroTradingStrategy(MarketMakingStrategy):
    """
    High-Frequency Micro-Trading Strategy
    
    Captures small price movements with very short holding periods.
    Uses tick-by-tick analysis for entry/exit decisions.
    """
    
    def __init__(self,
                 config: MarketMakingConfig = None,
                 tick_window: int = 50,
                 momentum_threshold: float = 0.0001):
        super().__init__(config)
        self.tick_window = tick_window
        self.momentum_threshold = momentum_threshold
        self.tick_history: Deque[Dict] = deque(maxlen=tick_window)
        self.last_trade_time: Optional[datetime] = None
    
    def update_tick(self, price: float, volume: float, timestamp: datetime):
        """Record a new tick."""
        self.tick_history.append({
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
    
    def calculate_tick_momentum(self) -> float:
        """Calculate momentum from recent ticks."""
        if len(self.tick_history) < 10:
            return 0.0
        
        prices = [t['price'] for t in self.tick_history]
        returns = np.diff(prices) / prices[:-1]
        
        # Volume-weighted momentum
        volumes = [t['volume'] for t in list(self.tick_history)[1:]]
        if sum(volumes) == 0:
            return np.mean(returns)
        
        weighted_returns = np.sum(returns * volumes) / np.sum(volumes)
        return weighted_returns
    
    def calculate_tick_volatility(self) -> float:
        """Calculate tick-level volatility."""
        if len(self.tick_history) < 10:
            return 0.0
        
        prices = [t['price'] for t in self.tick_history]
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def detect_microstructure_pattern(self) -> str:
        """
        Detect microstructure patterns in tick data.
        
        Returns pattern type: 'momentum_up', 'momentum_down', 
                             'mean_revert', 'range_bound', 'unknown'
        """
        if len(self.tick_history) < self.tick_window:
            return 'unknown'
        
        prices = [t['price'] for t in self.tick_history]
        returns = np.diff(prices) / prices[:-1]
        
        momentum = self.calculate_tick_momentum()
        
        # Check for trending
        if momentum > self.momentum_threshold:
            return 'momentum_up'
        elif momentum < -self.momentum_threshold:
            return 'momentum_down'
        
        # Check for mean reversion
        first_half = returns[:len(returns)//2]
        second_half = returns[len(returns)//2:]
        
        if np.mean(first_half) * np.mean(second_half) < 0:
            return 'mean_revert'
        
        # Range bound
        price_range = (max(prices) - min(prices)) / np.mean(prices)
        if price_range < 0.0005:
            return 'range_bound'
        
        return 'unknown'
    
    def generate_quotes(self,
                        order_book: OrderBook,
                        **kwargs) -> Tuple[Optional[Order], Optional[Order]]:
        """
        Generate quotes based on microstructure analysis.
        """
        mid_price = order_book.mid_price
        if mid_price <= 0:
            return None, None
        
        # Check cooldown
        now = datetime.now()
        if self.last_trade_time:
            elapsed = (now - self.last_trade_time).total_seconds() * 1000
            if elapsed < self.config.min_time_between_trades_ms:
                return None, None
        
        pattern = self.detect_microstructure_pattern()
        momentum = self.calculate_tick_momentum()
        volatility = self.calculate_tick_volatility()
        
        # Adjust spread based on volatility
        vol_multiplier = 1 + volatility * 1000
        spread_bps = self.config.target_spread_bps * vol_multiplier
        spread_bps = max(self.config.min_spread_bps, 
                        min(spread_bps, self.config.max_spread_bps))
        half_spread = mid_price * (spread_bps / 10000) / 2
        
        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread
        bid_size = ask_size = self.config.base_order_size
        
        if pattern == 'momentum_up':
            # Chase the move
            bid_price += half_spread * 0.3
            ask_price += half_spread * 0.5
            bid_size *= 1.5
        elif pattern == 'momentum_down':
            bid_price -= half_spread * 0.5
            ask_price -= half_spread * 0.3
            ask_size *= 1.5
        elif pattern == 'mean_revert':
            # Fade the move
            if momentum > 0:
                ask_size *= 1.5
            else:
                bid_size *= 1.5
        elif pattern == 'range_bound':
            # Tight quotes, capture spread
            half_spread *= 0.7
            bid_price = mid_price - half_spread
            ask_price = mid_price + half_spread
        
        # Apply inventory skew
        skew = self.inventory.get_inventory_skew()
        adjustment = mid_price * abs(skew) * 0.0005
        if skew > 0:
            bid_price -= adjustment
            ask_price -= adjustment * 0.5
            ask_size *= (1 + abs(skew))
        elif skew < 0:
            bid_price += adjustment * 0.5
            ask_price += adjustment
            bid_size *= (1 + abs(skew))
        
        self.last_trade_time = now
        
        bid_order = Order(
            order_id=self._generate_order_id(),
            symbol=order_book.symbol,
            side=OrderSide.BUY,
            price=round(bid_price, 2),
            quantity=max(bid_size, self.config.min_order_size)
        )
        
        ask_order = Order(
            order_id=self._generate_order_id(),
            symbol=order_book.symbol,
            side=OrderSide.SELL,
            price=round(ask_price, 2),
            quantity=max(ask_size, self.config.min_order_size)
        )
        
        return bid_order, ask_order


class SpreadCaptureStrategy(MarketMakingStrategy):
    """
    Spread Capture Strategy
    
    Focuses on capturing bid-ask spread by providing liquidity
    with adaptive quote placement based on fill probability.
    """
    
    def __init__(self,
                 config: MarketMakingConfig = None,
                 target_fill_rate: float = 0.3,
                 spread_capture_target: float = 0.5):
        super().__init__(config)
        self.target_fill_rate = target_fill_rate
        self.spread_capture_target = spread_capture_target
        self.fill_history: Deque[bool] = deque(maxlen=100)
        self.quote_history: Deque[Dict] = deque(maxlen=100)
    
    def estimate_fill_probability(self, 
                                   price: float,
                                   side: OrderSide,
                                   order_book: OrderBook) -> float:
        """
        Estimate probability of getting filled at a given price.
        """
        mid = order_book.mid_price
        spread = order_book.spread
        
        if side == OrderSide.BUY:
            # Distance from best bid as fraction of spread
            best_bid = order_book.best_bid.price if order_book.best_bid else mid - spread/2
            distance = (best_bid - price) / spread if spread > 0 else 0
        else:
            # Distance from best ask as fraction of spread
            best_ask = order_book.best_ask.price if order_book.best_ask else mid + spread/2
            distance = (price - best_ask) / spread if spread > 0 else 0
        
        # Simple logistic function for fill probability
        # Closer to best price = higher fill probability
        prob = 1 / (1 + np.exp(distance * 5))
        return prob
    
    def calculate_optimal_quote(self,
                                 side: OrderSide,
                                 order_book: OrderBook) -> float:
        """
        Calculate optimal quote price balancing fill probability
        and spread capture.
        """
        mid = order_book.mid_price
        spread = order_book.spread
        half_spread = spread / 2
        
        # Target: capture X% of spread while maintaining target fill rate
        target_capture = half_spread * self.spread_capture_target
        
        # Current fill rate
        if self.fill_history:
            current_fill_rate = sum(self.fill_history) / len(self.fill_history)
        else:
            current_fill_rate = 0.5
        
        # Adjust based on fill rate
        if current_fill_rate < self.target_fill_rate * 0.8:
            # Not getting enough fills, be more aggressive
            adjustment = -target_capture * 0.2
        elif current_fill_rate > self.target_fill_rate * 1.2:
            # Getting too many fills, can afford to be passive
            adjustment = target_capture * 0.2
        else:
            adjustment = 0
        
        if side == OrderSide.BUY:
            price = mid - target_capture + adjustment
        else:
            price = mid + target_capture - adjustment
        
        return price
    
    def record_quote_result(self, was_filled: bool):
        """Record whether a quote was filled."""
        self.fill_history.append(was_filled)
    
    def generate_quotes(self,
                        order_book: OrderBook,
                        **kwargs) -> Tuple[Optional[Order], Optional[Order]]:
        """
        Generate quotes optimized for spread capture.
        """
        mid_price = order_book.mid_price
        if mid_price <= 0:
            return None, None
        
        bid_price = self.calculate_optimal_quote(OrderSide.BUY, order_book)
        ask_price = self.calculate_optimal_quote(OrderSide.SELL, order_book)
        
        # Ensure minimum spread
        min_spread = mid_price * (self.config.min_spread_bps / 10000)
        if ask_price - bid_price < min_spread:
            half_min = min_spread / 2
            bid_price = mid_price - half_min
            ask_price = mid_price + half_min
        
        # Get sizes
        bid_size = self.inventory.get_adjusted_size(OrderSide.BUY)
        ask_size = self.inventory.get_adjusted_size(OrderSide.SELL)
        
        bid_order = Order(
            order_id=self._generate_order_id(),
            symbol=order_book.symbol,
            side=OrderSide.BUY,
            price=round(bid_price, 2),
            quantity=bid_size
        )
        
        ask_order = Order(
            order_id=self._generate_order_id(),
            symbol=order_book.symbol,
            side=OrderSide.SELL,
            price=round(ask_price, 2),
            quantity=ask_size
        )
        
        # Record quote for later analysis
        self.quote_history.append({
            'timestamp': datetime.now(),
            'bid_price': bid_price,
            'ask_price': ask_price,
            'mid': mid_price,
            'spread': order_book.spread
        })
        
        return bid_order, ask_order


class MarketMakingSuite:
    """
    Unified interface for market making strategies.
    """
    
    def __init__(self, config: MarketMakingConfig = None):
        self.config = config or MarketMakingConfig()
        self.strategies = {
            'basic': BasicMarketMaker(self.config),
            'imbalance': OrderBookImbalanceStrategy(self.config),
            'micro': MicroTradingStrategy(self.config),
            'spread_capture': SpreadCaptureStrategy(self.config)
        }
    
    def get_strategy(self, name: str) -> MarketMakingStrategy:
        """Get a specific strategy."""
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return self.strategies[name]
    
    def generate_quotes(self,
                        strategy_name: str,
                        order_book: OrderBook,
                        **kwargs) -> Tuple[Optional[Order], Optional[Order]]:
        """Generate quotes using specified strategy."""
        strategy = self.get_strategy(strategy_name)
        return strategy.generate_quotes(order_book, **kwargs)
    
    def simulate_market_making(self,
                                quotes: List[Quote],
                                strategy_name: str = 'basic',
                                fill_probability: float = 0.3) -> Dict:
        """
        Simulate market making on historical quote data.
        
        Args:
            quotes: List of historical quotes
            strategy_name: Strategy to use
            fill_probability: Probability of getting filled per quote
            
        Returns:
            Simulation results
        """
        strategy = self.get_strategy(strategy_name)
        np.random.seed(42)
        
        for quote in quotes:
            # Create order book from quote
            order_book = OrderBook(
                symbol=quote.symbol,
                bids=[OrderBookLevel(quote.bid_price, quote.bid_size)],
                asks=[OrderBookLevel(quote.ask_price, quote.ask_size)],
                timestamp=quote.timestamp
            )
            
            # Generate our quotes
            bid_order, ask_order = strategy.generate_quotes(order_book)
            
            if bid_order and ask_order:
                # Simulate fills
                if np.random.random() < fill_probability:
                    # Bid filled
                    fill_price = bid_order.price
                    strategy.on_fill(bid_order.order_id, bid_order.quantity, fill_price)
                
                if np.random.random() < fill_probability:
                    # Ask filled  
                    fill_price = ask_order.price
                    strategy.on_fill(ask_order.order_id, ask_order.quantity, fill_price)
        
        return {
            'total_trades': strategy.inventory.trade_count,
            'realized_pnl': strategy.inventory.realized_pnl,
            'final_position': strategy.inventory.position,
            'trade_history': strategy.trade_history[-20:]  # Last 20 trades
        }
    
    def get_performance_metrics(self, strategy_name: str) -> Dict:
        """Get performance metrics for a strategy."""
        strategy = self.get_strategy(strategy_name)
        trades = strategy.trade_history
        
        if not trades:
            return {'error': 'No trades recorded'}
        
        pnls = []
        for i in range(1, len(trades)):
            pnls.append(trades[i]['pnl'] - trades[i-1]['pnl'])
        
        if not pnls:
            return {'total_trades': len(trades), 'realized_pnl': strategy.inventory.realized_pnl}
        
        return {
            'total_trades': len(trades),
            'realized_pnl': strategy.inventory.realized_pnl,
            'avg_pnl_per_trade': np.mean(pnls),
            'pnl_std': np.std(pnls),
            'sharpe': np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 78) if np.std(pnls) > 0 else 0,
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
            'max_position': max(abs(t['position']) for t in trades),
            'avg_position': np.mean([abs(t['position']) for t in trades])
        }


# Factory function
def create_market_making_strategy(
    strategy_type: str,
    config: MarketMakingConfig = None,
    **kwargs
) -> MarketMakingStrategy:
    """
    Factory function to create market making strategies.
    
    Args:
        strategy_type: 'basic', 'imbalance', 'micro', or 'spread_capture'
        config: Strategy configuration
        **kwargs: Strategy-specific parameters
    """
    strategies = {
        'basic': BasicMarketMaker,
        'imbalance': OrderBookImbalanceStrategy,
        'micro': MicroTradingStrategy,
        'spread_capture': SpreadCaptureStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_type}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_type](config, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("=== Market Making Strategy Demo ===\n")
    
    # Create sample order book
    order_book = OrderBook(
        symbol="AAPL",
        bids=[
            OrderBookLevel(150.00, 100),
            OrderBookLevel(149.99, 200),
            OrderBookLevel(149.98, 300),
        ],
        asks=[
            OrderBookLevel(150.02, 150),
            OrderBookLevel(150.03, 250),
            OrderBookLevel(150.04, 350),
        ],
        timestamp=datetime.now()
    )
    
    print(f"Order Book: {order_book.symbol}")
    print(f"Best Bid: {order_book.best_bid.price} x {order_book.best_bid.quantity}")
    print(f"Best Ask: {order_book.best_ask.price} x {order_book.best_ask.quantity}")
    print(f"Mid Price: {order_book.mid_price}")
    print(f"Spread: {order_book.spread} ({order_book.spread/order_book.mid_price*10000:.1f} bps)")
    
    # Create suite
    suite = MarketMakingSuite()
    
    print("\n--- Basic Market Maker Quotes ---")
    bid, ask = suite.generate_quotes('basic', order_book)
    if bid and ask:
        print(f"Bid: {bid.price} x {bid.quantity}")
        print(f"Ask: {ask.price} x {ask.quantity}")
        print(f"Our Spread: {ask.price - bid.price:.2f}")
    
    print("\n--- Order Book Imbalance Analysis ---")
    imb_strategy = suite.get_strategy('imbalance')
    imbalance = imb_strategy.calculate_imbalance(order_book)
    direction = imb_strategy.predict_price_direction(order_book)
    print(f"Imbalance: {imbalance:.3f}")
    print(f"Predicted Direction: {'Up' if direction > 0 else 'Down' if direction < 0 else 'Neutral'}")
    
    bid, ask = suite.generate_quotes('imbalance', order_book)
    if bid and ask:
        print(f"Bid: {bid.price} x {bid.quantity}")
        print(f"Ask: {ask.price} x {ask.quantity}")
    
    print("\n--- Spread Capture Strategy ---")
    bid, ask = suite.generate_quotes('spread_capture', order_book)
    if bid and ask:
        print(f"Bid: {bid.price} x {bid.quantity}")
        print(f"Ask: {ask.price} x {ask.quantity}")
        print(f"Potential Capture: {(ask.price - bid.price) / 2:.4f} per side")
