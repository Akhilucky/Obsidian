"""
Paper Trading Simulator
========================
A realistic paper trading environment with simulated order execution,
portfolio tracking, and performance analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import threading
import time
import warnings
warnings.filterwarnings('ignore')


class OrderType(Enum):
    """Types of orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order direction"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionType(Enum):
    """Position type"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    trail_percent: Optional[float] = None  # For trailing stops
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    commission: float = 0
    slippage: float = 0
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'commission': self.commission
        }


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    quantity: float
    entry_price: float
    position_type: PositionType
    opened_at: datetime = field(default_factory=datetime.now)
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    current_price: float = 0
    
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.entry_price
    
    @property
    def pnl_percent(self) -> float:
        if self.cost_basis == 0:
            return 0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class Trade:
    """Represents a completed trade"""
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    hold_time: timedelta
    commission: float


@dataclass
class AccountState:
    """Current account state"""
    cash: float
    equity: float
    buying_power: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float = 0
    margin_available: float = 0


class MarketSimulator:
    """
    Simulates realistic market conditions for paper trading
    
    Features:
    - Configurable slippage
    - Volume-based fill simulation
    - Market hours enforcement
    - Order book simulation
    """
    
    def __init__(self, slippage_model: str = 'fixed', 
                 slippage_bps: float = 5,
                 commission_per_share: float = 0.005,
                 min_commission: float = 1.0):
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.market_hours = (9, 30, 16, 0)  # Start H, M, End H, M
        self.current_prices: Dict[str, float] = {}
    
    def update_price(self, symbol: str, price: float):
        """Update current market price for a symbol"""
        self.current_prices[symbol] = price
    
    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        return self.current_prices.get(symbol, 0)
    
    def calculate_slippage(self, symbol: str, side: OrderSide, 
                          quantity: float) -> float:
        """Calculate slippage for an order"""
        base_price = self.current_prices.get(symbol, 0)
        
        if self.slippage_model == 'fixed':
            slippage = base_price * (self.slippage_bps / 10000)
        elif self.slippage_model == 'volume':
            # More slippage for larger orders
            slippage = base_price * (self.slippage_bps / 10000) * (1 + quantity / 10000)
        elif self.slippage_model == 'random':
            slippage = base_price * (np.random.uniform(0, self.slippage_bps * 2) / 10000)
        else:
            slippage = 0
        
        # Apply direction
        if side == OrderSide.BUY:
            return slippage
        else:
            return -slippage
    
    def calculate_commission(self, quantity: float) -> float:
        """Calculate commission for a trade"""
        commission = abs(quantity) * self.commission_per_share
        return max(commission, self.min_commission)
    
    def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if market is open"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check weekend
        if timestamp.weekday() >= 5:
            return False
        
        # Check hours
        start_h, start_m, end_h, end_m = self.market_hours
        market_open = timestamp.replace(hour=start_h, minute=start_m, second=0)
        market_close = timestamp.replace(hour=end_h, minute=end_m, second=0)
        
        return market_open <= timestamp <= market_close
    
    def simulate_fill(self, order: Order, current_price: float) -> Tuple[bool, float, float]:
        """
        Simulate order fill
        
        Returns: (filled, fill_price, slippage)
        """
        slippage = self.calculate_slippage(order.symbol, order.side, order.quantity)
        
        if order.order_type == OrderType.MARKET:
            fill_price = current_price + slippage
            return True, fill_price, slippage
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                fill_price = min(current_price + slippage, order.price)
                return True, fill_price, slippage
            elif order.side == OrderSide.SELL and current_price >= order.price:
                fill_price = max(current_price + slippage, order.price)
                return True, fill_price, slippage
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.SELL and current_price <= order.stop_price:
                fill_price = current_price + slippage
                return True, fill_price, slippage
            elif order.side == OrderSide.BUY and current_price >= order.stop_price:
                fill_price = current_price + slippage
                return True, fill_price, slippage
        
        return False, 0, 0


class PaperTradingAccount:
    """
    Paper Trading Account
    ======================
    Simulates a real trading account with:
    - Order management
    - Position tracking
    - P&L calculation
    - Risk management
    """
    
    def __init__(self, initial_capital: float = 100000,
                 margin_rate: float = 0.5,
                 max_position_size: float = 0.25):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.margin_rate = margin_rate
        self.max_position_size = max_position_size
        
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.order_history: List[Order] = []
        
        self.market = MarketSimulator()
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Callbacks
        self.on_fill: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
    
    @property
    def equity(self) -> float:
        """Total account equity"""
        positions_value = sum(
            p.quantity * self.market.get_price(p.symbol)
            for p in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def buying_power(self) -> float:
        """Available buying power with margin"""
        return self.cash / self.margin_rate
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Total realized P&L"""
        return sum(t.pnl for t in self.trades)
    
    def get_account_state(self) -> AccountState:
        """Get current account state"""
        positions_value = sum(p.market_value for p in self.positions.values())
        
        return AccountState(
            cash=self.cash,
            equity=self.equity,
            buying_power=self.buying_power,
            positions_value=positions_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl
        )
    
    def submit_order(self, symbol: str, side: OrderSide, quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    trail_percent: Optional[float] = None) -> Order:
        """
        Submit a new order
        
        Parameters:
        -----------
        symbol : Stock symbol
        side : BUY or SELL
        quantity : Number of shares
        order_type : MARKET, LIMIT, STOP, etc.
        price : Limit price (for limit orders)
        stop_price : Stop trigger price
        trail_percent : Trailing stop percentage
        """
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            trail_percent=trail_percent,
            status=OrderStatus.OPEN
        )
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            return order
        
        self.orders[order.order_id] = order
        
        # Try immediate fill for market orders
        if order_type == OrderType.MARKET:
            self._process_order(order)
        
        return order
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission"""
        current_price = self.market.get_price(order.symbol)
        if current_price == 0:
            return False
        
        order_value = order.quantity * current_price
        
        # Check buying power for buys
        if order.side == OrderSide.BUY:
            if order_value > self.buying_power:
                return False
            
            # Check max position size
            if order_value > self.equity * self.max_position_size:
                return False
        
        # Check position for sells
        elif order.side == OrderSide.SELL:
            position = self.positions.get(order.symbol)
            if position is None or position.quantity < order.quantity:
                # Allow short selling (simplified)
                pass
        
        return True
    
    def _process_order(self, order: Order):
        """Process and potentially fill an order"""
        current_price = self.market.get_price(order.symbol)
        if current_price == 0:
            return
        
        filled, fill_price, slippage = self.market.simulate_fill(order, current_price)
        
        if filled:
            self._execute_fill(order, fill_price, slippage)
    
    def _execute_fill(self, order: Order, fill_price: float, slippage: float):
        """Execute order fill and update positions"""
        commission = self.market.calculate_commission(order.quantity)
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.filled_at = datetime.now()
        order.commission = commission
        order.slippage = slippage
        
        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= (fill_price * order.quantity + commission)
        else:
            self.cash += (fill_price * order.quantity - commission)
        
        # Update positions
        self._update_position(order)
        
        # Move to history
        self.order_history.append(order)
        if order.order_id in self.orders:
            del self.orders[order.order_id]
        
        # Callback
        if self.on_fill:
            self.on_fill(order)
    
    def _update_position(self, order: Order):
        """Update position after fill"""
        symbol = order.symbol
        
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if order.side == OrderSide.BUY:
                # Adding to long or covering short
                if position.quantity > 0:
                    # Adding to long
                    total_cost = (position.quantity * position.entry_price + 
                                 order.filled_quantity * order.filled_price)
                    position.quantity += order.filled_quantity
                    position.entry_price = total_cost / position.quantity
                else:
                    # Covering short
                    position.quantity += order.filled_quantity
                    if position.quantity >= 0:
                        # Record trade
                        self._record_trade(position, order)
                        if position.quantity == 0:
                            del self.positions[symbol]
                        else:
                            position.position_type = PositionType.LONG
                            position.entry_price = order.filled_price
            
            else:  # SELL
                if position.quantity > 0:
                    # Reducing long
                    position.quantity -= order.filled_quantity
                    if position.quantity <= 0:
                        self._record_trade(position, order)
                        if position.quantity == 0:
                            del self.positions[symbol]
                        else:
                            position.position_type = PositionType.SHORT
                            position.entry_price = order.filled_price
                else:
                    # Adding to short
                    total_cost = (abs(position.quantity) * position.entry_price + 
                                 order.filled_quantity * order.filled_price)
                    position.quantity -= order.filled_quantity
                    position.entry_price = total_cost / abs(position.quantity)
        else:
            # New position
            position_type = PositionType.LONG if order.side == OrderSide.BUY else PositionType.SHORT
            quantity = order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity
            
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=order.filled_price,
                position_type=position_type,
                current_price=order.filled_price
            )
        
        # Callback
        if self.on_position_update and symbol in self.positions:
            self.on_position_update(self.positions[symbol])
    
    def _record_trade(self, position: Position, exit_order: Order):
        """Record a completed trade"""
        pnl = (exit_order.filled_price - position.entry_price) * exit_order.filled_quantity
        if position.position_type == PositionType.SHORT:
            pnl = -pnl
        
        pnl -= exit_order.commission
        pnl_percent = (pnl / (position.entry_price * exit_order.filled_quantity)) * 100
        
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=position.symbol,
            side=OrderSide.BUY if position.position_type == PositionType.LONG else OrderSide.SELL,
            quantity=exit_order.filled_quantity,
            entry_price=position.entry_price,
            exit_price=exit_order.filled_price,
            entry_time=position.opened_at,
            exit_time=exit_order.filled_at,
            pnl=pnl,
            pnl_percent=pnl_percent,
            hold_time=exit_order.filled_at - position.opened_at,
            commission=exit_order.commission
        )
        
        self.trades.append(trade)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = OrderStatus.CANCELLED
            self.order_history.append(order)
            del self.orders[order_id]
            return True
        return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all open orders, optionally for a specific symbol"""
        to_cancel = []
        for order_id, order in self.orders.items():
            if symbol is None or order.symbol == symbol:
                to_cancel.append(order_id)
        
        for order_id in to_cancel:
            self.cancel_order(order_id)
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close an entire position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        
        return self.submit_order(
            symbol=symbol,
            side=side,
            quantity=abs(position.quantity),
            order_type=OrderType.MARKET
        )
    
    def close_all_positions(self):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            self.close_position(symbol)
    
    def update_prices(self, prices: Dict[str, float]):
        """Update market prices and recalculate P&L"""
        for symbol, price in prices.items():
            self.market.update_price(symbol, price)
            
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                position.unrealized_pnl = (price - position.entry_price) * position.quantity
        
        # Process pending orders
        for order in list(self.orders.values()):
            if order.symbol in prices:
                self._process_order(order)
        
        # Record equity
        self.equity_curve.append((datetime.now(), self.equity))
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'trade_id': t.trade_id,
            'symbol': t.symbol,
            'side': t.side.value,
            'quantity': t.quantity,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'hold_time': str(t.hold_time),
            'entry_time': t.entry_time,
            'exit_time': t.exit_time
        } for t in self.trades])
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Calculate performance statistics"""
        if not self.trades:
            return {}
        
        pnls = [t.pnl for t in self.trades]
        pnl_percents = [t.pnl_percent for t in self.trades]
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        
        # Calculate max drawdown from equity curve
        if self.equity_curve:
            equities = [e[1] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0
            for equity in equities:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(self.trades) if self.trades else 0,
            'total_pnl': sum(pnls),
            'average_pnl': np.mean(pnls),
            'average_win': np.mean(wins) if wins else 0,
            'average_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else float('inf'),
            'largest_win': max(pnls) if pnls else 0,
            'largest_loss': min(pnls) if pnls else 0,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'sharpe_ratio': self._calculate_sharpe(pnl_percents),
            'sortino_ratio': self._calculate_sortino(pnl_percents)
        }
    
    def _calculate_sharpe(self, returns: List[float], risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = np.array(returns) - (risk_free / 252)
        if np.std(excess_returns) == 0:
            return 0
        
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_sortino(self, returns: List[float], risk_free: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = np.array(returns) - (risk_free / 252)
        downside = excess_returns[excess_returns < 0]
        
        if len(downside) == 0 or np.std(downside) == 0:
            return 0
        
        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside)
    
    def reset(self):
        """Reset account to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.order_history = []
        self.equity_curve = []


class PaperTradingSimulator:
    """
    Full Paper Trading Simulation Environment
    ==========================================
    Runs paper trading with historical or live data.
    
    Features:
    - Historical simulation mode
    - Real-time simulation mode
    - Strategy integration
    - Performance tracking
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.account = PaperTradingAccount(initial_capital)
        self.is_running = False
        self.simulation_speed = 1.0  # 1x = real-time
        self._thread = None
    
    def run_historical(self, data: Dict[str, pd.DataFrame],
                      strategy: Callable[[Dict[str, pd.DataFrame], 'PaperTradingAccount'], None],
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run simulation on historical data
        
        Parameters:
        -----------
        data : Dictionary mapping symbols to OHLCV DataFrames
        strategy : Function that takes (current_data, account) and places orders
        """
        self.account.reset()
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in data.values():
            if 'date' in df.columns:
                all_timestamps.update(df['date'].tolist())
            elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                all_timestamps.update(df.index.tolist())
        
        timestamps = sorted(all_timestamps)
        
        if start_date:
            timestamps = [t for t in timestamps if t >= start_date]
        if end_date:
            timestamps = [t for t in timestamps if t <= end_date]
        
        # Run simulation
        for timestamp in timestamps:
            # Get current prices
            current_prices = {}
            current_data = {}
            
            for symbol, df in data.items():
                if 'date' in df.columns:
                    row = df[df['date'] <= timestamp].iloc[-1] if len(df[df['date'] <= timestamp]) > 0 else None
                else:
                    row = df[df.index <= timestamp].iloc[-1] if len(df[df.index <= timestamp]) > 0 else None
                
                if row is not None:
                    current_prices[symbol] = row['close']
                    current_data[symbol] = df[df.index <= timestamp] if 'date' not in df.columns else df[df['date'] <= timestamp]
            
            # Update account with prices
            self.account.update_prices(current_prices)
            
            # Call strategy
            try:
                strategy(current_data, self.account)
            except Exception as e:
                print(f"Strategy error at {timestamp}: {e}")
        
        return {
            'final_equity': self.account.equity,
            'total_return': (self.account.equity - self.account.initial_capital) / self.account.initial_capital,
            'stats': self.account.get_performance_stats(),
            'trades': self.account.get_trade_history(),
            'equity_curve': self.account.equity_curve
        }
    
    def start_live(self, symbols: List[str],
                  strategy: Callable[[Dict[str, float], 'PaperTradingAccount'], None],
                  price_feed: Callable[[], Dict[str, float]],
                  interval_seconds: float = 1.0):
        """
        Start live paper trading simulation
        
        Parameters:
        -----------
        symbols : List of symbols to trade
        strategy : Function that receives current prices and account
        price_feed : Function that returns current prices for symbols
        interval_seconds : Update interval
        """
        self.is_running = True
        
        def _run():
            while self.is_running:
                try:
                    prices = price_feed()
                    self.account.update_prices(prices)
                    strategy(prices, self.account)
                except Exception as e:
                    print(f"Live simulation error: {e}")
                
                time.sleep(interval_seconds / self.simulation_speed)
        
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
    
    def stop_live(self):
        """Stop live simulation"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def get_results(self) -> Dict[str, Any]:
        """Get current simulation results"""
        return {
            'account_state': self.account.get_account_state().__dict__,
            'positions': {s: p.__dict__ for s, p in self.account.positions.items()},
            'stats': self.account.get_performance_stats(),
            'trades': len(self.account.trades),
            'equity': self.account.equity
        }


# ============================================================================
# EXAMPLE STRATEGIES FOR PAPER TRADING
# ============================================================================

def simple_moving_average_strategy(data: Dict[str, pd.DataFrame], 
                                   account: PaperTradingAccount):
    """Simple SMA crossover strategy for paper trading"""
    for symbol, df in data.items():
        if len(df) < 50:
            continue
        
        # Calculate SMAs
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        prev_sma_20 = df['close'].rolling(20).mean().iloc[-2]
        prev_sma_50 = df['close'].rolling(50).mean().iloc[-2]
        current_price = df['close'].iloc[-1]
        
        position = account.positions.get(symbol)
        
        # Buy signal: SMA20 crosses above SMA50
        if prev_sma_20 <= prev_sma_50 and sma_20 > sma_50:
            if position is None:
                # Size: 10% of equity
                size = (account.equity * 0.1) / current_price
                account.submit_order(symbol, OrderSide.BUY, int(size))
        
        # Sell signal: SMA20 crosses below SMA50
        elif prev_sma_20 >= prev_sma_50 and sma_20 < sma_50:
            if position and position.quantity > 0:
                account.close_position(symbol)


def rsi_mean_reversion_strategy(data: Dict[str, pd.DataFrame],
                                account: PaperTradingAccount):
    """RSI mean reversion strategy"""
    for symbol, df in data.items():
        if len(df) < 20:
            continue
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        current_price = df['close'].iloc[-1]
        
        position = account.positions.get(symbol)
        
        # Buy when oversold
        if current_rsi < 30 and position is None:
            size = (account.equity * 0.1) / current_price
            account.submit_order(symbol, OrderSide.BUY, int(size))
        
        # Sell when overbought
        elif current_rsi > 70 and position and position.quantity > 0:
            account.close_position(symbol)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PAPER TRADING SIMULATOR DEMO")
    print("=" * 60)
    
    # Create simulator
    sim = PaperTradingSimulator(initial_capital=100000)
    
    # Generate sample historical data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    
    data = {}
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        price = 100 + np.random.randn(252).cumsum()
        data[symbol] = pd.DataFrame({
            'open': price + np.random.randn(252) * 0.5,
            'high': price + abs(np.random.randn(252)) * 2,
            'low': price - abs(np.random.randn(252)) * 2,
            'close': price,
            'volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)
    
    print("\nRunning historical simulation...")
    
    results = sim.run_historical(data, simple_moving_average_strategy)
    
    print(f"\nFinal Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"\nPerformance Stats:")
    
    stats = results['stats']
    print(f"  Total Trades: {stats.get('total_trades', 0)}")
    print(f"  Win Rate: {stats.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown: {stats.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
