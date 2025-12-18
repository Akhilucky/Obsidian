"""
Advanced Backtesting Engine
============================

Institutional-grade backtesting framework with:
- Event-driven architecture
- Multiple strategy support
- Walk-forward optimization
- Monte Carlo simulation
- Transaction cost modeling
- Slippage modeling
- Risk-adjusted metrics
- Benchmark comparison

Designed to match professional trading systems.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: float = 0
    commission: float = 0
    slippage: float = 0


@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    def update_price(self, price: float):
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_price) * self.quantity


@dataclass
class Trade:
    """Represents a completed trade."""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    holding_period: int  # days


@dataclass 
class BacktestResult:
    """Comprehensive backtest results."""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    positions_history: pd.DataFrame
    metrics: Dict[str, float]
    benchmark_comparison: Optional[Dict] = None
    drawdowns: Optional[pd.Series] = None


class TransactionCostModel:
    """Model transaction costs including commission and slippage."""
    
    def __init__(self, 
                 commission_pct: float = 0.001,  # 0.1% = 10 bps
                 commission_min: float = 1.0,
                 slippage_pct: float = 0.0005,   # 0.05% = 5 bps
                 spread_pct: float = 0.0001):    # 0.01% = 1 bp
        self.commission_pct = commission_pct
        self.commission_min = commission_min
        self.slippage_pct = slippage_pct
        self.spread_pct = spread_pct
    
    def calculate_commission(self, order_value: float) -> float:
        """Calculate commission for an order."""
        commission = order_value * self.commission_pct
        return max(commission, self.commission_min)
    
    def calculate_slippage(self, price: float, side: OrderSide, 
                          volume: float = None) -> float:
        """Calculate slippage."""
        # Base slippage
        slippage = price * self.slippage_pct
        
        # Add spread
        slippage += price * self.spread_pct / 2
        
        # Adverse slippage for buys is positive, sells is negative
        if side == OrderSide.BUY:
            return slippage
        else:
            return -slippage
    
    def get_execution_price(self, price: float, side: OrderSide) -> float:
        """Get execution price including slippage."""
        slippage = self.calculate_slippage(price, side)
        return price + slippage


class Strategy(ABC):
    """Base class for trading strategies."""
    
    def __init__(self, name: str = "Strategy"):
        self.name = name
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity = 0
        
    @abstractmethod
    def on_data(self, timestamp: datetime, data: Dict[str, pd.Series]) -> List[Order]:
        """
        Called on each bar of data.
        
        Args:
            timestamp: Current timestamp
            data: Dict of symbol -> OHLCV series
        
        Returns:
            List of orders to execute
        """
        pass
    
    def on_order_filled(self, order: Order):
        """Called when an order is filled."""
        pass
    
    def on_trade_closed(self, trade: Trade):
        """Called when a trade is closed."""
        pass


class MomentumStrategy(Strategy):
    """
    Example momentum strategy.
    Buys when price is above moving average, sells when below.
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 position_size: float = 0.1):
        super().__init__(name="Momentum")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        self.price_history = defaultdict(list)
    
    def on_data(self, timestamp: datetime, data: Dict[str, pd.Series]) -> List[Order]:
        orders = []
        
        for symbol, ohlcv in data.items():
            price = ohlcv['Close']
            self.price_history[symbol].append(price)
            
            if len(self.price_history[symbol]) < self.slow_period:
                continue
            
            prices = np.array(self.price_history[symbol][-self.slow_period:])
            fast_ma = np.mean(prices[-self.fast_period:])
            slow_ma = np.mean(prices)
            
            has_position = symbol in self.positions and self.positions[symbol].quantity > 0
            
            if fast_ma > slow_ma and not has_position:
                # Buy signal
                quantity = self.equity * self.position_size / price
                orders.append(Order(
                    id=f"{symbol}_{timestamp.timestamp()}",
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    timestamp=timestamp
                ))
            elif fast_ma < slow_ma and has_position:
                # Sell signal
                orders.append(Order(
                    id=f"{symbol}_{timestamp.timestamp()}",
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=self.positions[symbol].quantity,
                    timestamp=timestamp
                ))
        
        return orders


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy.
    Buys when price is oversold, sells when overbought.
    """
    
    def __init__(self, lookback: int = 20, entry_zscore: float = 2.0,
                 exit_zscore: float = 0.5, position_size: float = 0.1):
        super().__init__(name="MeanReversion")
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.position_size = position_size
        self.price_history = defaultdict(list)
    
    def on_data(self, timestamp: datetime, data: Dict[str, pd.Series]) -> List[Order]:
        orders = []
        
        for symbol, ohlcv in data.items():
            price = ohlcv['Close']
            self.price_history[symbol].append(price)
            
            if len(self.price_history[symbol]) < self.lookback:
                continue
            
            prices = np.array(self.price_history[symbol][-self.lookback:])
            mean = np.mean(prices)
            std = np.std(prices)
            
            if std == 0:
                continue
            
            zscore = (price - mean) / std
            has_position = symbol in self.positions and self.positions[symbol].quantity > 0
            
            if zscore < -self.entry_zscore and not has_position:
                # Oversold - buy
                quantity = self.equity * self.position_size / price
                orders.append(Order(
                    id=f"{symbol}_{timestamp.timestamp()}",
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    timestamp=timestamp
                ))
            elif has_position:
                position_side = 1 if self.positions[symbol].quantity > 0 else -1
                
                if position_side > 0 and zscore > self.exit_zscore:
                    # Exit long
                    orders.append(Order(
                        id=f"{symbol}_{timestamp.timestamp()}",
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=abs(self.positions[symbol].quantity),
                        timestamp=timestamp
                    ))
        
        return orders


class BacktestEngine:
    """
    Event-driven backtesting engine.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 cost_model: TransactionCostModel = None):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.positions_history = []
        
        # Tracking
        self.order_counter = 0
        self.trade_counter = 0
    
    def run(self, strategy: Strategy, data: Dict[str, pd.DataFrame],
            benchmark: pd.Series = None) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            strategy: Strategy to test
            data: Dict of symbol -> OHLCV DataFrame
            benchmark: Optional benchmark returns for comparison
        
        Returns:
            BacktestResult
        """
        # Align data by timestamp
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        
        # Initialize strategy
        strategy.positions = self.positions
        strategy.equity = self.initial_capital
        
        # Run simulation
        for timestamp in timestamps:
            # Get data for this bar
            bar_data = {}
            for symbol, df in data.items():
                if timestamp in df.index:
                    bar_data[symbol] = df.loc[timestamp]
            
            if not bar_data:
                continue
            
            # Update positions with current prices
            self._update_positions(bar_data)
            
            # Get orders from strategy
            orders = strategy.on_data(timestamp, bar_data)
            
            # Execute orders
            for order in orders:
                self._execute_order(order, bar_data)
            
            # Record equity
            equity = self._calculate_equity()
            strategy.equity = equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': self.cash
            })
            
            # Record positions
            pos_record = {'timestamp': timestamp}
            for symbol, pos in self.positions.items():
                pos_record[f'{symbol}_qty'] = pos.quantity
                pos_record[f'{symbol}_value'] = pos.market_value
            self.positions_history.append(pos_record)
        
        # Close all positions at end
        self._close_all_positions(timestamps[-1], data)
        
        # Calculate results
        result = self._calculate_results(benchmark)
        
        return result
    
    def _update_positions(self, bar_data: Dict[str, pd.Series]):
        """Update position prices."""
        for symbol, pos in self.positions.items():
            if symbol in bar_data:
                pos.update_price(bar_data[symbol]['Close'])
    
    def _execute_order(self, order: Order, bar_data: Dict[str, pd.Series]):
        """Execute an order."""
        if order.symbol not in bar_data:
            order.status = OrderStatus.REJECTED
            return
        
        current_price = bar_data[order.symbol]['Close']
        
        # Calculate execution price with slippage
        exec_price = self.cost_model.get_execution_price(current_price, order.side)
        
        # Calculate order value and commission
        order_value = exec_price * order.quantity
        commission = self.cost_model.calculate_commission(order_value)
        
        # Check if we have enough capital for buys
        if order.side == OrderSide.BUY:
            if order_value + commission > self.cash:
                order.status = OrderStatus.REJECTED
                return
            
            self.cash -= (order_value + commission)
            
            # Update or create position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                new_qty = pos.quantity + order.quantity
                pos.avg_price = (pos.avg_price * pos.quantity + exec_price * order.quantity) / new_qty
                pos.quantity = new_qty
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=exec_price,
                    current_price=exec_price
                )
        
        else:  # SELL
            if order.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                return
            
            pos = self.positions[order.symbol]
            sell_qty = min(order.quantity, pos.quantity)
            
            # Calculate PnL
            pnl = (exec_price - pos.avg_price) * sell_qty - commission
            pos.realized_pnl += pnl
            
            self.cash += (exec_price * sell_qty - commission)
            
            # Update position
            pos.quantity -= sell_qty
            if pos.quantity <= 0:
                # Record trade
                self.trade_counter += 1
                self.trades.append(Trade(
                    id=f"trade_{self.trade_counter}",
                    symbol=order.symbol,
                    side=OrderSide.BUY,  # Original side
                    quantity=sell_qty,
                    entry_price=pos.avg_price,
                    exit_price=exec_price,
                    entry_time=order.timestamp - timedelta(days=1),  # Approximate
                    exit_time=order.timestamp,
                    pnl=pnl,
                    pnl_pct=(exec_price / pos.avg_price - 1) * 100,
                    commission=commission,
                    slippage=exec_price - current_price,
                    holding_period=1  # Approximate
                ))
                del self.positions[order.symbol]
        
        order.status = OrderStatus.FILLED
        order.filled_price = exec_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.slippage = exec_price - current_price
        
        self.orders.append(order)
    
    def _close_all_positions(self, timestamp: datetime, data: Dict[str, pd.DataFrame]):
        """Close all remaining positions."""
        for symbol, pos in list(self.positions.items()):
            if pos.quantity > 0:
                bar_data = {symbol: data[symbol].iloc[-1]}
                order = Order(
                    id=f"close_{symbol}",
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    timestamp=timestamp
                )
                self._execute_order(order, bar_data)
    
    def _calculate_equity(self) -> float:
        """Calculate total equity."""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    def _calculate_results(self, benchmark: pd.Series = None) -> BacktestResult:
        """Calculate comprehensive results."""
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        equity_series = equity_df['equity']
        returns = equity_series.pct_change().dropna()
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_series, returns)
        
        # Drawdowns
        cummax = equity_series.cummax()
        drawdowns = (equity_series - cummax) / cummax
        
        # Benchmark comparison
        benchmark_comparison = None
        if benchmark is not None:
            benchmark_comparison = self._compare_benchmark(returns, benchmark)
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_series,
            returns=returns,
            positions_history=pd.DataFrame(self.positions_history),
            metrics=metrics,
            benchmark_comparison=benchmark_comparison,
            drawdowns=drawdowns
        )
    
    def _calculate_metrics(self, equity: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        
        # Annualized metrics
        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        annual_vol = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        risk_free = 0.03  # 3% annual
        excess_return = annual_return - risk_free * 100
        sharpe = excess_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = excess_return / (downside * 100) if downside > 0 else 0
        
        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) \
                        if losing_trades and sum(t.pnl for t in losing_trades) != 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commission': sum(o.commission for o in self.orders),
            'total_slippage': sum(abs(o.slippage) for o in self.orders),
            'final_equity': equity.iloc[-1],
            'initial_capital': self.initial_capital
        }
    
    def _compare_benchmark(self, returns: pd.Series, 
                           benchmark: pd.Series) -> Dict:
        """Compare strategy to benchmark."""
        # Align series
        aligned = pd.concat([returns, benchmark], axis=1, join='inner')
        aligned.columns = ['strategy', 'benchmark']
        
        # Calculate metrics
        strat_total = (1 + aligned['strategy']).cumprod().iloc[-1] - 1
        bench_total = (1 + aligned['benchmark']).cumprod().iloc[-1] - 1
        
        # Alpha and Beta
        covariance = np.cov(aligned['strategy'], aligned['benchmark'])[0, 1]
        bench_var = np.var(aligned['benchmark'])
        beta = covariance / bench_var if bench_var > 0 else 1
        
        alpha = (aligned['strategy'].mean() - beta * aligned['benchmark'].mean()) * 252
        
        # Information ratio
        tracking_error = (aligned['strategy'] - aligned['benchmark']).std() * np.sqrt(252)
        info_ratio = (strat_total - bench_total) / tracking_error if tracking_error > 0 else 0
        
        return {
            'strategy_return': strat_total * 100,
            'benchmark_return': bench_total * 100,
            'excess_return': (strat_total - bench_total) * 100,
            'alpha': alpha * 100,
            'beta': beta,
            'tracking_error': tracking_error * 100,
            'information_ratio': info_ratio
        }


class WalkForwardOptimizer:
    """
    Walk-forward optimization for strategy parameters.
    """
    
    def __init__(self, strategy_class: type, param_grid: Dict,
                 train_period: int = 252, test_period: int = 63):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.train_period = train_period
        self.test_period = test_period
    
    def optimize(self, data: Dict[str, pd.DataFrame],
                 metric: str = 'sharpe_ratio') -> Dict:
        """
        Run walk-forward optimization.
        
        Args:
            data: Historical data
            metric: Metric to optimize
        
        Returns:
            Optimization results
        """
        results = []
        
        # Generate all parameter combinations
        param_combos = self._generate_param_combos()
        
        # Get all timestamps
        all_timestamps = sorted(set().union(*[set(df.index) for df in data.values()]))
        
        # Walk forward through time
        current_idx = self.train_period
        
        while current_idx + self.test_period < len(all_timestamps):
            train_start = current_idx - self.train_period
            train_end = current_idx
            test_end = current_idx + self.test_period
            
            train_timestamps = all_timestamps[train_start:train_end]
            test_timestamps = all_timestamps[train_end:test_end]
            
            # Split data
            train_data = {s: df.loc[df.index.isin(train_timestamps)] 
                         for s, df in data.items()}
            test_data = {s: df.loc[df.index.isin(test_timestamps)] 
                        for s, df in data.items()}
            
            # Find best params on training data
            best_params, best_score = self._optimize_params(
                train_data, param_combos, metric
            )
            
            # Test on out-of-sample data
            test_result = self._run_backtest(test_data, best_params)
            
            results.append({
                'train_start': train_timestamps[0],
                'train_end': train_timestamps[-1],
                'test_start': test_timestamps[0],
                'test_end': test_timestamps[-1],
                'best_params': best_params,
                'train_score': best_score,
                'test_score': test_result.metrics.get(metric, 0),
                'test_return': test_result.metrics.get('total_return', 0)
            })
            
            current_idx += self.test_period
        
        return {
            'walk_forward_results': results,
            'aggregate_metrics': self._aggregate_results(results)
        }
    
    def _generate_param_combos(self) -> List[Dict]:
        """Generate all parameter combinations."""
        import itertools
        
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))
        
        return combos
    
    def _optimize_params(self, data: Dict[str, pd.DataFrame],
                         param_combos: List[Dict],
                         metric: str) -> Tuple[Dict, float]:
        """Find best parameters on training data."""
        best_params = None
        best_score = -float('inf')
        
        for params in param_combos:
            result = self._run_backtest(data, params)
            score = result.metrics.get(metric, 0)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    def _run_backtest(self, data: Dict[str, pd.DataFrame],
                      params: Dict) -> BacktestResult:
        """Run backtest with given parameters."""
        strategy = self.strategy_class(**params)
        engine = BacktestEngine()
        return engine.run(strategy, data)
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward results."""
        test_returns = [r['test_return'] for r in results]
        
        return {
            'avg_test_return': np.mean(test_returns),
            'median_test_return': np.median(test_returns),
            'std_test_return': np.std(test_returns),
            'min_test_return': np.min(test_returns),
            'max_test_return': np.max(test_returns),
            'positive_periods': sum(1 for r in test_returns if r > 0),
            'total_periods': len(test_returns)
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for backtest results.
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
    
    def simulate(self, returns: pd.Series, 
                 initial_capital: float = 100000) -> Dict:
        """
        Run Monte Carlo simulation.
        
        Args:
            returns: Historical returns
            initial_capital: Starting capital
        
        Returns:
            Simulation results
        """
        n_days = len(returns)
        
        # Generate simulated return paths
        simulated_paths = []
        final_values = []
        max_drawdowns = []
        
        for _ in range(self.n_simulations):
            # Bootstrap returns (random sampling with replacement)
            sim_returns = np.random.choice(returns.values, size=n_days, replace=True)
            
            # Calculate equity curve
            equity = initial_capital * np.cumprod(1 + sim_returns)
            simulated_paths.append(equity)
            final_values.append(equity[-1])
            
            # Calculate max drawdown
            cummax = np.maximum.accumulate(equity)
            drawdown = (equity - cummax) / cummax
            max_drawdowns.append(np.min(drawdown))
        
        # Calculate statistics
        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'probability_profit': np.mean(final_values > initial_capital) * 100,
            'probability_double': np.mean(final_values > initial_capital * 2) * 100,
            'mean_max_drawdown': np.mean(max_drawdowns) * 100,
            'worst_max_drawdown': np.min(max_drawdowns) * 100,
            'var_95': initial_capital - np.percentile(final_values, 5),
            'cvar_95': initial_capital - np.mean(final_values[final_values <= np.percentile(final_values, 5)])
        }


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report."""
    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)
    
    m = result.metrics
    
    print(f"\n--- Performance ---")
    print(f"Total Return: {m['total_return']:.2f}%")
    print(f"Annual Return: {m['annual_return']:.2f}%")
    print(f"Annual Volatility: {m['annual_volatility']:.2f}%")
    print(f"Sharpe Ratio: {m['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {m['sortino_ratio']:.3f}")
    print(f"Calmar Ratio: {m['calmar_ratio']:.3f}")
    print(f"Max Drawdown: {m['max_drawdown']:.2f}%")
    
    print(f"\n--- Trade Statistics ---")
    print(f"Total Trades: {m['total_trades']}")
    print(f"Win Rate: {m['win_rate']:.1f}%")
    print(f"Winning Trades: {m['winning_trades']}")
    print(f"Losing Trades: {m['losing_trades']}")
    print(f"Average Win: ${m['avg_win']:.2f}")
    print(f"Average Loss: ${m['avg_loss']:.2f}")
    print(f"Profit Factor: {m['profit_factor']:.2f}")
    
    print(f"\n--- Costs ---")
    print(f"Total Commission: ${m['total_commission']:.2f}")
    print(f"Total Slippage: ${m['total_slippage']:.2f}")
    
    print(f"\n--- Capital ---")
    print(f"Initial Capital: ${m['initial_capital']:,.2f}")
    print(f"Final Equity: ${m['final_equity']:,.2f}")
    
    if result.benchmark_comparison:
        print(f"\n--- Benchmark Comparison ---")
        b = result.benchmark_comparison
        print(f"Strategy Return: {b['strategy_return']:.2f}%")
        print(f"Benchmark Return: {b['benchmark_return']:.2f}%")
        print(f"Excess Return: {b['excess_return']:.2f}%")
        print(f"Alpha: {b['alpha']:.2f}%")
        print(f"Beta: {b['beta']:.3f}")
        print(f"Information Ratio: {b['information_ratio']:.3f}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Backtesting Engine")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_days = 500
    
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    # Simulate price data for multiple stocks
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    data = {}
    
    for symbol in symbols:
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = 100 * np.cumprod(1 + returns)
        
        data[symbol] = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'High': prices * (1 + abs(np.random.normal(0, 0.01, n_days))),
            'Low': prices * (1 - abs(np.random.normal(0, 0.01, n_days))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
    
    # Create benchmark
    benchmark_returns = pd.Series(
        np.random.normal(0.0003, 0.015, n_days),
        index=dates
    )
    
    # Test Momentum Strategy
    print("\n--- Running Momentum Strategy Backtest ---")
    strategy = MomentumStrategy(fast_period=10, slow_period=30, position_size=0.2)
    
    cost_model = TransactionCostModel(
        commission_pct=0.001,
        slippage_pct=0.0005
    )
    
    engine = BacktestEngine(initial_capital=100000, cost_model=cost_model)
    result = engine.run(strategy, data, benchmark=benchmark_returns)
    
    print_backtest_report(result)
    
    # Monte Carlo simulation
    print("\n--- Monte Carlo Simulation ---")
    mc = MonteCarloSimulator(n_simulations=1000)
    mc_result = mc.simulate(result.returns, initial_capital=100000)
    
    print(f"Mean Final Value: ${mc_result['mean_final_value']:,.2f}")
    print(f"95% Confidence Interval: ${mc_result['percentile_5']:,.2f} - ${mc_result['percentile_95']:,.2f}")
    print(f"Probability of Profit: {mc_result['probability_profit']:.1f}%")
    print(f"Value at Risk (95%): ${mc_result['var_95']:,.2f}")
    
    print("\n" + "=" * 60)
    print("Backtesting Complete!")
    print("=" * 60)
