"""
Performance Analytics Dashboard
================================
Advanced performance metrics, visualizations, and strategy comparison tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a strategy"""
    # Returns
    total_return: float
    annualized_return: float
    daily_return_mean: float
    daily_return_std: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade: float
    
    # Streaks
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Time-based
    avg_holding_period: float
    trading_days: int
    
    def to_dict(self) -> Dict[str, float]:
        return self.__dict__
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])


class PerformanceAnalyzer:
    """
    Comprehensive Performance Analysis
    ====================================
    Calculate 30+ performance metrics for trading strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252
    
    def analyze(self, equity_curve: pd.Series,
               trades: Optional[pd.DataFrame] = None,
               benchmark: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Perform complete performance analysis
        
        Parameters:
        -----------
        equity_curve : Series of portfolio values over time
        trades : DataFrame with trade data (optional)
        benchmark : Benchmark returns for comparison (optional)
        """
        returns = equity_curve.pct_change().dropna()
        
        # Returns metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        trading_days = len(returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Drawdown analysis
        dd_info = self._calculate_drawdown(equity_curve)
        
        # VaR and CVaR
        var_95 = self._calculate_var(returns, 0.05)
        var_99 = self._calculate_var(returns, 0.01)
        cvar_95 = self._calculate_cvar(returns, 0.05)
        
        # Risk-adjusted ratios
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        calmar = annualized_return / abs(dd_info['max_drawdown']) if dd_info['max_drawdown'] != 0 else 0
        omega = self._calculate_omega(returns)
        
        # Information ratio (if benchmark provided)
        if benchmark is not None:
            info_ratio = self._calculate_information_ratio(returns, benchmark)
        else:
            info_ratio = 0
        
        # Trade statistics
        trade_stats = self._analyze_trades(trades) if trades is not None else {}
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_return_mean=returns.mean(),
            daily_return_std=returns.std(),
            volatility=volatility,
            max_drawdown=dd_info['max_drawdown'],
            max_drawdown_duration=dd_info['max_duration'],
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            conditional_var_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            information_ratio=info_ratio,
            total_trades=trade_stats.get('total_trades', 0),
            winning_trades=trade_stats.get('winning_trades', 0),
            losing_trades=trade_stats.get('losing_trades', 0),
            win_rate=trade_stats.get('win_rate', 0),
            profit_factor=trade_stats.get('profit_factor', 0),
            average_win=trade_stats.get('average_win', 0),
            average_loss=trade_stats.get('average_loss', 0),
            largest_win=trade_stats.get('largest_win', 0),
            largest_loss=trade_stats.get('largest_loss', 0),
            average_trade=trade_stats.get('average_trade', 0),
            max_consecutive_wins=trade_stats.get('max_consecutive_wins', 0),
            max_consecutive_losses=trade_stats.get('max_consecutive_losses', 0),
            avg_holding_period=trade_stats.get('avg_holding_period', 0),
            trading_days=trading_days
        )
    
    def _calculate_drawdown(self, equity: pd.Series) -> Dict[str, float]:
        """Calculate drawdown metrics"""
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        
        max_dd = drawdown.min()
        
        # Calculate max drawdown duration
        in_drawdown = drawdown < 0
        dd_groups = (~in_drawdown).cumsum()[in_drawdown]
        
        if len(dd_groups) > 0:
            max_duration = dd_groups.value_counts().max()
        else:
            max_duration = 0
        
        return {
            'max_drawdown': max_dd,
            'max_duration': max_duration,
            'current_drawdown': drawdown.iloc[-1]
        }
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.daily_rf
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - self.daily_rf
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_omega(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        above = returns[returns > threshold].sum()
        below = abs(returns[returns < threshold].sum())
        
        if below == 0:
            return float('inf')
        
        return above / below
    
    def _calculate_information_ratio(self, returns: pd.Series,
                                     benchmark: pd.Series) -> float:
        """Calculate Information ratio"""
        active_return = returns - benchmark
        tracking_error = active_return.std()
        
        if tracking_error == 0:
            return 0
        
        return np.sqrt(252) * active_return.mean() / tracking_error
    
    def _analyze_trades(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Analyze trade statistics"""
        if trades is None or len(trades) == 0:
            return {}
        
        pnls = trades['pnl'].values if 'pnl' in trades.columns else []
        
        if len(pnls) == 0:
            return {}
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # Calculate streaks
        streaks = self._calculate_streaks(pnls)
        
        # Holding period
        if 'hold_time' in trades.columns:
            avg_hold = trades['hold_time'].mean()
            if isinstance(avg_hold, timedelta):
                avg_hold = avg_hold.total_seconds() / 86400
        else:
            avg_hold = 0
        
        return {
            'total_trades': len(pnls),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else float('inf'),
            'average_win': np.mean(wins) if wins else 0,
            'average_loss': np.mean(losses) if losses else 0,
            'largest_win': max(pnls) if pnls else 0,
            'largest_loss': min(pnls) if pnls else 0,
            'average_trade': np.mean(pnls),
            'max_consecutive_wins': streaks['max_wins'],
            'max_consecutive_losses': streaks['max_losses'],
            'avg_holding_period': avg_hold
        }
    
    def _calculate_streaks(self, pnls: List[float]) -> Dict[str, int]:
        """Calculate win/loss streaks"""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return {'max_wins': max_wins, 'max_losses': max_losses}


class StrategyComparison:
    """
    Compare multiple strategies side-by-side
    """
    
    def __init__(self):
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.analyzer = PerformanceAnalyzer()
    
    def add_strategy(self, name: str, equity_curve: pd.Series,
                    trades: Optional[pd.DataFrame] = None,
                    description: str = ""):
        """Add a strategy for comparison"""
        metrics = self.analyzer.analyze(equity_curve, trades)
        
        self.strategies[name] = {
            'equity_curve': equity_curve,
            'trades': trades,
            'metrics': metrics,
            'description': description
        }
    
    def compare(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare strategies across selected metrics
        
        Parameters:
        -----------
        metrics : List of metric names to compare (uses defaults if None)
        """
        if metrics is None:
            metrics = [
                'total_return', 'annualized_return', 'volatility',
                'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
                'win_rate', 'profit_factor', 'calmar_ratio'
            ]
        
        data = {}
        for name, strategy in self.strategies.items():
            m = strategy['metrics']
            data[name] = {metric: getattr(m, metric, None) for metric in metrics}
        
        df = pd.DataFrame(data).T
        return df
    
    def rank_strategies(self, metric: str = 'sharpe_ratio',
                       ascending: bool = False) -> pd.DataFrame:
        """Rank strategies by a specific metric"""
        comparison = self.compare([metric])
        return comparison.sort_values(metric, ascending=ascending)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calculate return correlations between strategies"""
        returns = {}
        for name, strategy in self.strategies.items():
            returns[name] = strategy['equity_curve'].pct_change().dropna()
        
        df = pd.DataFrame(returns)
        return df.corr()


class PerformanceVisualizer:
    """
    Create professional performance visualizations
    """
    
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Visualizations will be limited.")
    
    def plot_equity_curve(self, equity_curve: pd.Series,
                         benchmark: Optional[pd.Series] = None,
                         title: str = "Equity Curve") -> go.Figure:
        """Plot equity curve with optional benchmark"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        # Strategy equity
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name='Strategy',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # Benchmark
        if benchmark is not None:
            # Normalize to same starting point
            benchmark_norm = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
            fig.add_trace(go.Scatter(
                x=benchmark_norm.index,
                y=benchmark_norm.values,
                name='Benchmark',
                line=dict(color='#A23B72', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_drawdown(self, equity_curve: pd.Series,
                     title: str = "Drawdown Analysis") -> go.Figure:
        """Plot drawdown chart"""
        if not PLOTLY_AVAILABLE:
            return None
        
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#E63946', width=1),
            fillcolor='rgba(230, 57, 70, 0.3)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white'
        )
        
        return fig
    
    def plot_monthly_returns(self, equity_curve: pd.Series,
                            title: str = "Monthly Returns Heatmap") -> go.Figure:
        """Plot monthly returns heatmap"""
        if not PLOTLY_AVAILABLE:
            return None
        
        returns = equity_curve.pct_change().dropna()
        
        # Resample to monthly
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values * 100
        })
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:pivot.shape[1]],
            y=pivot.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate='%{text:.1f}%',
            hovertemplate='%{y} %{x}: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white'
        )
        
        return fig
    
    def plot_returns_distribution(self, equity_curve: pd.Series,
                                  title: str = "Returns Distribution") -> go.Figure:
        """Plot returns distribution histogram"""
        if not PLOTLY_AVAILABLE:
            return None
        
        returns = equity_curve.pct_change().dropna() * 100
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=['Daily Returns Distribution', 'Q-Q Plot'])
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='#2E86AB'
        ), row=1, col=1)
        
        # Q-Q plot (normal comparison)
        sorted_returns = np.sort(returns)
        theoretical = np.random.normal(returns.mean(), returns.std(), len(returns))
        theoretical = np.sort(theoretical)
        
        fig.add_trace(go.Scatter(
            x=theoretical,
            y=sorted_returns,
            mode='markers',
            name='Q-Q',
            marker=dict(color='#2E86AB', size=4)
        ), row=1, col=2)
        
        # 45-degree line
        fig.add_trace(go.Scatter(
            x=[theoretical.min(), theoretical.max()],
            y=[theoretical.min(), theoretical.max()],
            mode='lines',
            name='Normal',
            line=dict(color='red', dash='dash')
        ), row=1, col=2)
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def plot_rolling_metrics(self, equity_curve: pd.Series,
                            window: int = 63,
                            title: str = "Rolling Performance") -> go.Figure:
        """Plot rolling Sharpe and volatility"""
        if not PLOTLY_AVAILABLE:
            return None
        
        returns = equity_curve.pct_change().dropna()
        
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=['Rolling Return', 'Rolling Volatility', 'Rolling Sharpe'])
        
        fig.add_trace(go.Scatter(
            x=rolling_return.index, y=rolling_return.values,
            name='Return', line=dict(color='#2E86AB')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol.values,
            name='Volatility', line=dict(color='#E63946')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            name='Sharpe', line=dict(color='#2A9D8F')
        ), row=3, col=1)
        
        # Add zero line for Sharpe
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=3, col=1)
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_strategy_comparison(self, strategies: Dict[str, pd.Series],
                                title: str = "Strategy Comparison") -> go.Figure:
        """Compare multiple strategy equity curves"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        colors = ['#2E86AB', '#E63946', '#2A9D8F', '#F4A261', '#9B5DE5', '#00BBF9']
        
        for i, (name, equity) in enumerate(strategies.items()):
            # Normalize to 100
            normalized = equity / equity.iloc[0] * 100
            
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized.values,
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Normalized Value (Base 100)',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    def create_tearsheet(self, equity_curve: pd.Series,
                        trades: Optional[pd.DataFrame] = None,
                        benchmark: Optional[pd.Series] = None,
                        name: str = "Strategy") -> go.Figure:
        """Create a comprehensive performance tearsheet"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Equity Curve', 'Drawdown',
                'Monthly Returns', 'Returns Distribution',
                'Rolling Sharpe (63d)', 'Trade P&L Distribution'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.08
        )
        
        returns = equity_curve.pct_change().dropna()
        
        # 1. Equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index, y=equity_curve.values,
            name='Equity', line=dict(color='#2E86AB', width=2)
        ), row=1, col=1)
        
        if benchmark is not None:
            benchmark_norm = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
            fig.add_trace(go.Scatter(
                x=benchmark_norm.index, y=benchmark_norm.values,
                name='Benchmark', line=dict(color='gray', width=1, dash='dash')
            ), row=1, col=1)
        
        # 2. Drawdown
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak * 100
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill='tozeroy', name='Drawdown',
            line=dict(color='#E63946', width=1),
            fillcolor='rgba(230, 57, 70, 0.3)'
        ), row=1, col=2)
        
        # 3. Monthly returns heatmap
        if len(returns) > 30:
            monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_df = pd.DataFrame({
                'year': monthly.index.year,
                'month': monthly.index.month,
                'return': monthly.values * 100
            })
            if len(monthly_df) > 0:
                pivot = monthly_df.pivot(index='year', columns='month', values='return')
                fig.add_trace(go.Heatmap(
                    z=pivot.values,
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:pivot.shape[1]],
                    y=pivot.index.astype(str),
                    colorscale='RdYlGn', zmid=0,
                    showscale=False
                ), row=2, col=1)
        
        # 4. Returns distribution
        fig.add_trace(go.Histogram(
            x=returns * 100, nbinsx=50, name='Returns',
            marker_color='#2E86AB'
        ), row=2, col=2)
        
        # 5. Rolling Sharpe
        rolling_sharpe = (returns.rolling(63).mean() / returns.rolling(63).std()) * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            name='Rolling Sharpe', line=dict(color='#2A9D8F', width=1)
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=3, col=1)
        fig.add_hline(y=1, line_dash='dot', line_color='green', row=3, col=1)
        
        # 6. Trade P&L
        if trades is not None and 'pnl' in trades.columns:
            fig.add_trace(go.Histogram(
                x=trades['pnl'], nbinsx=30, name='Trade P&L',
                marker_color='#2E86AB'
            ), row=3, col=2)
        
        fig.update_layout(
            title=f"{name} Performance Tearsheet",
            template='plotly_white',
            height=900,
            showlegend=False
        )
        
        return fig


class ReportGenerator:
    """Generate performance reports in various formats"""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.visualizer = PerformanceVisualizer()
    
    def generate_text_report(self, equity_curve: pd.Series,
                            trades: Optional[pd.DataFrame] = None,
                            name: str = "Strategy") -> str:
        """Generate text-based performance report"""
        metrics = self.analyzer.analyze(equity_curve, trades)
        
        report = f"""
{'='*60}
PERFORMANCE REPORT: {name}
{'='*60}

RETURNS
-------
Total Return:        {metrics.total_return*100:>10.2f}%
Annualized Return:   {metrics.annualized_return*100:>10.2f}%
Daily Return (Avg):  {metrics.daily_return_mean*100:>10.4f}%
Daily Return (Std):  {metrics.daily_return_std*100:>10.4f}%

RISK METRICS
------------
Volatility (Ann):    {metrics.volatility*100:>10.2f}%
Max Drawdown:        {metrics.max_drawdown*100:>10.2f}%
Max DD Duration:     {metrics.max_drawdown_duration:>10d} days
VaR (95%):           {metrics.value_at_risk_95*100:>10.2f}%
VaR (99%):           {metrics.value_at_risk_99*100:>10.2f}%
CVaR (95%):          {metrics.conditional_var_95*100:>10.2f}%

RISK-ADJUSTED RETURNS
---------------------
Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}
Sortino Ratio:       {metrics.sortino_ratio:>10.2f}
Calmar Ratio:        {metrics.calmar_ratio:>10.2f}
Omega Ratio:         {metrics.omega_ratio:>10.2f}

TRADE STATISTICS
----------------
Total Trades:        {metrics.total_trades:>10d}
Winning Trades:      {metrics.winning_trades:>10d}
Losing Trades:       {metrics.losing_trades:>10d}
Win Rate:            {metrics.win_rate*100:>10.2f}%
Profit Factor:       {metrics.profit_factor:>10.2f}
Average Win:         ${metrics.average_win:>9.2f}
Average Loss:        ${metrics.average_loss:>9.2f}
Largest Win:         ${metrics.largest_win:>9.2f}
Largest Loss:        ${metrics.largest_loss:>9.2f}

STREAKS
-------
Max Consecutive Wins:   {metrics.max_consecutive_wins:>7d}
Max Consecutive Losses: {metrics.max_consecutive_losses:>7d}

{'='*60}
"""
        return report
    
    def generate_html_report(self, equity_curve: pd.Series,
                            trades: Optional[pd.DataFrame] = None,
                            benchmark: Optional[pd.Series] = None,
                            name: str = "Strategy",
                            save_path: Optional[str] = None) -> str:
        """Generate HTML performance report with charts"""
        metrics = self.analyzer.analyze(equity_curve, trades, benchmark)
        
        # Generate charts
        tearsheet = self.visualizer.create_tearsheet(equity_curve, trades, benchmark, name)
        chart_html = tearsheet.to_html(full_html=False) if tearsheet else ""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{name} Performance Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .metric-label {{ color: #666; font-size: 12px; }}
        .positive {{ color: #2A9D8F; }}
        .negative {{ color: #E63946; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #2E86AB; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {name} Performance Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.total_return > 0 else 'negative'}">{metrics.total_return*100:.2f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{metrics.max_drawdown*100:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.win_rate*100:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
        </div>
        
        <h2>Performance Charts</h2>
        {chart_html}
        
        <h2>Detailed Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Annualized Return</td><td>{metrics.annualized_return*100:.2f}%</td></tr>
            <tr><td>Volatility</td><td>{metrics.volatility*100:.2f}%</td></tr>
            <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.2f}</td></tr>
            <tr><td>Calmar Ratio</td><td>{metrics.calmar_ratio:.2f}</td></tr>
            <tr><td>Profit Factor</td><td>{metrics.profit_factor:.2f}</td></tr>
            <tr><td>Total Trades</td><td>{metrics.total_trades}</td></tr>
            <tr><td>Average Trade</td><td>${metrics.average_trade:.2f}</td></tr>
        </table>
    </div>
</body>
</html>
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html)
        
        return html


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE ANALYTICS DEMO")
    print("=" * 60)
    
    # Generate sample equity curve
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = np.random.randn(252) * 0.02 + 0.0005
    equity = pd.Series(100000 * (1 + returns).cumprod(), index=dates)
    
    # Generate sample trades
    trades = pd.DataFrame({
        'pnl': np.random.randn(50) * 500 + 100,
        'hold_time': pd.to_timedelta(np.random.randint(1, 30, 50), unit='D')
    })
    
    # Analyze performance
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(equity, trades)
    
    # Print report
    reporter = ReportGenerator()
    print(reporter.generate_text_report(equity, trades, "Sample Strategy"))
    
    # Compare strategies
    print("\nStrategy Comparison:")
    comparison = StrategyComparison()
    
    # Add multiple strategies
    for i, name in enumerate(['Momentum', 'Mean Reversion', 'ML Ensemble']):
        strategy_returns = np.random.randn(252) * 0.02 + 0.0003 * (i + 1)
        strategy_equity = pd.Series(100000 * (1 + strategy_returns).cumprod(), index=dates)
        comparison.add_strategy(name, strategy_equity)
    
    print(comparison.compare())
