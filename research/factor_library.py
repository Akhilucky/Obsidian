import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime

class FactorLibrary:
    def __init__(self):
        pass
    
    def calculate_momentum(self, data, window=20):
        return data['Close'].pct_change(window)
    
    def calculate_mean_reversion(self, data, window=20):
        ma = data['Close'].rolling(window).mean()
        return data['Close'] - ma
    
    def calculate_trend_following(self, data, short_window=20, long_window=50):
        ma_short = data['Close'].rolling(short_window).mean()
        ma_long = data['Close'].rolling(long_window).mean()
        return ma_short - ma_long
    
    def calculate_volatility(self, data, window=20):
        return data['Close'].pct_change().rolling(window).std() * np.sqrt(252)
    
    def calculate_volume_trend(self, data, window=20):
        return data['Volume'].rolling(window).mean()
    
    def calculate_rsi(self, data, window=14):
        return ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
    
    def calculate_macd(self, data, window_slow=26, window_fast=12, window_sign=9):
        macd = ta.trend.MACD(data['Close'], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
        return macd.macd(), macd.macd_signal(), macd.macd_diff()
    
    def calculate_bollinger_bands(self, data, window=20, window_dev=2):
        bb = ta.volatility.BollingerBands(data['Close'], window=window, window_dev=window_dev)
        return bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband(), bb.bollinger_pband()
    
    def calculate_on_balance_volume(self, data):
        return ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    
    def calculate_accumulation_distribution_index(self, data):
        return ta.volume.AccumulationDistributionIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
    
    def calculate_money_flow_index(self, data, window=14):
        return ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=window).money_flow_index()
    
    def calculate_ease_of_movement(self, data, window=14):
        em = ta.volume.EaseOfMovementIndicator(data['High'], data['Low'], data['Volume'], window=window)
        return em.ease_of_movement(), em.sma_ease_of_movement()
    
    def calculate_volume_price_trend(self, data, window=14):
        return ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume'], window=window).volume_price_trend()
    
    def calculate_force_index(self, data, window=13):
        return ta.volume.ForceIndexIndicator(data['Close'], data['Volume'], window=window).force_index()
    
    def calculate_negative_volume_index(self, data):
        return ta.volume.NegativeVolumeIndexIndicator(data['Close'], data['Volume']).negative_volume_index()
    
    def calculate_volume_zone_momentum_oscillator(self, data, window=30):
        return ta.volume.VolumeZoneMomentumOscillator(data['High'], data['Low'], data['Close'], data['Volume'], window=window).volume_zone_momentum_oscillator()
    
    def calculate_chaikin_money_flow(self, data, window=21):
        return ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=window).chaikin_money_flow()
    
    def calculate_daily_return(self, data):
        return data['Close'].pct_change()
    
    def calculate_daily_log_return(self, data):
        return np.log(data['Close'] / data['Close'].shift(1))
    
    def calculate_cumulative_return(self, data):
        return (data['Close'] / data['Close'].iloc[0]) - 1
    
    def calculate_simple_return(self, data):
        return data['Close'] / data['Close'].shift(1) - 1
    
    def calculate_log_return(self, data):
        return np.log(data['Close'] / data['Close'].shift(1))
    
    def calculate_geometric_mean_return(self, data):
        return data['Close'].prod() ** (1 / len(data)) - 1
    
    def calculate_arithmetic_mean_return(self, data):
        return data['Close'].mean()
    
    def calculate_median_return(self, data):
        return data['Close'].median()
    
    def calculate_sharpe_ratio(self, data, risk_free_rate=0.01):
        returns = data['Close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        return (mean_return - risk_free_rate) / std_return
    
    def calculate_sortino_ratio(self, data, risk_free_rate=0.01):
        returns = data['Close'].pct_change().dropna()
        mean_return = returns.mean()
        downside_returns = returns.copy()
        downside_returns[returns >= 0] = 0
        std_downside_return = downside_returns.std()
        return (mean_return - risk_free_rate) / std_downside_return
    
    def calculate_maximum_drawdown(self, data):
        cumulative_returns = (data['Close'] / data['Close'].iloc[0]) - 1
        rolling_max = cumulative_returns.cummax()
        drawdown = cumulative_returns - rolling_max
        return drawdown.min()
    
    def calculate_calmar_ratio(self, data):
        cumulative_returns = (data['Close'] / data['Close'].iloc[0]) - 1
        max_drawdown = self.calculate_maximum_drawdown(data)
        return cumulative_returns[-1] / abs(max_drawdown)
    
    def calculate_stability(self, data):
        returns = data['Close'].pct_change().dropna()
        return returns.kurtosis()
    
    def calculate_recovery_factor(self, data):
        cumulative_returns = (data['Close'] / data['Close'].iloc[0]) - 1
        max_drawdown = self.calculate_maximum_drawdown(data)
        return cumulative_returns[-1] / abs(max_drawdown)
    
    def calculate_pain_index(self, data):
        returns = data['Close'].pct_change().dropna()
        return returns[returns < 0].mean()
    
    def calculate_gain_loss_ratio(self, data):
        returns = data['Close'].pct_change().dropna()
        gains = returns[returns > 0].mean()
        losses = returns[returns < 0].mean()
        return gains / abs(losses)
    
    def calculate_profit_ratio(self, data):
        returns = data['Close'].pct_change().dropna()
        positive_returns = returns[returns > 0].count()
        negative_returns = returns[returns < 0].count()
        return positive_returns / negative_returns
    
    def calculate_profit_factor(self, data):
        returns = data['Close'].pct_change().dropna()
        positive_returns = returns[returns > 0].sum()
        negative_returns = returns[returns < 0].sum()
        return positive_returns / abs(negative_returns)
    
    def calculate_cagr(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        cumulative_returns = (1 + daily_returns).prod()
        years = len(data) / 252
        return cumulative_returns ** (1 / years) - 1
    
    def calculate_annual_volatility(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        return daily_returns.std() * np.sqrt(252)
    
    def calculate_downside_risk(self, data, risk_free_rate=0.01):
        daily_returns = data['Close'].pct_change().dropna()
        downside_returns = daily_returns.copy()
        downside_returns[daily_returns >= risk_free_rate] = 0
        return downside_returns.std() * np.sqrt(252)
    
    def calculate_semivariance(self, data, threshold=0):
        daily_returns = data['Close'].pct_change().dropna()
        below_threshold = daily_returns[daily_returns < threshold]
        return below_threshold.var()
    
    def calculate_downside_variance(self, data, threshold=0):
        daily_returns = data['Close'].pct_change().dropna()
        below_threshold = daily_returns[daily_returns < threshold]
        return below_threshold.var()
    
    def calculate_lower_partial_moment(self, data, threshold=0, degree=2):
        daily_returns = data['Close'].pct_change().dropna()
        below_threshold = daily_returns[daily_returns < threshold]
        return ((threshold - below_threshold) ** degree).mean()
    
    def calculate_capital_allocation_line(self, data, risk_free_rate=0.01):
        returns = data['Close'].pct_change().dropna()
        expected_return = returns.mean()
        standard_deviation = returns.std()
        return (expected_return - risk_free_rate) / standard_deviation
    
    def calculate_beta(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        covariance = security_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        return covariance / benchmark_variance
    
    def calculate_alpha(self, data, benchmark_data, risk_free_rate=0.01):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        beta = self.calculate_beta(data, benchmark_data)
        expected_security_return = security_returns.mean()
        expected_benchmark_return = benchmark_returns.mean()
        return expected_security_return - (risk_free_rate + beta * (expected_benchmark_return - risk_free_rate))
    
    def calculate_jensen_alpha(self, data, benchmark_data, risk_free_rate=0.01):
        return self.calculate_alpha(data, benchmark_data, risk_free_rate)
    
    def calculate_tracking_error(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        return (security_returns - benchmark_returns).std()
    
    def calculate_information_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        active_returns = security_returns - benchmark_returns
        return active_returns.mean() / active_returns.std()
    
    def calculate_tail_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        return daily_returns[daily_returns > 0].quantile(0.95) / abs(daily_returns[daily_returns < 0].quantile(0.05))
    
    def calculate_capture_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        # Up capture
        up_market = benchmark_returns[benchmark_returns > 0]
        up_security = security_returns[benchmark_returns > 0]
        up_capture = up_security.mean() / up_market.mean()
        
        # Down capture
        down_market = benchmark_returns[benchmark_returns < 0]
        down_security = security_returns[benchmark_returns < 0]
        down_capture = down_security.mean() / down_market.mean()
        
        return up_capture, down_capture
    
    def calculate_up_down_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].count()
        negative_returns = daily_returns[daily_returns < 0].count()
        return positive_returns / negative_returns
    
    def calculate_up_down_balance(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].sum()
        negative_returns = daily_returns[daily_returns < 0].sum()
        return positive_returns / abs(negative_returns)
    
    def calculate_win_loss_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].mean()
        negative_returns = daily_returns[daily_returns < 0].mean()
        return positive_returns / abs(negative_returns)
    
    def calculate_positive_negative_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].count()
        negative_returns = daily_returns[daily_returns < 0].count()
        return positive_returns / negative_returns
    
    def calculate_positive_negative_balance(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].sum()
        negative_returns = daily_returns[daily_returns < 0].sum()
        return positive_returns / abs(negative_returns)
    
    def calculate_positive_negative_variance_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.var() / negative_returns.var()
    
    def calculate_positive_negative_skewness_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.skew() / negative_returns.skew()
    
    def calculate_positive_negative_kurtosis_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.kurtosis() / negative_returns.kurtosis()
    
    def calculate_positive_negative_sharpe_ratio(self, data, risk_free_rate=0.01):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_sharpe = (positive_returns.mean() - risk_free_rate) / positive_returns.std()
        negative_sharpe = (negative_returns.mean() - risk_free_rate) / negative_returns.std()
        
        return positive_sharpe / abs(negative_sharpe)
    
    def calculate_positive_negative_sortino_ratio(self, data, risk_free_rate=0.01):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_sortino = (positive_returns.mean() - risk_free_rate) / positive_returns.std()
        negative_sortino = (negative_returns.mean() - risk_free_rate) / negative_returns.std()
        
        return positive_sortino / abs(negative_sortino)
    
    def calculate_positive_negative_calmar_ratio(self, data):
        positive_returns = data[data > 0]
        negative_returns = data[data < 0]
        
        positive_calmar = positive_returns.mean() / abs(positive_returns.min())
        negative_calmar = negative_returns.mean() / abs(negative_returns.min())
        
        return positive_calmar / abs(negative_calmar)
    
    def calculate_positive_negative_stability_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.kurtosis() / negative_returns.kurtosis()
    
    def calculate_positive_negative_recovery_factor_ratio(self, data):
        positive_returns = data[data > 0]
        negative_returns = data[data < 0]
        
        positive_recovery = positive_returns.mean() / abs(positive_returns.min())
        negative_recovery = negative_returns.mean() / abs(negative_returns.min())
        
        return positive_recovery / abs(negative_recovery)
    
    def calculate_positive_negative_pain_index_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.mean() / abs(negative_returns.mean())
    
    def calculate_positive_negative_gain_loss_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.mean() / abs(negative_returns.mean())
    
    def calculate_positive_negative_profit_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].count()
        negative_returns = daily_returns[daily_returns < 0].count()
        return positive_returns / negative_returns
    
    def calculate_positive_negative_profit_factor(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].sum()
        negative_returns = daily_returns[daily_returns < 0].sum()
        return positive_returns / abs(negative_returns)
    
    def calculate_positive_negative_cagr_ratio(self, data):
        positive_returns = data[data > 0]
        negative_returns = data[data < 0]
        
        positive_cagr = self.calculate_cagr(positive_returns)
        negative_cagr = self.calculate_cagr(negative_returns)
        
        return positive_cagr / abs(negative_cagr)
    
    def calculate_positive_negative_annual_volatility_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.std() / negative_returns.std()
    
    def calculate_positive_negative_downside_risk_ratio(self, data, risk_free_rate=0.01):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_downside = self.calculate_downside_risk(positive_returns, risk_free_rate)
        negative_downside = self.calculate_downside_risk(negative_returns, risk_free_rate)
        
        return positive_downside / abs(negative_downside)
    
    def calculate_positive_negative_semivariance_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.var() / negative_returns.var()
    
    def calculate_positive_negative_downside_variance_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.var() / negative_returns.var()
    
    def calculate_positive_negative_lpm_ratio(self, data, threshold=0, degree=2):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_lpm = self.calculate_lower_partial_moment(positive_returns, threshold, degree)
        negative_lpm = self.calculate_lower_partial_moment(negative_returns, threshold, degree)
        
        return positive_lpm / abs(negative_lpm)
    
    def calculate_positive_negative_capital_allocation_line_ratio(self, data, risk_free_rate=0.01):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_cal = self.calculate_capital_allocation_line(positive_returns, risk_free_rate)
        negative_cal = self.calculate_capital_allocation_line(negative_returns, risk_free_rate)
        
        return positive_cal / abs(negative_cal)
    
    def calculate_positive_negative_beta_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_beta = self.calculate_beta(positive_returns, benchmark_returns)
        negative_beta = self.calculate_beta(negative_returns, benchmark_returns)
        
        return positive_beta / abs(negative_beta)
    
    def calculate_positive_negative_alpha_ratio(self, data, benchmark_data, risk_free_rate=0.01):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_alpha = self.calculate_alpha(positive_returns, benchmark_returns, risk_free_rate)
        negative_alpha = self.calculate_alpha(negative_returns, benchmark_returns, risk_free_rate)
        
        return positive_alpha / abs(negative_alpha)
    
    def calculate_positive_negative_jensen_alpha_ratio(self, data, benchmark_data, risk_free_rate=0.01):
        return self.calculate_positive_negative_alpha_ratio(data, benchmark_data, risk_free_rate)
    
    def calculate_positive_negative_tracking_error_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_te = self.calculate_tracking_error(positive_returns, benchmark_returns)
        negative_te = self.calculate_tracking_error(negative_returns, benchmark_returns)
        
        return positive_te / abs(negative_te)
    
    def calculate_positive_negative_information_ratio_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_ir = self.calculate_information_ratio(positive_returns, benchmark_returns)
        negative_ir = self.calculate_information_ratio(negative_returns, benchmark_returns)
        
        return positive_ir / abs(negative_ir)
    
    def calculate_positive_negative_tail_ratio_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_tail = self.calculate_tail_ratio(positive_returns)
        negative_tail = self.calculate_tail_ratio(negative_returns)
        
        return positive_tail / abs(negative_tail)
    
    def calculate_positive_negative_capture_ratio_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_up_cap, positive_down_cap = self.calculate_capture_ratio(positive_returns, benchmark_returns)
        negative_up_cap, negative_down_cap = self.calculate_capture_ratio(negative_returns, benchmark_returns)
        
        return (positive_up_cap / abs(negative_up_cap), positive_down_cap / abs(negative_down_cap))
    
    def calculate_positive_negative_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].count()
        negative_returns = daily_returns[daily_returns < 0].count()
        return positive_returns / negative_returns
    
    def calculate_positive_negative_balance(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].sum()
        negative_returns = daily_returns[daily_returns < 0].sum()
        return positive_returns / abs(negative_returns)
    
    def calculate_positive_negative_win_loss_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].mean()
        negative_returns = daily_returns[daily_returns < 0].mean()
        return positive_returns / abs(negative_returns)
    
    def calculate_positive_negative_kurtosis_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.kurtosis() / negative_returns.kurtosis()
    
    def calculate_positive_negative_skewness_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.skew() / negative_returns.skew()
    
    def calculate_positive_negative_sortino_ratio_ratio(self, data, risk_free_rate=0.01):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_sortino = self.calculate_sortino_ratio(positive_returns, risk_free_rate)
        negative_sortino = self.calculate_sortino_ratio(negative_returns, risk_free_rate)
        
        return positive_sortino / abs(negative_sortino)
    
    def calculate_positive_negative_calmar_ratio_ratio(self, data):
        positive_returns = data[data > 0]
        negative_returns = data[data < 0]
        
        positive_calmar = self.calculate_calmar_ratio(positive_returns)
        negative_calmar = self.calculate_calmar_ratio(negative_returns)
        
        return positive_calmar / abs(negative_calmar)
    
    def calculate_positive_negative_recovery_factor_ratio(self, data):
        positive_returns = data[data > 0]
        negative_returns = data[data < 0]
        
        positive_recovery = self.calculate_recovery_factor(positive_returns)
        negative_recovery = self.calculate_recovery_factor(negative_returns)
        
        return positive_recovery / abs(negative_recovery)
    
    def calculate_positive_negative_pain_index_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.mean() / abs(negative_returns.mean())
    
    def calculate_positive_negative_gain_loss_ratio_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.mean() / abs(negative_returns.mean())
    
    def calculate_positive_negative_profit_ratio_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].count()
        negative_returns = daily_returns[daily_returns < 0].count()
        return positive_returns / negative_returns
    
    def calculate_positive_negative_profit_factor_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0].sum()
        negative_returns = daily_returns[daily_returns < 0].sum()
        return positive_returns / abs(negative_returns)
    
    def calculate_positive_negative_cagr_ratio_ratio(self, data):
        positive_returns = data[data > 0]
        negative_returns = data[data < 0]
        
        positive_cagr = self.calculate_cagr(positive_returns)
        negative_cagr = self.calculate_cagr(negative_returns)
        
        return positive_cagr / abs(negative_cagr)
    
    def calculate_positive_negative_annual_volatility_ratio_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.std() / negative_returns.std()
    
    def calculate_positive_negative_downside_risk_ratio_ratio(self, data, risk_free_rate=0.01):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_downside = self.calculate_downside_risk(positive_returns, risk_free_rate)
        negative_downside = self.calculate_downside_risk(negative_returns, risk_free_rate)
        
        return positive_downside / abs(negative_downside)
    
    def calculate_positive_negative_semivariance_ratio_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.var() / negative_returns.var()
    
    def calculate_positive_negative_downside_variance_ratio_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        return positive_returns.var() / negative_returns.var()
    
    def calculate_positive_negative_lpm_ratio_ratio(self, data, threshold=0, degree=2):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_lpm = self.calculate_lower_partial_moment(positive_returns, threshold, degree)
        negative_lpm = self.calculate_lower_partial_moment(negative_returns, threshold, degree)
        
        return positive_lpm / abs(negative_lpm)
    
    def calculate_positive_negative_capital_allocation_line_ratio_ratio(self, data, risk_free_rate=0.01):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_cal = self.calculate_capital_allocation_line(positive_returns, risk_free_rate)
        negative_cal = self.calculate_capital_allocation_line(negative_returns, risk_free_rate)
        
        return positive_cal / abs(negative_cal)
    
    def calculate_positive_negative_beta_ratio_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_beta = self.calculate_beta(positive_returns, benchmark_returns)
        negative_beta = self.calculate_beta(negative_returns, benchmark_returns)
        
        return positive_beta / abs(negative_beta)
    
    def calculate_positive_negative_alpha_ratio_ratio(self, data, benchmark_data, risk_free_rate=0.01):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_alpha = self.calculate_alpha(positive_returns, benchmark_returns, risk_free_rate)
        negative_alpha = self.calculate_alpha(negative_returns, benchmark_returns, risk_free_rate)
        
        return positive_alpha / abs(negative_alpha)
    
    def calculate_positive_negative_jensen_alpha_ratio_ratio(self, data, benchmark_data, risk_free_rate=0.01):
        return self.calculate_positive_negative_alpha_ratio_ratio(data, benchmark_data, risk_free_rate)
    
    def calculate_positive_negative_tracking_error_ratio_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_te = self.calculate_tracking_error(positive_returns, benchmark_returns)
        negative_te = self.calculate_tracking_error(negative_returns, benchmark_returns)
        
        return positive_te / abs(negative_te)
    
    def calculate_positive_negative_information_ratio_ratio_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_ir = self.calculate_information_ratio(positive_returns, benchmark_returns)
        negative_ir = self.calculate_information_ratio(negative_returns, benchmark_returns)
        
        return positive_ir / abs(negative_ir)
    
    def calculate_positive_negative_tail_ratio_ratio_ratio(self, data):
        daily_returns = data['Close'].pct_change().dropna()
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        
        positive_tail = self.calculate_tail_ratio(positive_returns)
        negative_tail = self.calculate_tail_ratio(negative_returns)
        
        return positive_tail / abs(negative_tail)
    
    def calculate_positive_negative_capture_ratio_ratio_ratio(self, data, benchmark_data):
        security_returns = data['Close'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        positive_returns = security_returns[security_returns > 0]
        negative_returns = security_returns[security_returns < 0]
        
        positive_up_cap, positive_down_cap = self.calculate_capture_ratio(positive_returns, benchmark_returns)
        negative_up_cap, negative_down_cap = self.calculate_capture_ratio(negative_returns, benchmark_returns)
        
        return (positive_up_cap / abs(negative_up_cap), positive_down_cap / abs(negative_down_cap))

if __name__ == "__main__":
    # Initialize factor library
    factor_lib = FactorLibrary()
    
    # Fetch data
    data = yf.download('AAPL', start='2023-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    
    # Calculate factors
    data['Momentum'] = factor_lib.calculate_momentum(data)
    data['Mean Reversion'] = factor_lib.calculate_mean_reversion(data)
    data['Trend Following'] = factor_lib.calculate_trend_following(data)
    data['Volatility'] = factor_lib.calculate_volatility(data)
    data['Volume Trend'] = factor_lib.calculate_volume_trend(data)
    data['RSI'] = factor_lib.calculate_rsi(data)
    macd, macd_signal, macd_diff = factor_lib.calculate_macd(data)
    data['MACD'] = macd
    data['MACD Signal'] = macd_signal
    data['MACD Diff'] = macd_diff
    mavg, hband, lband, wband, pband = factor_lib.calculate_bollinger_bands(data)
    data['Bollinger Mavg'] = mavg
    data['Bollinger Hband'] = hband
    data['Bollinger Lband'] = lband
    data['Bollinger Wband'] = wband
    data['Bollinger Pband'] = pband
    data['On Balance Volume'] = factor_lib.calculate_on_balance_volume(data)
    data['Accumulation Distribution Index'] = factor_lib.calculate_accumulation_distribution_index(data)
    data['Money Flow Index'] = factor_lib.calculate_money_flow_index(data)
    em, sma_em = factor_lib.calculate_Ease_of_Movement(data)
    data['Ease of Movement'] = em
    data['SMA Ease of Movement'] = sma_em
    data['Volume Price Trend'] = factor_lib.calculate_volume_price_trend(data)
    data['Force Index'] = factor_lib.calculate_force_index(data)
    data['Negative Volume Index'] = factor_lib.calculate_negative_volume_index(data)
    data['Volume Zone Momentum Oscillator'] = factor_lib.calculate_volume_zone_momentum_oscillator(data)
    data['Chaikin Money Flow'] = factor_lib.calculate_chaikin_money_flow(data)
    
    # Display factors
    print("\nSample Factors:")
    print(data[['Momentum', 'Mean Reversion', 'Trend Following', 'Volatility', 'Volume Trend', 'RSI', 'MACD', 'MACD Signal', 'MACD Diff', 'Bollinger Mavg', 'Bollinger Hband', 'Bollinger Lband', 'Bollinger Wband', 'Bollinger Pband', 'On Balance Volume', 'Accumulation Distribution Index', 'Money Flow Index', 'Ease of Movement', 'SMA Ease of Movement', 'Volume Price Trend', 'Force Index', 'Negative Volume Index', 'Volume Zone Momentum Oscillator', 'Chaikin Money Flow']].tail())
