"""
Strategy Builder - Custom Trading Strategy Framework
====================================================
Build, test, and deploy custom trading strategies with a modular architecture.
Supports visual strategy building, code-based strategies, and hybrid approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SignalType(Enum):
    """Types of trading signals"""
    BUY = 1
    SELL = -1
    HOLD = 0
    STRONG_BUY = 2
    STRONG_SELL = -2


class IndicatorType(Enum):
    """Available technical indicators"""
    SMA = "Simple Moving Average"
    EMA = "Exponential Moving Average"
    RSI = "Relative Strength Index"
    MACD = "MACD"
    BOLLINGER = "Bollinger Bands"
    ATR = "Average True Range"
    STOCHASTIC = "Stochastic Oscillator"
    ADX = "Average Directional Index"
    CCI = "Commodity Channel Index"
    WILLIAMS_R = "Williams %R"
    OBV = "On Balance Volume"
    VWAP = "Volume Weighted Average Price"
    ICHIMOKU = "Ichimoku Cloud"
    PIVOT = "Pivot Points"
    FIBONACCI = "Fibonacci Retracement"


class ConditionOperator(Enum):
    """Operators for building conditions"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL = "=="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    BETWEEN = "between"
    NOT_BETWEEN = "not_between"


@dataclass
class Indicator:
    """Represents a technical indicator with parameters"""
    type: IndicatorType
    params: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"{self.type.name}_{hash(str(self.params)) % 10000}"


@dataclass
class Condition:
    """A single condition in a strategy rule"""
    left_operand: Union[str, float]  # Indicator name or value
    operator: ConditionOperator
    right_operand: Union[str, float]  # Indicator name or value
    
    def evaluate(self, data: pd.DataFrame, idx: int) -> bool:
        """Evaluate the condition at a specific index"""
        left_val = self._get_value(self.left_operand, data, idx)
        right_val = self._get_value(self.right_operand, data, idx)
        
        if self.operator == ConditionOperator.GREATER_THAN:
            return left_val > right_val
        elif self.operator == ConditionOperator.LESS_THAN:
            return left_val < right_val
        elif self.operator == ConditionOperator.EQUAL:
            return left_val == right_val
        elif self.operator == ConditionOperator.GREATER_EQUAL:
            return left_val >= right_val
        elif self.operator == ConditionOperator.LESS_EQUAL:
            return left_val <= right_val
        elif self.operator == ConditionOperator.CROSSES_ABOVE:
            if idx == 0:
                return False
            prev_left = self._get_value(self.left_operand, data, idx - 1)
            prev_right = self._get_value(self.right_operand, data, idx - 1)
            return prev_left <= prev_right and left_val > right_val
        elif self.operator == ConditionOperator.CROSSES_BELOW:
            if idx == 0:
                return False
            prev_left = self._get_value(self.left_operand, data, idx - 1)
            prev_right = self._get_value(self.right_operand, data, idx - 1)
            return prev_left >= prev_right and left_val < right_val
        return False
    
    def _get_value(self, operand: Union[str, float], data: pd.DataFrame, idx: int) -> float:
        """Get value from data or return literal"""
        if isinstance(operand, (int, float)):
            return operand
        if operand in data.columns:
            return data.iloc[idx][operand]
        return float(operand)


@dataclass
class Rule:
    """A rule combining multiple conditions with AND/OR logic"""
    conditions: List[Condition]
    logic: str = "AND"  # "AND" or "OR"
    signal: SignalType = SignalType.BUY
    
    def evaluate(self, data: pd.DataFrame, idx: int) -> Optional[SignalType]:
        """Evaluate all conditions and return signal if triggered"""
        if self.logic == "AND":
            result = all(c.evaluate(data, idx) for c in self.conditions)
        else:  # OR
            result = any(c.evaluate(data, idx) for c in self.conditions)
        
        return self.signal if result else None


class IndicatorCalculator:
    """Calculates technical indicators"""
    
    @staticmethod
    def calculate(data: pd.DataFrame, indicator: Indicator) -> pd.Series:
        """Calculate indicator and return as Series"""
        calc = IndicatorCalculator()
        method_map = {
            IndicatorType.SMA: calc.sma,
            IndicatorType.EMA: calc.ema,
            IndicatorType.RSI: calc.rsi,
            IndicatorType.MACD: calc.macd,
            IndicatorType.BOLLINGER: calc.bollinger,
            IndicatorType.ATR: calc.atr,
            IndicatorType.STOCHASTIC: calc.stochastic,
            IndicatorType.ADX: calc.adx,
            IndicatorType.CCI: calc.cci,
            IndicatorType.WILLIAMS_R: calc.williams_r,
            IndicatorType.OBV: calc.obv,
            IndicatorType.VWAP: calc.vwap,
        }
        
        if indicator.type in method_map:
            return method_map[indicator.type](data, **indicator.params)
        return pd.Series(index=data.index)
    
    def sma(self, data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Simple Moving Average"""
        return data[column].rolling(window=period).mean()
    
    def ema(self, data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Exponential Moving Average"""
        return data[column].ewm(span=period, adjust=False).mean()
    
    def rsi(self, data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """Relative Strength Index"""
        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, 
             signal: int = 9, column: str = 'close') -> pd.Series:
        """MACD Line"""
        ema_fast = data[column].ewm(span=fast, adjust=False).mean()
        ema_slow = data[column].ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow
    
    def macd_signal(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, 
                    signal: int = 9, column: str = 'close') -> pd.Series:
        """MACD Signal Line"""
        macd_line = self.macd(data, fast, slow, signal, column)
        return macd_line.ewm(span=signal, adjust=False).mean()
    
    def bollinger(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                  column: str = 'close', band: str = 'middle') -> pd.Series:
        """Bollinger Bands"""
        sma = data[column].rolling(window=period).mean()
        std = data[column].rolling(window=period).std()
        
        if band == 'upper':
            return sma + (std_dev * std)
        elif band == 'lower':
            return sma - (std_dev * std)
        return sma  # middle
    
    def atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def stochastic(self, data: pd.DataFrame, k_period: int = 14, 
                   d_period: int = 3, line: str = 'k') -> pd.Series:
        """Stochastic Oscillator"""
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        k = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        
        if line == 'd':
            return k.rolling(window=d_period).mean()
        return k
    
    def adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        plus_dm = data['high'].diff()
        minus_dm = data['low'].diff().abs() * -1
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr = self.atr(data, 1) * period  # Approximate
        
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / tr.ewm(span=period).mean())
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / tr.ewm(span=period).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.ewm(span=period).mean()
    
    def cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_dev = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        return (typical_price - sma) / (0.015 * mean_dev)
    
    def williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        return -100 * (highest_high - data['close']) / (highest_high - lowest_low)
    
    def obv(self, data: pd.DataFrame) -> pd.Series:
        """On Balance Volume"""
        return (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
    
    def vwap(self, data: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()


class CustomStrategy(ABC):
    """Base class for custom strategies"""
    
    def __init__(self, name: str = "CustomStrategy"):
        self.name = name
        self.indicators: List[Indicator] = []
        self.entry_rules: List[Rule] = []
        self.exit_rules: List[Rule] = []
        self.parameters: Dict[str, Any] = {}
        self.position_sizing: Callable = lambda capital, price: capital * 0.1 / price
        
    @abstractmethod
    def initialize(self):
        """Initialize strategy parameters and rules"""
        pass
    
    def add_indicator(self, indicator: Indicator) -> str:
        """Add an indicator to the strategy"""
        self.indicators.append(indicator)
        return indicator.name
    
    def add_entry_rule(self, rule: Rule):
        """Add an entry rule"""
        self.entry_rules.append(rule)
    
    def add_exit_rule(self, rule: Rule):
        """Add an exit rule"""
        self.exit_rules.append(rule)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators and add to dataframe"""
        df = data.copy()
        calculator = IndicatorCalculator()
        
        for indicator in self.indicators:
            df[indicator.name] = calculator.calculate(df, indicator)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on rules"""
        df = self.prepare_data(data)
        df['signal'] = 0
        
        for idx in range(len(df)):
            # Check entry rules
            for rule in self.entry_rules:
                signal = rule.evaluate(df, idx)
                if signal:
                    df.iloc[idx, df.columns.get_loc('signal')] = signal.value
                    break
            
            # Check exit rules if in position
            if df.iloc[idx]['signal'] == 0:
                for rule in self.exit_rules:
                    signal = rule.evaluate(df, idx)
                    if signal:
                        df.iloc[idx, df.columns.get_loc('signal')] = signal.value
                        break
        
        return df
    
    def to_dict(self) -> Dict:
        """Serialize strategy to dictionary"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'indicators': [
                {'type': i.type.name, 'params': i.params, 'name': i.name}
                for i in self.indicators
            ],
            'entry_rules': self._serialize_rules(self.entry_rules),
            'exit_rules': self._serialize_rules(self.exit_rules)
        }
    
    def _serialize_rules(self, rules: List[Rule]) -> List[Dict]:
        """Serialize rules to list of dicts"""
        return [
            {
                'conditions': [
                    {
                        'left': c.left_operand,
                        'operator': c.operator.name,
                        'right': c.right_operand
                    }
                    for c in r.conditions
                ],
                'logic': r.logic,
                'signal': r.signal.name
            }
            for r in rules
        ]
    
    def save(self, filepath: str):
        """Save strategy to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CustomStrategy':
        """Load strategy from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return StrategyBuilder.from_dict(data)


class StrategyBuilder:
    """
    Visual/Programmatic Strategy Builder
    =====================================
    Build trading strategies using a modular, no-code approach.
    """
    
    def __init__(self, name: str = "MyStrategy"):
        self.name = name
        self.indicators: List[Indicator] = []
        self.entry_rules: List[Rule] = []
        self.exit_rules: List[Rule] = []
        self.parameters: Dict[str, Any] = {}
        self.risk_management = RiskManagement()
        
    def add_sma(self, period: int = 20, column: str = 'close') -> str:
        """Add Simple Moving Average indicator"""
        indicator = Indicator(
            type=IndicatorType.SMA,
            params={'period': period, 'column': column},
            name=f"SMA_{period}"
        )
        self.indicators.append(indicator)
        return indicator.name
    
    def add_ema(self, period: int = 20, column: str = 'close') -> str:
        """Add Exponential Moving Average indicator"""
        indicator = Indicator(
            type=IndicatorType.EMA,
            params={'period': period, 'column': column},
            name=f"EMA_{period}"
        )
        self.indicators.append(indicator)
        return indicator.name
    
    def add_rsi(self, period: int = 14) -> str:
        """Add RSI indicator"""
        indicator = Indicator(
            type=IndicatorType.RSI,
            params={'period': period},
            name=f"RSI_{period}"
        )
        self.indicators.append(indicator)
        return indicator.name
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> str:
        """Add MACD indicator"""
        indicator = Indicator(
            type=IndicatorType.MACD,
            params={'fast': fast, 'slow': slow, 'signal': signal},
            name=f"MACD_{fast}_{slow}"
        )
        self.indicators.append(indicator)
        return indicator.name
    
    def add_bollinger(self, period: int = 20, std_dev: float = 2.0) -> str:
        """Add Bollinger Bands"""
        for band in ['upper', 'middle', 'lower']:
            indicator = Indicator(
                type=IndicatorType.BOLLINGER,
                params={'period': period, 'std_dev': std_dev, 'band': band},
                name=f"BB_{band}_{period}"
            )
            self.indicators.append(indicator)
        return f"BB_middle_{period}"
    
    def add_atr(self, period: int = 14) -> str:
        """Add ATR indicator"""
        indicator = Indicator(
            type=IndicatorType.ATR,
            params={'period': period},
            name=f"ATR_{period}"
        )
        self.indicators.append(indicator)
        return indicator.name
    
    def add_stochastic(self, k_period: int = 14, d_period: int = 3) -> str:
        """Add Stochastic Oscillator"""
        for line in ['k', 'd']:
            indicator = Indicator(
                type=IndicatorType.STOCHASTIC,
                params={'k_period': k_period, 'd_period': d_period, 'line': line},
                name=f"STOCH_{line}_{k_period}"
            )
            self.indicators.append(indicator)
        return f"STOCH_k_{k_period}"
    
    def add_custom_indicator(self, name: str, formula: Callable[[pd.DataFrame], pd.Series]) -> str:
        """Add a custom indicator with user-defined formula"""
        # Store as custom type
        indicator = Indicator(
            type=IndicatorType.SMA,  # Placeholder
            params={'formula': formula, 'custom': True},
            name=name
        )
        indicator._custom_formula = formula
        self.indicators.append(indicator)
        return name
    
    def when(self, left: str, operator: str, right: Union[str, float]) -> 'ConditionBuilder':
        """Start building a condition"""
        return ConditionBuilder(self, left, operator, right)
    
    def add_entry_condition(self, left: str, operator: str, right: Union[str, float],
                           signal: SignalType = SignalType.BUY):
        """Add a simple entry condition"""
        op_map = {
            '>': ConditionOperator.GREATER_THAN,
            '<': ConditionOperator.LESS_THAN,
            '==': ConditionOperator.EQUAL,
            '>=': ConditionOperator.GREATER_EQUAL,
            '<=': ConditionOperator.LESS_EQUAL,
            'crosses_above': ConditionOperator.CROSSES_ABOVE,
            'crosses_below': ConditionOperator.CROSSES_BELOW,
        }
        condition = Condition(left, op_map[operator], right)
        rule = Rule([condition], signal=signal)
        self.entry_rules.append(rule)
    
    def add_exit_condition(self, left: str, operator: str, right: Union[str, float],
                          signal: SignalType = SignalType.SELL):
        """Add a simple exit condition"""
        op_map = {
            '>': ConditionOperator.GREATER_THAN,
            '<': ConditionOperator.LESS_THAN,
            '==': ConditionOperator.EQUAL,
            '>=': ConditionOperator.GREATER_EQUAL,
            '<=': ConditionOperator.LESS_EQUAL,
            'crosses_above': ConditionOperator.CROSSES_ABOVE,
            'crosses_below': ConditionOperator.CROSSES_BELOW,
        }
        condition = Condition(left, op_map[operator], right)
        rule = Rule([condition], signal=signal)
        self.exit_rules.append(rule)
    
    def set_stop_loss(self, percentage: float):
        """Set stop loss percentage"""
        self.risk_management.stop_loss = percentage
    
    def set_take_profit(self, percentage: float):
        """Set take profit percentage"""
        self.risk_management.take_profit = percentage
    
    def set_trailing_stop(self, percentage: float):
        """Set trailing stop percentage"""
        self.risk_management.trailing_stop = percentage
    
    def set_position_size(self, method: str = 'fixed', value: float = 0.1):
        """Set position sizing method"""
        self.risk_management.position_sizing_method = method
        self.risk_management.position_size_value = value
    
    def build(self) -> 'BuiltStrategy':
        """Build and return the strategy"""
        return BuiltStrategy(
            name=self.name,
            indicators=self.indicators,
            entry_rules=self.entry_rules,
            exit_rules=self.exit_rules,
            risk_management=self.risk_management,
            parameters=self.parameters
        )
    
    @staticmethod
    def from_dict(data: Dict) -> 'BuiltStrategy':
        """Create strategy from dictionary"""
        builder = StrategyBuilder(data['name'])
        
        # Add indicators
        for ind_data in data.get('indicators', []):
            indicator = Indicator(
                type=IndicatorType[ind_data['type']],
                params=ind_data['params'],
                name=ind_data['name']
            )
            builder.indicators.append(indicator)
        
        # Add rules
        for rule_data in data.get('entry_rules', []):
            conditions = [
                Condition(
                    c['left'],
                    ConditionOperator[c['operator']],
                    c['right']
                )
                for c in rule_data['conditions']
            ]
            rule = Rule(conditions, rule_data['logic'], SignalType[rule_data['signal']])
            builder.entry_rules.append(rule)
        
        for rule_data in data.get('exit_rules', []):
            conditions = [
                Condition(
                    c['left'],
                    ConditionOperator[c['operator']],
                    c['right']
                )
                for c in rule_data['conditions']
            ]
            rule = Rule(conditions, rule_data['logic'], SignalType[rule_data['signal']])
            builder.exit_rules.append(rule)
        
        return builder.build()


class ConditionBuilder:
    """Helper for building conditions fluently"""
    
    def __init__(self, strategy_builder: StrategyBuilder, left: str, operator: str, right: Union[str, float]):
        self.builder = strategy_builder
        self.conditions = []
        self._add_condition(left, operator, right)
        self.logic = "AND"
        self.signal = SignalType.BUY
    
    def _add_condition(self, left: str, operator: str, right: Union[str, float]):
        op_map = {
            '>': ConditionOperator.GREATER_THAN,
            '<': ConditionOperator.LESS_THAN,
            '==': ConditionOperator.EQUAL,
            '>=': ConditionOperator.GREATER_EQUAL,
            '<=': ConditionOperator.LESS_EQUAL,
            'crosses_above': ConditionOperator.CROSSES_ABOVE,
            'crosses_below': ConditionOperator.CROSSES_BELOW,
        }
        self.conditions.append(Condition(left, op_map[operator], right))
    
    def and_when(self, left: str, operator: str, right: Union[str, float]) -> 'ConditionBuilder':
        """Add AND condition"""
        self._add_condition(left, operator, right)
        self.logic = "AND"
        return self
    
    def or_when(self, left: str, operator: str, right: Union[str, float]) -> 'ConditionBuilder':
        """Add OR condition"""
        self._add_condition(left, operator, right)
        self.logic = "OR"
        return self
    
    def then_buy(self):
        """Create entry rule with BUY signal"""
        rule = Rule(self.conditions, self.logic, SignalType.BUY)
        self.builder.entry_rules.append(rule)
        return self.builder
    
    def then_sell(self):
        """Create entry rule with SELL signal"""
        rule = Rule(self.conditions, self.logic, SignalType.SELL)
        self.builder.exit_rules.append(rule)
        return self.builder
    
    def then_exit(self):
        """Create exit rule"""
        rule = Rule(self.conditions, self.logic, SignalType.SELL)
        self.builder.exit_rules.append(rule)
        return self.builder


@dataclass
class RiskManagement:
    """Risk management settings"""
    stop_loss: Optional[float] = None  # Percentage
    take_profit: Optional[float] = None  # Percentage
    trailing_stop: Optional[float] = None  # Percentage
    max_position_size: float = 0.25  # Max % of portfolio
    max_positions: int = 10
    position_sizing_method: str = 'fixed'  # 'fixed', 'kelly', 'volatility'
    position_size_value: float = 0.1


class BuiltStrategy:
    """A fully constructed strategy ready for backtesting"""
    
    def __init__(self, name: str, indicators: List[Indicator], 
                 entry_rules: List[Rule], exit_rules: List[Rule],
                 risk_management: RiskManagement, parameters: Dict):
        self.name = name
        self.indicators = indicators
        self.entry_rules = entry_rules
        self.exit_rules = exit_rules
        self.risk_management = risk_management
        self.parameters = parameters
        self.calculator = IndicatorCalculator()
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators"""
        df = data.copy()
        
        for indicator in self.indicators:
            if hasattr(indicator, '_custom_formula'):
                df[indicator.name] = indicator._custom_formula(df)
            else:
                df[indicator.name] = self.calculator.calculate(df, indicator)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = self.prepare_data(data)
        df['signal'] = 0
        df['position'] = 0
        
        position = 0
        entry_price = 0
        
        for idx in range(len(df)):
            current_price = df.iloc[idx]['close']
            
            # Check risk management exits first
            if position != 0 and entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)
                
                # Stop loss
                if self.risk_management.stop_loss and pnl_pct < -self.risk_management.stop_loss:
                    df.iloc[idx, df.columns.get_loc('signal')] = -np.sign(position)
                    position = 0
                    continue
                
                # Take profit
                if self.risk_management.take_profit and pnl_pct > self.risk_management.take_profit:
                    df.iloc[idx, df.columns.get_loc('signal')] = -np.sign(position)
                    position = 0
                    continue
            
            # Check entry rules
            if position == 0:
                for rule in self.entry_rules:
                    signal = rule.evaluate(df, idx)
                    if signal:
                        df.iloc[idx, df.columns.get_loc('signal')] = signal.value
                        position = signal.value
                        entry_price = current_price
                        break
            
            # Check exit rules
            else:
                for rule in self.exit_rules:
                    signal = rule.evaluate(df, idx)
                    if signal:
                        df.iloc[idx, df.columns.get_loc('signal')] = signal.value
                        position = 0
                        break
            
            df.iloc[idx, df.columns.get_loc('position')] = position
        
        return df
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'indicators': [
                {'type': i.type.name, 'params': i.params, 'name': i.name}
                for i in self.indicators
            ],
            'entry_rules': self._serialize_rules(self.entry_rules),
            'exit_rules': self._serialize_rules(self.exit_rules),
            'risk_management': {
                'stop_loss': self.risk_management.stop_loss,
                'take_profit': self.risk_management.take_profit,
                'trailing_stop': self.risk_management.trailing_stop,
                'max_position_size': self.risk_management.max_position_size,
                'position_sizing_method': self.risk_management.position_sizing_method
            }
        }
    
    def _serialize_rules(self, rules: List[Rule]) -> List[Dict]:
        return [
            {
                'conditions': [
                    {'left': c.left_operand, 'operator': c.operator.name, 'right': c.right_operand}
                    for c in r.conditions
                ],
                'logic': r.logic,
                'signal': r.signal.name
            }
            for r in rules
        ]
    
    def save(self, filepath: str):
        """Save strategy to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# STRATEGY TEMPLATES
# ============================================================================

class StrategyTemplates:
    """Pre-built strategy templates for common trading approaches"""
    
    @staticmethod
    def golden_cross() -> BuiltStrategy:
        """
        Golden Cross Strategy
        ---------------------
        Buy when 50-day SMA crosses above 200-day SMA
        Sell when 50-day SMA crosses below 200-day SMA
        """
        builder = StrategyBuilder("Golden Cross")
        
        sma_50 = builder.add_sma(50)
        sma_200 = builder.add_sma(200)
        
        builder.when(sma_50, 'crosses_above', sma_200).then_buy()
        builder.when(sma_50, 'crosses_below', sma_200).then_sell()
        
        builder.set_stop_loss(0.05)
        
        return builder.build()
    
    @staticmethod
    def rsi_mean_reversion() -> BuiltStrategy:
        """
        RSI Mean Reversion Strategy
        ---------------------------
        Buy when RSI < 30 (oversold)
        Sell when RSI > 70 (overbought)
        """
        builder = StrategyBuilder("RSI Mean Reversion")
        
        rsi = builder.add_rsi(14)
        
        builder.add_entry_condition(rsi, '<', 30, SignalType.BUY)
        builder.add_exit_condition(rsi, '>', 70, SignalType.SELL)
        
        builder.set_stop_loss(0.03)
        builder.set_take_profit(0.06)
        
        return builder.build()
    
    @staticmethod
    def bollinger_breakout() -> BuiltStrategy:
        """
        Bollinger Band Breakout Strategy
        --------------------------------
        Buy when price breaks above upper band
        Sell when price breaks below lower band
        """
        builder = StrategyBuilder("Bollinger Breakout")
        
        builder.add_bollinger(20, 2.0)
        
        builder.add_entry_condition('close', '>', 'BB_upper_20', SignalType.BUY)
        builder.add_exit_condition('close', '<', 'BB_lower_20', SignalType.SELL)
        
        builder.set_trailing_stop(0.02)
        
        return builder.build()
    
    @staticmethod
    def macd_momentum() -> BuiltStrategy:
        """
        MACD Momentum Strategy
        ----------------------
        Buy when MACD crosses above signal line
        Sell when MACD crosses below signal line
        """
        builder = StrategyBuilder("MACD Momentum")
        
        macd = builder.add_macd(12, 26, 9)
        
        # Add signal line indicator manually
        signal_ind = Indicator(
            type=IndicatorType.MACD,
            params={'fast': 12, 'slow': 26, 'signal': 9, 'line': 'signal'},
            name='MACD_signal'
        )
        builder.indicators.append(signal_ind)
        
        builder.when(macd, 'crosses_above', 'MACD_signal').then_buy()
        builder.when(macd, 'crosses_below', 'MACD_signal').then_sell()
        
        builder.set_stop_loss(0.04)
        
        return builder.build()
    
    @staticmethod
    def dual_momentum() -> BuiltStrategy:
        """
        Dual Momentum Strategy
        ----------------------
        Combines absolute and relative momentum
        """
        builder = StrategyBuilder("Dual Momentum")
        
        ema_20 = builder.add_ema(20)
        ema_50 = builder.add_ema(50)
        rsi = builder.add_rsi(14)
        
        # Entry: EMA 20 > EMA 50 AND RSI > 50
        builder.when(ema_20, '>', ema_50).and_when(rsi, '>', 50).then_buy()
        
        # Exit: EMA 20 < EMA 50 OR RSI < 40
        builder.when(ema_20, '<', ema_50).then_sell()
        builder.add_exit_condition(rsi, '<', 40, SignalType.SELL)
        
        builder.set_stop_loss(0.05)
        builder.set_take_profit(0.15)
        
        return builder.build()
    
    @staticmethod
    def volatility_breakout() -> BuiltStrategy:
        """
        Volatility Breakout Strategy
        ----------------------------
        Buy when price moves beyond ATR threshold
        """
        builder = StrategyBuilder("Volatility Breakout")
        
        builder.add_atr(14)
        ema = builder.add_ema(20)
        
        # Custom indicator: EMA + 2*ATR
        def upper_band(df):
            atr = df['ATR_14']
            ema = df['EMA_20']
            return ema + 2 * atr
        
        def lower_band(df):
            atr = df['ATR_14']
            ema = df['EMA_20']
            return ema - 2 * atr
        
        builder.add_custom_indicator('volatility_upper', upper_band)
        builder.add_custom_indicator('volatility_lower', lower_band)
        
        builder.add_entry_condition('close', '>', 'volatility_upper', SignalType.BUY)
        builder.add_exit_condition('close', '<', 'volatility_lower', SignalType.SELL)
        
        builder.set_trailing_stop(0.03)
        
        return builder.build()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Build a custom strategy
    print("=" * 60)
    print("STRATEGY BUILDER DEMO")
    print("=" * 60)
    
    # Create a custom strategy
    builder = StrategyBuilder("My Custom Strategy")
    
    # Add indicators
    sma_fast = builder.add_sma(10)
    sma_slow = builder.add_sma(30)
    rsi = builder.add_rsi(14)
    
    # Add entry rules
    builder.when(sma_fast, 'crosses_above', sma_slow) \
           .and_when(rsi, '>', 50) \
           .then_buy()
    
    # Add exit rules
    builder.when(sma_fast, 'crosses_below', sma_slow).then_sell()
    builder.add_exit_condition(rsi, '<', 30, SignalType.SELL)
    
    # Set risk management
    builder.set_stop_loss(0.05)
    builder.set_take_profit(0.10)
    
    # Build the strategy
    strategy = builder.build()
    
    print(f"\nBuilt Strategy: {strategy.name}")
    print(f"Indicators: {[i.name for i in strategy.indicators]}")
    print(f"Entry Rules: {len(strategy.entry_rules)}")
    print(f"Exit Rules: {len(strategy.exit_rules)}")
    print(f"Stop Loss: {strategy.risk_management.stop_loss}")
    print(f"Take Profit: {strategy.risk_management.take_profit}")
    
    # Use a template
    print("\n" + "=" * 60)
    print("TEMPLATE STRATEGIES")
    print("=" * 60)
    
    templates = [
        StrategyTemplates.golden_cross(),
        StrategyTemplates.rsi_mean_reversion(),
        StrategyTemplates.bollinger_breakout(),
        StrategyTemplates.macd_momentum(),
        StrategyTemplates.dual_momentum()
    ]
    
    for t in templates:
        print(f"\n{t.name}:")
        print(f"  - Indicators: {len(t.indicators)}")
        print(f"  - Entry Rules: {len(t.entry_rules)}")
        print(f"  - Exit Rules: {len(t.exit_rules)}")
