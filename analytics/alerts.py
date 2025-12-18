"""
Real-Time Alerts System
=======================

Institutional-grade alerting system for:
- Price alerts (breakouts, support/resistance)
- Technical indicator alerts
- Volume spikes
- Volatility alerts
- News sentiment alerts
- Portfolio risk alerts
- Options flow alerts

Multi-channel notifications (console, email, webhook, Slack).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import queue
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AlertSystem')


class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertType(Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CHANGE_PCT = "price_change_pct"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY = "volatility"
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"
    MACD_CROSS = "macd_cross"
    MOVING_AVG_CROSS = "moving_avg_cross"
    SUPPORT_BREAK = "support_break"
    RESISTANCE_BREAK = "resistance_break"
    SENTIMENT = "sentiment"
    PORTFOLIO_RISK = "portfolio_risk"
    DRAWDOWN = "drawdown"
    OPTIONS_FLOW = "options_flow"
    EARNINGS = "earnings"
    NEWS = "news"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Represents an alert configuration."""
    id: str
    name: str
    alert_type: AlertType
    symbol: str
    condition: Dict[str, Any]
    priority: AlertPriority = AlertPriority.MEDIUM
    enabled: bool = True
    triggered: bool = False
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 15
    channels: List[str] = field(default_factory=lambda: ['console'])
    metadata: Dict = field(default_factory=dict)
    
    def can_trigger(self) -> bool:
        """Check if alert can trigger (respects cooldown)."""
        if not self.enabled:
            return False
        if self.last_triggered is None:
            return True
        elapsed = (datetime.now() - self.last_triggered).total_seconds() / 60
        return elapsed >= self.cooldown_minutes


@dataclass
class AlertEvent:
    """Represents a triggered alert event."""
    alert: Alert
    timestamp: datetime
    current_value: Any
    message: str
    data: Dict = field(default_factory=dict)


class AlertChannel:
    """Base class for notification channels."""
    
    def send(self, event: AlertEvent) -> bool:
        """Send alert notification. Override in subclass."""
        raise NotImplementedError


class ConsoleChannel(AlertChannel):
    """Console/terminal output channel."""
    
    PRIORITY_COLORS = {
        AlertPriority.LOW: '\033[92m',      # Green
        AlertPriority.MEDIUM: '\033[93m',   # Yellow
        AlertPriority.HIGH: '\033[91m',     # Red
        AlertPriority.CRITICAL: '\033[95m'  # Magenta
    }
    RESET = '\033[0m'
    
    def send(self, event: AlertEvent) -> bool:
        color = self.PRIORITY_COLORS.get(event.alert.priority, '')
        priority_name = event.alert.priority.name
        
        print(f"\n{color}{'='*60}")
        print(f"🚨 ALERT [{priority_name}] - {event.alert.name}")
        print(f"{'='*60}{self.RESET}")
        print(f"Symbol: {event.alert.symbol}")
        print(f"Type: {event.alert.alert_type.value}")
        print(f"Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Value: {event.current_value}")
        print(f"Message: {event.message}")
        print(f"{color}{'='*60}{self.RESET}\n")
        
        return True


class EmailChannel(AlertChannel):
    """Email notification channel."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def send(self, event: AlertEvent) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{event.alert.priority.name}] Trading Alert: {event.alert.name}"
            
            body = f"""
Trading Alert Triggered

Alert: {event.alert.name}
Symbol: {event.alert.symbol}
Type: {event.alert.alert_type.value}
Priority: {event.alert.priority.name}
Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Current Value: {event.current_value}

Message: {event.message}

Additional Data: {json.dumps(event.data, indent=2, default=str)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class WebhookChannel(AlertChannel):
    """Webhook notification channel (for Slack, Discord, custom services)."""
    
    def __init__(self, webhook_url: str, headers: Dict = None):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for webhooks")
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    def send(self, event: AlertEvent) -> bool:
        try:
            payload = {
                'alert_name': event.alert.name,
                'symbol': event.alert.symbol,
                'type': event.alert.alert_type.value,
                'priority': event.alert.priority.name,
                'timestamp': event.timestamp.isoformat(),
                'value': str(event.current_value),
                'message': event.message,
                'data': event.data
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False


class SlackChannel(WebhookChannel):
    """Slack-specific webhook channel."""
    
    def send(self, event: AlertEvent) -> bool:
        try:
            # Slack-formatted message
            priority_emoji = {
                AlertPriority.LOW: '🟢',
                AlertPriority.MEDIUM: '🟡',
                AlertPriority.HIGH: '🔴',
                AlertPriority.CRITICAL: '🚨'
            }
            
            emoji = priority_emoji.get(event.alert.priority, '⚪')
            
            payload = {
                'blocks': [
                    {
                        'type': 'header',
                        'text': {
                            'type': 'plain_text',
                            'text': f"{emoji} Trading Alert: {event.alert.name}"
                        }
                    },
                    {
                        'type': 'section',
                        'fields': [
                            {'type': 'mrkdwn', 'text': f"*Symbol:*\n{event.alert.symbol}"},
                            {'type': 'mrkdwn', 'text': f"*Priority:*\n{event.alert.priority.name}"},
                            {'type': 'mrkdwn', 'text': f"*Type:*\n{event.alert.alert_type.value}"},
                            {'type': 'mrkdwn', 'text': f"*Value:*\n{event.current_value}"}
                        ]
                    },
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': f"*Message:*\n{event.message}"
                        }
                    },
                    {
                        'type': 'context',
                        'elements': [
                            {
                                'type': 'mrkdwn',
                                'text': f"Triggered at {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False


class AlertConditionEvaluator:
    """Evaluate alert conditions against market data."""
    
    @staticmethod
    def evaluate_price_above(data: Dict, condition: Dict) -> tuple:
        """Check if price is above threshold."""
        current_price = data.get('price', 0)
        threshold = condition.get('threshold', 0)
        
        if current_price > threshold:
            return True, f"Price ${current_price:.2f} crossed above ${threshold:.2f}"
        return False, None
    
    @staticmethod
    def evaluate_price_below(data: Dict, condition: Dict) -> tuple:
        """Check if price is below threshold."""
        current_price = data.get('price', 0)
        threshold = condition.get('threshold', 0)
        
        if current_price < threshold:
            return True, f"Price ${current_price:.2f} dropped below ${threshold:.2f}"
        return False, None
    
    @staticmethod
    def evaluate_price_change_pct(data: Dict, condition: Dict) -> tuple:
        """Check if price changed by percentage."""
        change_pct = data.get('change_pct', 0)
        threshold = condition.get('threshold', 0)
        direction = condition.get('direction', 'any')
        
        if direction == 'up' and change_pct > threshold:
            return True, f"Price increased {change_pct:.2f}% (threshold: {threshold}%)"
        elif direction == 'down' and change_pct < -threshold:
            return True, f"Price decreased {change_pct:.2f}% (threshold: -{threshold}%)"
        elif direction == 'any' and abs(change_pct) > threshold:
            return True, f"Price changed {change_pct:.2f}% (threshold: ±{threshold}%)"
        return False, None
    
    @staticmethod
    def evaluate_volume_spike(data: Dict, condition: Dict) -> tuple:
        """Check for volume spike."""
        current_volume = data.get('volume', 0)
        avg_volume = data.get('avg_volume', current_volume)
        multiplier = condition.get('multiplier', 2.0)
        
        if avg_volume > 0 and current_volume > avg_volume * multiplier:
            ratio = current_volume / avg_volume
            return True, f"Volume spike: {ratio:.1f}x average volume"
        return False, None
    
    @staticmethod
    def evaluate_rsi(data: Dict, condition: Dict, alert_type: AlertType) -> tuple:
        """Check RSI overbought/oversold."""
        rsi = data.get('rsi', 50)
        
        if alert_type == AlertType.RSI_OVERBOUGHT:
            threshold = condition.get('threshold', 70)
            if rsi > threshold:
                return True, f"RSI is overbought at {rsi:.1f} (threshold: {threshold})"
        else:  # RSI_OVERSOLD
            threshold = condition.get('threshold', 30)
            if rsi < threshold:
                return True, f"RSI is oversold at {rsi:.1f} (threshold: {threshold})"
        
        return False, None
    
    @staticmethod
    def evaluate_macd_cross(data: Dict, condition: Dict) -> tuple:
        """Check MACD crossover."""
        macd = data.get('macd', 0)
        signal = data.get('macd_signal', 0)
        prev_macd = data.get('prev_macd', macd)
        prev_signal = data.get('prev_macd_signal', signal)
        
        direction = condition.get('direction', 'bullish')
        
        if direction == 'bullish':
            if prev_macd <= prev_signal and macd > signal:
                return True, f"Bullish MACD crossover detected"
        else:
            if prev_macd >= prev_signal and macd < signal:
                return True, f"Bearish MACD crossover detected"
        
        return False, None
    
    @staticmethod
    def evaluate_moving_avg_cross(data: Dict, condition: Dict) -> tuple:
        """Check moving average crossover."""
        fast_ma = data.get('fast_ma', 0)
        slow_ma = data.get('slow_ma', 0)
        prev_fast = data.get('prev_fast_ma', fast_ma)
        prev_slow = data.get('prev_slow_ma', slow_ma)
        
        direction = condition.get('direction', 'golden')
        
        if direction == 'golden':  # Fast crosses above slow
            if prev_fast <= prev_slow and fast_ma > slow_ma:
                return True, f"Golden Cross: Fast MA crossed above Slow MA"
        else:  # Death cross
            if prev_fast >= prev_slow and fast_ma < slow_ma:
                return True, f"Death Cross: Fast MA crossed below Slow MA"
        
        return False, None
    
    @staticmethod
    def evaluate_volatility(data: Dict, condition: Dict) -> tuple:
        """Check volatility threshold."""
        volatility = data.get('volatility', 0)
        threshold = condition.get('threshold', 0.3)
        direction = condition.get('direction', 'above')
        
        if direction == 'above' and volatility > threshold:
            return True, f"Volatility spiked to {volatility*100:.1f}% (threshold: {threshold*100:.1f}%)"
        elif direction == 'below' and volatility < threshold:
            return True, f"Volatility dropped to {volatility*100:.1f}% (threshold: {threshold*100:.1f}%)"
        
        return False, None
    
    @staticmethod
    def evaluate_drawdown(data: Dict, condition: Dict) -> tuple:
        """Check portfolio drawdown."""
        drawdown = data.get('drawdown', 0)
        threshold = condition.get('threshold', 0.1)
        
        if drawdown > threshold:
            return True, f"Drawdown alert: {drawdown*100:.1f}% (threshold: {threshold*100:.1f}%)"
        return False, None
    
    @classmethod
    def evaluate(cls, alert: Alert, data: Dict) -> tuple:
        """Evaluate alert condition."""
        evaluators = {
            AlertType.PRICE_ABOVE: cls.evaluate_price_above,
            AlertType.PRICE_BELOW: cls.evaluate_price_below,
            AlertType.PRICE_CHANGE_PCT: cls.evaluate_price_change_pct,
            AlertType.VOLUME_SPIKE: cls.evaluate_volume_spike,
            AlertType.RSI_OVERBOUGHT: lambda d, c: cls.evaluate_rsi(d, c, AlertType.RSI_OVERBOUGHT),
            AlertType.RSI_OVERSOLD: lambda d, c: cls.evaluate_rsi(d, c, AlertType.RSI_OVERSOLD),
            AlertType.MACD_CROSS: cls.evaluate_macd_cross,
            AlertType.MOVING_AVG_CROSS: cls.evaluate_moving_avg_cross,
            AlertType.VOLATILITY: cls.evaluate_volatility,
            AlertType.DRAWDOWN: cls.evaluate_drawdown,
        }
        
        evaluator = evaluators.get(alert.alert_type)
        if evaluator:
            return evaluator(data, alert.condition)
        
        return False, None


class AlertManager:
    """
    Main alert management system.
    Manages alerts, evaluates conditions, and dispatches notifications.
    """
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.channels: Dict[str, AlertChannel] = {
            'console': ConsoleChannel()
        }
        self.event_queue = queue.Queue()
        self.alert_history: List[AlertEvent] = []
        self.running = False
        self._worker_thread = None
        
        # Statistics
        self.stats = defaultdict(int)
    
    def add_channel(self, name: str, channel: AlertChannel):
        """Add a notification channel."""
        self.channels[name] = channel
        logger.info(f"Added notification channel: {name}")
    
    def create_alert(self, name: str, alert_type: AlertType, symbol: str,
                     condition: Dict, priority: AlertPriority = AlertPriority.MEDIUM,
                     channels: List[str] = None, cooldown_minutes: int = 15,
                     **kwargs) -> Alert:
        """Create and register a new alert."""
        alert_id = f"{symbol}_{alert_type.value}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            name=name,
            alert_type=alert_type,
            symbol=symbol,
            condition=condition,
            priority=priority,
            channels=channels or ['console'],
            cooldown_minutes=cooldown_minutes,
            **kwargs
        )
        
        self.alerts[alert_id] = alert
        logger.info(f"Created alert: {name} ({alert_id})")
        
        return alert
    
    def remove_alert(self, alert_id: str):
        """Remove an alert."""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Removed alert: {alert_id}")
    
    def enable_alert(self, alert_id: str):
        """Enable an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].enabled = True
    
    def disable_alert(self, alert_id: str):
        """Disable an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].enabled = False
    
    def check_alerts(self, symbol: str, data: Dict):
        """Check all alerts for a symbol against current data."""
        for alert in self.alerts.values():
            if alert.symbol != symbol:
                continue
            
            if not alert.can_trigger():
                continue
            
            triggered, message = AlertConditionEvaluator.evaluate(alert, data)
            
            if triggered:
                event = AlertEvent(
                    alert=alert,
                    timestamp=datetime.now(),
                    current_value=data.get('price', data.get('value', 'N/A')),
                    message=message,
                    data=data
                )
                
                self._trigger_alert(alert, event)
    
    def _trigger_alert(self, alert: Alert, event: AlertEvent):
        """Handle triggered alert."""
        alert.triggered = True
        alert.last_triggered = event.timestamp
        
        self.alert_history.append(event)
        self.stats['total_triggered'] += 1
        self.stats[f'{alert.alert_type.value}_triggered'] += 1
        
        # Send to all configured channels
        for channel_name in alert.channels:
            if channel_name in self.channels:
                try:
                    self.channels[channel_name].send(event)
                    self.stats[f'{channel_name}_sent'] += 1
                except Exception as e:
                    logger.error(f"Failed to send to {channel_name}: {e}")
                    self.stats[f'{channel_name}_failed'] += 1
    
    def get_alert_history(self, symbol: str = None, 
                          limit: int = 100) -> List[AlertEvent]:
        """Get alert history, optionally filtered by symbol."""
        history = self.alert_history
        
        if symbol:
            history = [e for e in history if e.alert.symbol == symbol]
        
        return history[-limit:]
    
    def get_active_alerts(self, symbol: str = None) -> List[Alert]:
        """Get all active (enabled) alerts."""
        alerts = [a for a in self.alerts.values() if a.enabled]
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        return alerts
    
    def get_statistics(self) -> Dict:
        """Get alert system statistics."""
        return dict(self.stats)
    
    # Quick alert creation methods
    def price_alert(self, symbol: str, threshold: float, 
                    direction: str = 'above', **kwargs) -> Alert:
        """Create a price alert."""
        alert_type = AlertType.PRICE_ABOVE if direction == 'above' else AlertType.PRICE_BELOW
        name = f"{symbol} price {direction} ${threshold}"
        
        return self.create_alert(
            name=name,
            alert_type=alert_type,
            symbol=symbol,
            condition={'threshold': threshold},
            **kwargs
        )
    
    def volume_alert(self, symbol: str, multiplier: float = 2.0, **kwargs) -> Alert:
        """Create a volume spike alert."""
        return self.create_alert(
            name=f"{symbol} volume spike ({multiplier}x)",
            alert_type=AlertType.VOLUME_SPIKE,
            symbol=symbol,
            condition={'multiplier': multiplier},
            priority=AlertPriority.HIGH,
            **kwargs
        )
    
    def rsi_alert(self, symbol: str, overbought: float = 70, 
                  oversold: float = 30, **kwargs) -> List[Alert]:
        """Create RSI overbought/oversold alerts."""
        alerts = []
        
        alerts.append(self.create_alert(
            name=f"{symbol} RSI overbought",
            alert_type=AlertType.RSI_OVERBOUGHT,
            symbol=symbol,
            condition={'threshold': overbought},
            **kwargs
        ))
        
        alerts.append(self.create_alert(
            name=f"{symbol} RSI oversold",
            alert_type=AlertType.RSI_OVERSOLD,
            symbol=symbol,
            condition={'threshold': oversold},
            **kwargs
        ))
        
        return alerts
    
    def macd_alert(self, symbol: str, **kwargs) -> List[Alert]:
        """Create MACD crossover alerts."""
        alerts = []
        
        alerts.append(self.create_alert(
            name=f"{symbol} bullish MACD cross",
            alert_type=AlertType.MACD_CROSS,
            symbol=symbol,
            condition={'direction': 'bullish'},
            priority=AlertPriority.HIGH,
            **kwargs
        ))
        
        alerts.append(self.create_alert(
            name=f"{symbol} bearish MACD cross",
            alert_type=AlertType.MACD_CROSS,
            symbol=symbol,
            condition={'direction': 'bearish'},
            priority=AlertPriority.HIGH,
            **kwargs
        ))
        
        return alerts
    
    def drawdown_alert(self, threshold: float = 0.1, **kwargs) -> Alert:
        """Create portfolio drawdown alert."""
        return self.create_alert(
            name=f"Portfolio drawdown >{threshold*100}%",
            alert_type=AlertType.DRAWDOWN,
            symbol='PORTFOLIO',
            condition={'threshold': threshold},
            priority=AlertPriority.CRITICAL,
            **kwargs
        )


class RealTimeMonitor:
    """
    Real-time market data monitor that feeds the alert system.
    """
    
    def __init__(self, alert_manager: AlertManager, update_interval: int = 60):
        self.alert_manager = alert_manager
        self.update_interval = update_interval
        self.symbols: List[str] = []
        self.running = False
        self._thread = None
        self._data_fetcher = None
    
    def set_data_fetcher(self, fetcher: Callable):
        """Set the data fetcher function."""
        self._data_fetcher = fetcher
    
    def add_symbol(self, symbol: str):
        """Add a symbol to monitor."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from monitoring."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
    
    def start(self):
        """Start real-time monitoring."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Real-time monitor started")
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Real-time monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            for symbol in self.symbols:
                try:
                    if self._data_fetcher:
                        data = self._data_fetcher(symbol)
                        if data:
                            self.alert_manager.check_alerts(symbol, data)
                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {e}")
            
            time.sleep(self.update_interval)
    
    def check_now(self, symbol: str, data: Dict):
        """Manually trigger an alert check."""
        self.alert_manager.check_alerts(symbol, data)


# Pre-built alert templates
class AlertTemplates:
    """Common alert configurations."""
    
    @staticmethod
    def day_trader_alerts(manager: AlertManager, symbol: str):
        """Set up alerts for day trading."""
        manager.rsi_alert(symbol, overbought=70, oversold=30)
        manager.macd_alert(symbol)
        manager.volume_alert(symbol, multiplier=2.0)
        manager.create_alert(
            name=f"{symbol} 2% move",
            alert_type=AlertType.PRICE_CHANGE_PCT,
            symbol=symbol,
            condition={'threshold': 2.0, 'direction': 'any'},
            priority=AlertPriority.HIGH
        )
    
    @staticmethod
    def swing_trader_alerts(manager: AlertManager, symbol: str):
        """Set up alerts for swing trading."""
        manager.create_alert(
            name=f"{symbol} golden cross",
            alert_type=AlertType.MOVING_AVG_CROSS,
            symbol=symbol,
            condition={'direction': 'golden'},
            priority=AlertPriority.HIGH,
            cooldown_minutes=1440  # 1 day
        )
        manager.create_alert(
            name=f"{symbol} death cross",
            alert_type=AlertType.MOVING_AVG_CROSS,
            symbol=symbol,
            condition={'direction': 'death'},
            priority=AlertPriority.HIGH,
            cooldown_minutes=1440
        )
    
    @staticmethod
    def risk_management_alerts(manager: AlertManager):
        """Set up portfolio risk alerts."""
        manager.drawdown_alert(threshold=0.05)  # 5%
        manager.drawdown_alert(threshold=0.10)  # 10%
        manager.create_alert(
            name="Volatility spike",
            alert_type=AlertType.VOLATILITY,
            symbol='PORTFOLIO',
            condition={'threshold': 0.3, 'direction': 'above'},
            priority=AlertPriority.HIGH
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Real-Time Alerts System")
    print("=" * 60)
    
    # Create alert manager
    manager = AlertManager()
    
    # Create some test alerts
    print("\n--- Creating Alerts ---")
    
    manager.price_alert('AAPL', 200, 'above', priority=AlertPriority.HIGH)
    manager.price_alert('AAPL', 150, 'below', priority=AlertPriority.CRITICAL)
    manager.volume_alert('AAPL', multiplier=2.5)
    manager.rsi_alert('AAPL')
    manager.macd_alert('TSLA')
    manager.drawdown_alert(0.1)
    
    print(f"Created {len(manager.alerts)} alerts")
    
    # Test alert triggering
    print("\n--- Testing Alert Triggers ---")
    
    # Simulate market data
    test_data = [
        {'symbol': 'AAPL', 'price': 205, 'change_pct': 3.5, 'volume': 150000000, 
         'avg_volume': 50000000, 'rsi': 75},
        {'symbol': 'AAPL', 'price': 145, 'change_pct': -5.0, 'rsi': 25},
        {'symbol': 'TSLA', 'macd': 1.5, 'macd_signal': 1.0, 
         'prev_macd': 0.8, 'prev_macd_signal': 1.0},
        {'symbol': 'PORTFOLIO', 'drawdown': 0.12}
    ]
    
    for data in test_data:
        symbol = data.get('symbol', 'UNKNOWN')
        manager.check_alerts(symbol, data)
    
    # Show statistics
    print("\n--- Alert Statistics ---")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Show alert history
    print("\n--- Alert History ---")
    history = manager.get_alert_history(limit=5)
    for event in history:
        print(f"  [{event.timestamp.strftime('%H:%M:%S')}] {event.alert.name}")
    
    print("\n" + "=" * 60)
    print("Alert System Demo Complete!")
    print("=" * 60)
