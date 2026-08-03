"""
Bloomberg Terminal - Quick Start Runner
========================================
One-click setup and run script for the trading terminal.

Usage:
    python run.py              # Run the dashboard
    python run.py --setup      # Setup environment first
    python run.py --check      # Check dependencies only
    python run.py --demo       # Run demo/examples
    python run.py --agents     # Run the agent pipeline
    python run.py --agents --continuous  # Run agents continuously
"""

import subprocess
import sys
import os
from pathlib import Path
from core.logging_config import setup_logging

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header():
    print(f"""
{Colors.BLUE}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║            BLOOMBERG TERMINAL - TRADING PLATFORM             ║
║                    Institutional-Grade Trading               ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}
""")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.9+ required, found {version.major}.{version.minor}")
        return False

def check_core_dependencies():
    """Check if core dependencies are installed."""
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'yfinance': 'yfinance',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'ta': 'ta',
        'scipy': 'scipy'
    }
    
    missing = []
    installed = []
    
    for name, package in required.items():
        try:
            __import__(package)
            installed.append(name)
        except ImportError:
            missing.append(name)
    
    if installed:
        print_success(f"Core packages: {', '.join(installed)}")
    
    if missing:
        print_warning(f"Missing packages: {', '.join(missing)}")
        return False, missing
    
    return True, []

def check_optional_dependencies():
    """Check optional dependencies."""
    optional = {
        'scikit-learn': 'sklearn',
        'torch': 'torch',
        'transformers': 'transformers',
        'redis': 'redis',
        'numba': 'numba'
    }
    
    available = []
    for name, package in optional.items():
        try:
            __import__(package)
            available.append(name)
        except ImportError:
            pass
    
    if available:
        print_info(f"Optional packages available: {', '.join(available)}")

def install_dependencies(requirements_file='requirements-core.txt'):
    """Install dependencies from requirements file."""
    print_info(f"Installing dependencies from {requirements_file}...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '-r', requirements_file, 
            '--quiet', '--disable-pip-version-check'
        ])
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def install_missing(packages):
    """Install specific missing packages."""
    if not packages:
        return True
    
    print_info(f"Installing missing packages: {', '.join(packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            *packages,
            '--quiet', '--disable-pip-version-check'
        ])
        print_success("Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install packages: {e}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / 'dashboard' / 'app_streamlit.py'
    
    if not dashboard_path.exists():
        # Try alternative paths
        alt_paths = [
            Path(__file__).parent / 'dashboard' / 'app.py',
            Path(__file__).parent / 'dashboard' / 'main.py',
            Path(__file__).parent / 'app.py',
        ]
        for path in alt_paths:
            if path.exists():
                dashboard_path = path
                break
        else:
            print_warning("Dashboard file not found. Creating a simple demo...")
            create_demo_dashboard()
            dashboard_path = Path(__file__).parent / 'dashboard' / 'app.py'
    
    print_info(f"Starting dashboard: {dashboard_path}")
    print_info("Opening browser at http://localhost:8501")
    print_info("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(dashboard_path),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n")
        print_info("Dashboard stopped")

def create_demo_dashboard():
    """Create a simple demo dashboard if none exists."""
    dashboard_dir = Path(__file__).parent / 'dashboard'
    dashboard_dir.mkdir(exist_ok=True)
    
    demo_code = '''"""
Bloomberg Terminal - Trading Dashboard
======================================
"""

import streamlit as st
import pandas as pd
import numpy as np

# Try to import data sources
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Trading Terminal",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Trading Terminal")
st.markdown("---")

# Sidebar
st.sidebar.title("Settings")
symbol = st.sidebar.text_input("Symbol", value="AAPL")
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"])

# Main content
if YFINANCE_AVAILABLE:
    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        info = ticker.info
        
        if len(data) > 0:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data["Close"].iloc[-1]
            prev_price = data["Close"].iloc[-2] if len(data) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            col1.metric("Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
            col2.metric("High", f"${data['High'].iloc[-1]:.2f}")
            col3.metric("Low", f"${data['Low'].iloc[-1]:.2f}")
            col4.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            
            st.markdown("---")
            
            # Chart
            if PLOTLY_AVAILABLE:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.03,
                                   row_heights=[0.7, 0.3])
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name="OHLC"
                ), row=1, col=1)
                
                # Volume
                colors = ["red" if row["Open"] > row["Close"] else "green" 
                         for _, row in data.iterrows()]
                fig.add_trace(go.Bar(
                    x=data.index,
                    y=data["Volume"],
                    marker_color=colors,
                    name="Volume"
                ), row=2, col=1)
                
                fig.update_layout(
                    title=f"{symbol} - Price Chart",
                    xaxis_rangeslider_visible=False,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(data["Close"])
            
            # Data table
            with st.expander("📋 Raw Data"):
                st.dataframe(data.tail(20))
            
            # Company info
            with st.expander("ℹ️ Company Info"):
                if "longName" in info:
                    st.write(f"**{info.get('longName', symbol)}**")
                if "sector" in info:
                    st.write(f"Sector: {info.get('sector')}")
                if "industry" in info:
                    st.write(f"Industry: {info.get('industry')}")
                if "marketCap" in info:
                    st.write(f"Market Cap: ${info.get('marketCap', 0):,.0f}")
        else:
            st.error(f"No data found for {symbol}")
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.warning("yfinance not installed. Run: pip install yfinance")
    
    # Demo with random data
    st.subheader("Demo Mode (Random Data)")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    demo_data = pd.DataFrame({"Close": prices}, index=dates)
    st.line_chart(demo_data)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
'''
    
    with open(dashboard_dir / 'app.py', 'w') as f:
        f.write(demo_code)
    
    print_success("Created demo dashboard")

def run_demo():
    """Run strategy demos."""
    print_header()
    print_info("Running strategy demos...\n")
    
    demos = [
        ('strategies/trend_following.py', 'Trend Following'),
        ('strategies/mean_reversion.py', 'Mean Reversion'),
        ('strategies/ml_models.py', 'Machine Learning'),
        ('strategies/multi_factor.py', 'Multi-Factor'),
    ]
    
    base_path = Path(__file__).parent
    
    for demo_file, name in demos:
        demo_path = base_path / demo_file
        if demo_path.exists():
            print(f"\n{Colors.BOLD}{'='*50}{Colors.END}")
            print(f"{Colors.BLUE}{name} Demo{Colors.END}")
            print(f"{'='*50}\n")
            
            try:
                subprocess.run([sys.executable, str(demo_path)], timeout=30)
            except subprocess.TimeoutExpired:
                print_warning(f"{name} demo timed out")
            except Exception as e:
                print_error(f"Error running {name}: {e}")

def run_agents(continuous=False, symbols=None, interval=300, iterations=None):
    """Run the multi-agent pipeline."""
    print_header()
    
    print(f"""
{Colors.BLUE}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║              OBSIDIAN QUANT - AGENT PIPELINE                    ║
║      Cooperative Multi-Agent Financial Reasoning Engine       ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}
""")
    
    # Add project root to path
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from agents.orchestrator import AgentOrchestrator
        print_success("Agent system loaded")
    except ImportError as e:
        print_error(f"Failed to import agent system: {e}")
        return
    
    symbols = symbols or ["AAPL", "MSFT", "GOOGL"]
    
    orch = AgentOrchestrator()
    orch.start()
    
    # Print pipeline diagram
    print(f"\n{orch.pipeline_diagram()}\n")
    
    if continuous:
        print_info(f"Running continuously: {symbols} every {interval}s")
        print_info("Press Ctrl+C to stop\n")
        orch.run_continuous(
            symbols=symbols,
            interval_seconds=interval,
            max_iterations=iterations,
        )
    else:
        print_info(f"Running single pipeline pass for: {symbols}\n")
        results = orch.run_pipeline(symbols)
        
        print(f"\n{'='*60}")
        print(f"{Colors.BOLD}PIPELINE RESULTS{Colors.END}")
        print(f"{'='*60}")
        
        for sym, result in results.items():
            d = result.to_dict()
            status = f"{Colors.GREEN}SUCCESS{Colors.END}" if d['success'] else f"{Colors.YELLOW}PARTIAL{Colors.END}"
            print(f"\n{Colors.BOLD}{sym}{Colors.END} — {status} ({d['duration_ms']:.0f}ms)")
            print(f"  Stages: {', '.join(d['stages_completed'])}")
            if d['stages_failed']:
                print(f"  Failed: {', '.join(d['stages_failed'])}")
            for key in ['signal', 'confidence', 'regime', 'conviction', 'direction', 'resilience_score']:
                if key in d['data']:
                    val = d['data'][key]
                    if isinstance(val, float):
                        print(f"  {key}: {val:.4f}")
                    else:
                        print(f"  {key}: {val}")
        
        import json
        summary = orch.summary()
        print(f"\n{Colors.BOLD}Summary:{Colors.END}")
        print(f"  Success Rate: {summary['recent_success_rate']:.0%}")
        print(f"  Avg Duration: {summary['avg_duration_ms']:.0f}ms")
        print(f"  System Health: {summary['system_health']}")

def main():
    setup_logging()
    print_header()
    
    # Parse arguments
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        print(__doc__)
        return
    
    # Check dependencies
    print(f"{Colors.BOLD}Checking environment...{Colors.END}\n")
    
    python_ok = check_python_version()
    if not python_ok:
        print_error("Please install Python 3.9 or higher")
        return
    
    deps_ok, missing = check_core_dependencies()
    check_optional_dependencies()
    
    print()
    
    # Handle commands
    if '--setup' in args:
        print(f"{Colors.BOLD}Setting up environment...{Colors.END}\n")
        
        # Determine which requirements to install
        if '--full' in args:
            install_dependencies('requirements.txt')
        elif '--ml' in args:
            install_dependencies('requirements-ml.txt')
        else:
            install_dependencies('requirements-core.txt')
        
        print()
        print_success("Setup complete! Run 'python run.py' to start the dashboard.")
        return
    
    if '--check' in args:
        if deps_ok:
            print_success("All core dependencies are installed")
        else:
            print_warning("Some dependencies are missing")
            print_info("Run 'python run.py --setup' to install them")
        return
    
    if '--demo' in args:
        run_demo()
        return
    
    if '--agents' in args:
        continuous = '--continuous' in args
        
        # Parse --symbols
        symbols = None
        for i, a in enumerate(args):
            if a == '--symbols' and i + 1 < len(args):
                symbols = [s.strip() for s in args[i + 1].split(',')]
        
        # Parse --interval
        interval = 300
        for i, a in enumerate(args):
            if a == '--interval' and i + 1 < len(args):
                try:
                    interval = int(args[i + 1])
                except ValueError:
                    pass
        
        run_agents(continuous=continuous, symbols=symbols, interval=interval)
        return
    
    # Auto-install missing dependencies
    if not deps_ok:
        print_warning("Missing core dependencies")
        response = input("Install them now? [Y/n]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            if install_missing(missing):
                deps_ok = True
            else:
                print_error("Failed to install dependencies")
                print_info("Try: pip install -r requirements-core.txt")
                return
        else:
            print_info("Skipping installation. Some features may not work.")
    
    # Run the dashboard
    print(f"\n{Colors.BOLD}Starting dashboard...{Colors.END}\n")
    run_dashboard()

if __name__ == '__main__':
    main()
