#!/bin/bash
# Quick Setup Script for OpenBB and Bloomberg Integration

echo "================================"
echo "JR Bloomberg Terminal Setup"
echo "================================"
echo ""

# Step 1: Install dependencies
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

echo "[Step 1] ✓ Dependencies installed"
echo ""

# Step 2: Test OpenBB
echo "[2/3] Testing OpenBB installation..."
python data/openbb_integration.py

echo "[Step 2] ✓ OpenBB configured"
echo ""

# Step 3: Fetch data
echo "[3/3] Fetching initial data..."
python data/ingest.py

echo "[Step 3] ✓ Data fetched and saved"
echo ""

echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Start dashboard:     streamlit run dashboard/app_streamlit.py"
echo "2. Run analytics:       python analytics/risk.py"
echo "3. Run backtest:        python research/backtester.py"
echo "4. Run execution:       python execution/omega.py"
echo ""
echo "Or use make commands:"
echo "  make dashboard"
echo "  make analytics"
echo "  make backtest"
echo "  make execution"
echo ""
echo "Documentation: See SETUP_GUIDE.md for detailed configuration"
