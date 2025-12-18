# Bloomberg Terminal - Easy Setup Script (PowerShell)
# =====================================================
# Usage: .\setup.ps1 [core|ml|full]

param(
    [ValidateSet("core", "ml", "full")]
    [string]$Mode = "core"
)

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║            BLOOMBERG TERMINAL - SETUP WIZARD                 ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "[1/4] Checking Python..." -ForegroundColor Yellow

$pythonVersion = & python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Step 2: Create/activate virtual environment
Write-Host "[2/4] Setting up virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path ".venv")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Gray
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate venv
$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

if (Test-Path $venvPython) {
    Write-Host "  ✓ Virtual environment ready" -ForegroundColor Green
} else {
    Write-Host "  ✗ Virtual environment not found" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "  Upgrading pip..." -ForegroundColor Gray
& $venvPip install --upgrade pip --quiet 2>$null

# Step 3: Install dependencies
Write-Host "[3/4] Installing dependencies ($Mode mode)..." -ForegroundColor Yellow

$requirementsFile = switch ($Mode) {
    "core" { "requirements-core.txt" }
    "ml"   { "requirements-ml.txt" }
    "full" { "requirements.txt" }
}

if (Test-Path $requirementsFile) {
    Write-Host "  Installing from $requirementsFile..." -ForegroundColor Gray
    & $venvPip install -r $requirementsFile --quiet 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "  ! Some dependencies may have failed (non-critical)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ! $requirementsFile not found, using requirements.txt" -ForegroundColor Yellow
    & $venvPip install -r requirements.txt --quiet 2>$null
}

# Step 4: Verify installation
Write-Host "[4/4] Verifying installation..." -ForegroundColor Yellow

$testCode = @"
import sys
packages = ['numpy', 'pandas', 'yfinance', 'streamlit']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print('Missing: ' + ', '.join(missing))
    sys.exit(1)
print('All core packages OK')
"@

$result = & $venvPython -c $testCode 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ $result" -ForegroundColor Green
} else {
    Write-Host "  ! $result" -ForegroundColor Yellow
}

# Done
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    SETUP COMPLETE!                           ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "Quick Start Commands:" -ForegroundColor Cyan
Write-Host "  .\run.ps1                    # Start the dashboard" -ForegroundColor White
Write-Host "  .\.venv\Scripts\python run.py    # Alternative: use run.py" -ForegroundColor Gray
Write-Host ""

Write-Host "Manual Commands:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1     # Activate environment" -ForegroundColor White
Write-Host "  streamlit run dashboard\app.py  # Run dashboard" -ForegroundColor White
Write-Host "  python strategies\ml_models.py  # Run ML demo" -ForegroundColor White
Write-Host ""

Write-Host "Install Options:" -ForegroundColor Cyan
Write-Host "  .\setup.ps1 core   # Minimal (default)" -ForegroundColor Gray
Write-Host "  .\setup.ps1 ml     # + Machine Learning" -ForegroundColor Gray
Write-Host "  .\setup.ps1 full   # Everything" -ForegroundColor Gray
Write-Host ""
