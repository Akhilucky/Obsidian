# Bloomberg Terminal - Quick Launcher (PowerShell)
# Usage: .\run.ps1 [dashboard|demo|check]

param(
    [ValidateSet("dashboard", "demo", "check", "")]
    [string]$Action = "dashboard"
)

$venvPython = ".\.venv\Scripts\python.exe"
$venvStreamlit = ".\.venv\Scripts\streamlit.exe"

# Check if venv exists
if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found. Running setup first..." -ForegroundColor Yellow
    & .\setup.ps1 core
}

switch ($Action) {
    "dashboard" {
        Write-Host "Starting AEGIS Terminal Dashboard..." -ForegroundColor Cyan
        if (Test-Path "dashboard\app_streamlit.py") {
            & $venvStreamlit run dashboard\app_streamlit.py
        } else {
            & $venvPython run.py
        }
    }
    "demo" {
        Write-Host "Running strategy demos..." -ForegroundColor Cyan
        & $venvPython run.py --demo
    }
    "check" {
        Write-Host "Checking dependencies..." -ForegroundColor Cyan
        & $venvPython run.py --check
    }
    default {
        Write-Host "Starting Bloomberg Terminal Dashboard..." -ForegroundColor Cyan
        & $venvPython run.py
    }
}
