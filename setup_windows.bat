@echo off
REM One-click Windows setup for sybil-fl: venv, CPU PyTorch, package, E0 check, tests.
REM Full output goes to setup_log.txt next to this file.
setlocal
cd /d "%~dp0"
set "LOG=%~dp0setup_log.txt"
echo sybil-fl setup started %date% %time% > "%LOG%"

echo [1/6] Checking Python...
REM Prefer the py launcher, but only if it actually runs (it can point at a removed install).
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (python --version >nul 2>&1 && set "PY=python")
if not defined PY (echo FAILED: Python not found. Install Python 3.10+ from python.org >> "%LOG%" & goto fail)
echo using: %PY% >> "%LOG%"
%PY% --version >> "%LOG%" 2>&1

echo [2/6] Creating virtual environment .venv ...
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv >> "%LOG%" 2>&1
  if errorlevel 1 (echo FAILED: venv creation >> "%LOG%" & goto fail)
)
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --progress-bar off >> "%LOG%" 2>&1

echo [3/6] Installing CPU PyTorch (about 200 MB, takes a few minutes)...
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --progress-bar off >> "%LOG%" 2>&1
if errorlevel 1 (echo FAILED: torch install >> "%LOG%" & goto fail)

echo [4/6] Installing sybil-fl and test tools...
"%VPY%" -m pip install -e ".[dev]" --progress-bar off >> "%LOG%" 2>&1
if errorlevel 1 (echo FAILED: package install >> "%LOG%" & goto fail)

echo [5/6] Running E0 central training check (downloads MNIST, ~1-2 min)...
echo ===== E0 run_central ===== >> "%LOG%"
"%VPY%" experiments\run_central.py >> "%LOG%" 2>&1
if errorlevel 1 (echo FAILED: run_central >> "%LOG%" & goto fail)

echo [6/6] Running test suite...
echo ===== pytest ===== >> "%LOG%"
"%VPY%" -m pytest -q >> "%LOG%" 2>&1
if errorlevel 1 (echo FAILED: tests >> "%LOG%" & goto fail)

echo ===== versions ===== >> "%LOG%"
"%VPY%" -c "import sys,platform,torch,torchvision,numpy; print('python',sys.version.split()[0]); print(platform.platform()); print('torch',torch.__version__,'torchvision',torchvision.__version__,'numpy',numpy.__version__)" >> "%LOG%" 2>&1
echo SETUP OK >> "%LOG%"
echo.
echo Setup finished successfully. Details in setup_log.txt
goto end

:fail
echo.
echo Setup FAILED - see setup_log.txt
:end
pause
