@echo off
REM Live demo for sybil-fl: the same 8-Sybil label-flip attack (7 -> 1), 5 rounds each,
REM first undefended, then with the secret-projection detector. About 1-2 minutes in total.
REM Needs setup_windows.bat to have run first (.venv and MNIST in data\).
REM Run outputs go to results\demo\attack and results\demo\defended (git-ignored).
setlocal
cd /d "%~dp0"
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (echo .venv not found - run setup_windows.bat first. & goto fail)

echo ============================================================
echo  Run 1 of 2: 20 honest clients + 8 Sybils, NO defence
echo  Watch ASR (share of test 7s classified as 1) and "flagged 0".
echo ============================================================
"%VPY%" experiments\run_main.py --config configs\demo_attack.yaml --out results\demo\attack
if errorlevel 1 (echo FAILED: run 1 & goto fail)

echo.
echo ============================================================
echo  Run 2 of 2: same clients and attack, PROJECTED detector
echo  (k = 64, secret +-1/sqrt(k) matrix from the server key, tau = 0.15)
echo  Watch "flagged 8 (tp 8, fp 0)" and aggregated 20/28.
echo ============================================================
"%VPY%" experiments\run_main.py --config configs\demo_defended.yaml --out results\demo\defended
if errorlevel 1 (echo FAILED: run 2 & goto fail)

echo.
echo Demo finished. Figures for the full study are in plots\
goto end

:fail
echo.
echo Demo FAILED - see the messages above.
:end
pause
