@echo off
chcp 65001 >nul
title AI Knowledge Base

echo ============================================
echo   AI +-+B - Quick Start
echo ============================================
echo.

:: --- 1. Select Python 3.10+ and create the project environment ---
set "PYTHON_BOOTSTRAP="
py -3.12 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if %ERRORLEVEL% EQU 0 set "PYTHON_BOOTSTRAP=py -3.12"
if not defined PYTHON_BOOTSTRAP (
    python -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
    if %ERRORLEVEL% EQU 0 set "PYTHON_BOOTSTRAP=python"
)
if not defined PYTHON_BOOTSTRAP (
    echo [FAIL] Python 3.10+ not found. Python 3.12 is recommended.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [..] Creating project virtual environment...
    %PYTHON_BOOTSTRAP% -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [FAIL] Unable to create .venv
        pause
        exit /b 1
    )
)
set "PYTHON=.venv\Scripts\python.exe"
for /f "tokens=2" %%v in ('"%PYTHON%" --version 2^>^&1') do set pyver=%%v
echo [OK] Python %pyver% ^(.venv^)

:: --- 2. Install dependencies ---
echo.
echo [..] Checking Python dependencies and LangChain runtime...
"%PYTHON%" _check_deps.py >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [..] First run, installing dependencies ^(2-5 min^)...
    "%PYTHON%" -m pip install -r backend/requirements.txt -q
    if %ERRORLEVEL% NEQ 0 (
        echo [FAIL] Dependency installation failed
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies ready
)

:: --- 4. Kill old processes ---
echo.
echo [..] Cleaning up old processes...
for /L %%p in (8000,1,8005) do (
    for /f "tokens=4,5" %%a in ('netstat -ano ^| findstr ":%%p "') do (
        taskkill /PID %%b /F >nul 2>&1
    )
)
ping -n 2 127.0.0.1 >nul
echo [OK] Ports released

:: --- 5. Start service ---
echo.
echo [..] Starting service...
start "AI-KB" /B "%PYTHON%" run.py

:: Wait for service (up to 20 sec)
echo [..] Waiting for service to start...
set attempt=0
:retry
ping -n 2 127.0.0.1 >nul
set /a attempt+=1

netstat -ano | findstr ":8000" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Service started at http://localhost:8000
    echo.
    echo ============================================
    echo   Opening browser...
    echo   If not:  http://localhost:8000
    echo.
    echo   Press any key to stop
    echo ============================================
    start http://localhost:8000
) else (
    if %attempt% LSS 10 goto retry
    echo [FAIL] Service failed to start after 20 seconds
    echo   Try:  .venv\Scripts\python.exe run.py
    pause
    exit /b 1
)

pause >nul
