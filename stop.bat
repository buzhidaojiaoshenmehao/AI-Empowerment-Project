@echo off
chcp 65001 >nul
title Stop AI-KB

echo.
echo Looking for running services...

set STOPPED=0

:: Find and kill processes on ports 8000-8005
for /L %%p in (8000,1,8005) do (
    for /f "tokens=4,5" %%a in ('netstat -ano ^| findstr ":%%p "') do (
        echo   Stopping port %%p (PID: %%b)
        taskkill /PID %%b /F >nul 2>&1
        set STOPPED=1
    )
)

:: Fallback: kill any python/uvicorn processes
for /f "tokens=2" %%a in ('tasklist ^| findstr /I "AI-KB python uvicorn" 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
    set STOPPED=1
)

if %STOPPED% EQU 1 (
    echo [OK] Service stopped
) else (
    echo [INFO] No service found (already stopped)
)

echo.
pause
