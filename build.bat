@echo off
chcp 65001 >nul
title Build AI-KB EXE

setlocal enabledelayedexpansion

set VERSION=1.0.0
for /f "tokens=*" %%i in ('git describe --tags --abbrev=0 2^>nul') do set VERSION=%%i

echo ============================================
echo   AI +-+B - Single EXE Build
echo   Version: %VERSION%
echo ============================================
echo.

:: Check the project Python environment created by start.bat
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [FAIL] Project environment not found. Run start.bat once before building.
    pause & exit /b 1
)
"%PYTHON%" -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Python 3.10+ is required by LangChain 1.x.
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('"%PYTHON%" --version 2^>^&1') do set pyver=%%v
echo [OK] Python %pyver% ^(.venv^)

"%PYTHON%" _check_deps.py >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [..] Updating project dependencies...
    "%PYTHON%" -m pip install -r backend\requirements.txt -q
    if %ERRORLEVEL% NEQ 0 (
        echo [FAIL] Dependency installation failed
        pause & exit /b 1
    )
)

:: Check PyInstaller
echo [..] Checking PyInstaller...
"%PYTHON%" -m pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [..] Installing PyInstaller...
    "%PYTHON%" -m pip install pyinstaller -q
    if %ERRORLEVEL% NEQ 0 (
        echo [FAIL] PyInstaller install failed
        pause & exit /b 1
    )
    echo [OK] PyInstaller installed
) else (
    echo [OK] PyInstaller ready
)

:: Clean
echo.
echo [..] Cleaning previous build artifacts...
if exist "dist\AI-KB.exe" del /f /q "dist\AI-KB.exe" >nul 2>&1
if exist "build" rmdir /s /q "build" >nul 2>&1

:: Build
echo.
echo [..] Building single-file EXE (2-4 min)...
echo.
"%PYTHON%" -m PyInstaller --noconfirm "AI-KB.spec"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Build complete!

    :: Archive
    if not exist "dist\artifacts" mkdir "dist\artifacts"
    copy /Y "dist\AI-KB.exe" "dist\artifacts\AI-KB-%VERSION%.exe" >nul 2>&1

    :: Remove build cache (keep dist)
    if exist "build" rmdir /s /q "build" >nul 2>&1

    :: Show result
    set FILESIZE=0
    for %%s in (dist\AI-KB.exe) do set FILESIZE=%%~zs
    if !FILESIZE! GTR 0 (
        set /a MB=!FILESIZE!/1048576
    )

    echo.
    echo ============================================
    echo   Output: dist\AI-KB.exe
    if defined MB echo   Size:   !MB! MB
    echo   Arch:   dist\artifacts\AI-KB-%VERSION%.exe
    echo.
    echo   Run:    dist\AI-KB.exe
    echo ============================================
) else (
    echo.
    echo [FAIL] Build failed. See errors above.
)

echo.
pause
