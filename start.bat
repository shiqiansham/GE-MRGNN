@echo off
title GE-MRGNN System

cd /d "%~dp0"

echo ============================================================
echo        GE-MRGNN Detection System - Quick Start
echo ============================================================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python first.
    pause
    exit /b 1
)

:: Use venv if available
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
    echo [ENV] Using virtual environment: .venv
) else (
    set PYTHON=python
    echo [ENV] Using system Python
)

:: Check model
if exist "results\ge-mrgcn\rgcn_group_model.pt" (
    echo [MODEL] Found: results\ge-mrgcn\rgcn_group_model.pt
) else (
    echo [WARN] Model file not found. API will start without model.
)

:: Check data
if exist "features.pt" (
    echo [DATA] Found feature data
) else (
    echo [WARN] features.pt not found
)

echo.
echo [START] Starting API server on port 8000 ...
echo [URL]   http://127.0.0.1:8000
echo [STOP]  Press Ctrl+C to stop
echo ============================================================
echo.

%PYTHON% api_server.py --host 127.0.0.1 --port 8000

pause
