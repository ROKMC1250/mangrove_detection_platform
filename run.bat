@echo off
setlocal enabledelayedexpansion

REM Create venv if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

set PYTHONUNBUFFERED=1

REM --- User configuration ---
REM All deployment-specific settings live in .env (copy .env.example to .env and
REM edit that one file). Keep comments on their own lines. Variables already set
REM in the system environment take precedence.
if exist "%~dp0.env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%~dp0.env") do (
        if not defined %%a set "%%a=%%b"
    )
)

REM Earth Engine credentials
if not defined EE_SERVICE_ACCOUNT (
    for /f "tokens=2 delims=:," %%a in ('findstr /c:"client_email" backend\ee-service-account-key.json') do (
        set EE_SERVICE_ACCOUNT=%%a
        set EE_SERVICE_ACCOUNT=!EE_SERVICE_ACCOUNT:"=!
        set EE_SERVICE_ACCOUNT=!EE_SERVICE_ACCOUNT: =!
    )
)

if not defined EE_SERVICE_ACCOUNT_KEY (
    set EE_SERVICE_ACCOUNT_KEY=%~dp0backend\ee-service-account-key.json
)

if not defined GOOGLE_APPLICATION_CREDENTIALS (
    set GOOGLE_APPLICATION_CREDENTIALS=%EE_SERVICE_ACCOUNT_KEY%
)

REM GCS is optional - set GCS_BUCKET / GCS_BUCKET_REGION in .env to enable
REM server-side GeoTIFF export for large AOIs.

REM MODEL paths - these should be set by user or left empty to disable model
REM if not defined MODEL1_LOG_DIR (
REM     set MODEL1_LOG_DIR=C:\path\to\your\model\logs
REM )
REM if not defined MODEL_ROOT (
REM     set MODEL_ROOT=C:\path\to\your\model\root
REM )

cd /d %~dp0

python -m pip install -r backend\requirements.txt

echo ======================================
echo 🚀 Starting EarthScope Server
echo ======================================
echo.
echo 📍 Server will be accessible at:
echo    - Local:   http://localhost:8000
echo    - Local:   http://127.0.0.1:8000
echo.
echo 📂 Frontend is served from /static/
echo ======================================
echo.

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

