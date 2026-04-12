@echo off
chcp 65001 >nul 2>&1
echo ============================================
echo   LawNuri - One-Click Install (Windows)
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)
echo [OK] Python found.

:: Check Flutter
flutter --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Flutter is not installed or not in PATH.
    echo Please install Flutter from https://docs.flutter.dev/get-started/install
    pause
    exit /b 1
)
echo [OK] Flutter found.

:: Backend setup
echo.
echo --- Setting up Backend ---
cd /d "%~dp0backend"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt --quiet

echo [OK] Backend setup complete.

:: Frontend setup
echo.
echo --- Setting up Frontend (Flutter) ---
cd /d "%~dp0app"

echo Installing Flutter dependencies...
flutter pub get

echo [OK] Frontend setup complete.

:: Create .env if not exists
cd /d "%~dp0"
if not exist ".env" (
    echo Creating .env from .env.example...
    copy .env.example .env
    echo [INFO] Please edit .env if needed. API keys are managed via Settings UI.
)

echo.
echo ============================================
echo   Setup complete!
echo.
echo   To launch LawNuri:
echo     1) Backend  : cd backend ^&^& .venv\Scripts\activate ^&^& python run.py
echo     2) Frontend : cd app ^&^& flutter run -d windows
echo ============================================
pause
