@echo off
chcp 65001 >nul

echo Installing VoxCPM Kazakh TTS...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    echo Please install Python 3.8 or higher
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION% found
echo.

REM Check pip
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip not found
    pause
    exit /b 1
)
echo ✓ pip found
echo.

REM Create virtual environment (optional)
set /p CREATE_VENV="Create virtual environment? (recommended) [Y/n]: "
if "%CREATE_VENV%"=="" set CREATE_VENV=Y

if /i "%CREATE_VENV%"=="Y" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo ✓ Virtual environment created
    echo.
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo Installing dependencies...
python -m pip install -r requirements.txt
echo.

REM Check CUDA
python -c "import torch; print('✓ PyTorch installed'); print('✓ CUDA available' if torch.cuda.is_available() else '⚠️  CUDA not available, will use CPU mode')"
echo.

echo ==================================
echo ✅ Installation complete!
echo ==================================
echo.
echo To start: start.bat
echo.
echo First run will download models (~1.5 GB)
echo.
pause
