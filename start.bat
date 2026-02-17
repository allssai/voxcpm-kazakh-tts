@echo off
chcp 65001 >nul

echo Starting VoxCPM Kazakh TTS...
echo.

python web_app.py

if errorlevel 1 (
    echo.
    echo Error: Failed to start the application.
    echo Please make sure you have installed all dependencies:
    echo   pip install -r requirements.txt
    pause
)
