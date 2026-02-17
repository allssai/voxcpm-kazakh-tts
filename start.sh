#!/bin/bash

echo "Starting VoxCPM Kazakh TTS..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Start application
python3 web_app.py
