#!/bin/bash

set -e

echo "Installing VoxCPM Kazakh TTS..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python $PYTHON_VERSION found"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found"
    exit 1
fi
echo "✓ pip3 found"

# Create virtual environment (optional)
echo ""
read -p "Create virtual environment? (recommended) [Y/n]: " create_venv
create_venv=${create_venv:-Y}

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip3 install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Check CUDA
echo ""
python3 -c "import torch; print('✓ PyTorch installed'); print('✓ CUDA available' if torch.cuda.is_available() else '⚠️  CUDA not available, will use CPU mode')"

# Set permissions
echo ""
chmod +x start.sh
echo "✓ Permissions set"

echo ""
echo "=================================="
echo "✅ Installation complete!"
echo "=================================="
echo ""
echo "To start: ./start.sh"
echo ""
echo "First run will download models (~1.5 GB)"
echo ""
