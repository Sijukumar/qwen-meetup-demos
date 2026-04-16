#!/bin/bash

echo "============================================================"
echo "   Voice Chatbot - Installation Script"
echo "============================================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created."
else
    echo "Virtual environment already exists."
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated."
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ Pip upgraded."
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully!"
    echo ""
    echo "============================================================"
    echo "   Installation Complete!"
    echo "============================================================"
    echo ""
    echo "To start the voice chatbot:"
    echo ""
    echo "  1. Run: source venv/bin/activate"
    echo "  2. Run: python simple_voice_chat.py"
    echo ""
    echo "Note: Make sure your .env file has your API key configured."
    echo ""
else
    echo "✗ Installation failed."
    echo ""
    echo "If PyAudio fails, try:"
    echo "  macOS:   brew install portaudio"
    echo "  Ubuntu:  sudo apt-get install portaudio19-dev"
    echo "  Windows: pip install pipwin && pipwin install pyaudio"
    echo ""
fi
