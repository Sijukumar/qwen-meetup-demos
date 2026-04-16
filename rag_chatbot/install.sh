#!/bin/bash

echo "============================================================"
echo "   RAG Chatbot - Installation Script"
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
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found."
    echo "Please create a .env file with your configuration."
else
    echo ".env file found."
fi
echo ""

echo "============================================================"
echo "   Installation Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Run: source venv/bin/activate"
echo "2. Run: python simple_rag_chromadb.py"
echo ""
echo "To deactivate the virtual environment, run: deactivate"
