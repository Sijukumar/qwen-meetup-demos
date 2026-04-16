#!/bin/bash

echo "============================================================"
echo "   Image Generation - Installation Script"
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

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found."
    echo "Creating .env template..."
    cat > .env << EOF
# DashScope API Configuration
DASHSCOPE_API_KEY=your_api_key_here
EOF
    echo "✓ Created .env template. Please edit it and add your API key."
else
    echo ".env file found."
fi
echo ""

echo "============================================================"
echo "   Installation Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and set your DASHSCOPE_API_KEY"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python simple_image_gen_v2.py"
echo ""
echo "To deactivate the virtual environment, run: deactivate"
echo ""
