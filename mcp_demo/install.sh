#!/bin/bash

echo "============================================================"
echo "   MCP Demo - Installation Script"
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

# Check if Node.js is installed
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed."
    echo ""
    echo "Please install Node.js v18+ from: https://nodejs.org/"
    echo "Or use: brew install node"
    exit 1
fi

echo "Node.js version:"
node --version
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
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

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Python dependencies installed."
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << EOF
# DashScope API Configuration
DASHSCOPE_API_KEY=your_api_key_here
EOF
    echo "✓ Created .env template."
    echo "  Please edit .env and add your API key."
else
    echo ".env file found."
fi
echo ""

echo "============================================================"
echo "   Installation Complete!"
echo "============================================================"
echo ""
echo "Prerequisites installed:"
echo "  ✓ Python 3"
echo "  ✓ Node.js"
echo "  ✓ Python packages (openai, python-dotenv)"
echo ""
echo "To run the MCP demo:"
echo ""
echo "  1. Edit .env file and set your DASHSCOPE_API_KEY"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python mcp_demo.py"
echo ""
echo "The demo will:"
echo "  - Create demo files in ~/mcp_demo_files"
echo "  - Start the Filesystem MCP server automatically"
echo "  - Allow Qwen to read/write files via MCP"
echo ""
