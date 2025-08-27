#!/bin/bash

# Project setup script for Prostate Cancer Classification

echo "🔬 Setting up Prostate Cancer Classification Project..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p logs
mkdir -p checkpoints

echo "✅ Setup complete!"
echo ""
echo "🚀 To get started:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Run the main pipeline: python main.py"
echo "   3. Or use Make commands: make help"
echo ""
echo "📖 Check README.md for detailed documentation."
