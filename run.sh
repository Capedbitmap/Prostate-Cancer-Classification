#!/bin/bash

# Prostate Cancer Classification - Training Launcher
# This script ensures the environment is activated and runs the main pipeline

echo "🔬 Prostate Cancer Classification Pipeline"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "💡 Run: ./setup.sh to create the environment"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Verify key packages are installed
echo "🔍 Checking dependencies..."
python -c "import torch, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not found!"
    echo "💡 Run: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Environment ready!"
echo "🚀 Starting training pipeline..."
echo ""

# Run the main script
python main.py

echo ""
echo "🏁 Training completed!"
echo "📊 Check the SICAPv2_results/ directory for outputs"
