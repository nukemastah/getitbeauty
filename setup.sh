#!/bin/bash

# Setup script for Skincare Hybrid Recommendation System
# For Linux (Arch Linux compatible)

echo "============================================================"
echo "Skincare Hybrid Recommender - Setup Script"
echo "============================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo ""
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if models directory exists
if [ ! -d "models" ]; then
    mkdir -p models
    echo ""
    echo "Created models directory."
fi

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Train models: python train_models.py"
echo "2. Run app: streamlit run app.py"
echo ""
echo "To activate virtual environment manually:"
echo "  source venv/bin/activate"
echo ""
