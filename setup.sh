#!/bin/bash
# Setup script for Meta-Learning Pseudo-Labels Community Detection

set -e  # Exit on any error

echo "Setting up Meta-Learning Pseudo-Labels Community Detection"
echo "=========================================================="

# Check Python version
python3 --version || {
    echo "Error: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
}

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core requirements
echo "Installing core requirements..."
pip install -r requirements.txt

# Try to install PyTorch Geometric
echo "Attempting to install PyTorch Geometric..."
pip install torch-geometric torch-scatter torch-sparse torch-cluster || {
    echo "Warning: PyTorch Geometric installation failed. You may need to install manually."
    echo "See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
}

# Install optional requirements if they exist
if [ -f "requirements-optional.txt" ]; then
    echo "Installing optional requirements..."
    pip install -r requirements-optional.txt || {
        echo "Warning: Some optional packages failed to install."
    }
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed results logs

echo ""
echo "Setup completed successfully!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Download datasets: python data/download_datasets.py"
echo "3. Run quick test: python main.py --quick"
echo "4. Run full pipeline: python main.py"
