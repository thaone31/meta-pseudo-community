#!/bin/bash

# Alternative installation script for handling PyTorch Geometric dependencies
# This script installs dependencies in the correct order to avoid build issues

set -e

echo "Installing Meta-Learning Pseudo-Labels Dependencies..."
echo "====================================================="

# Function to detect CUDA version
detect_cuda() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        cuda_version=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
        echo "CUDA $cuda_version detected"
        return 0
    else
        echo "No CUDA detected, using CPU version"
        return 1
    fi
}

# Step 1: Install PyTorch first
echo "Step 1: Installing PyTorch..."
if detect_cuda; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Step 2: Install PyTorch Geometric
echo "Step 2: Installing PyTorch Geometric..."
pip install torch-geometric

# Step 3: Install PyG optional dependencies
echo "Step 3: Installing PyTorch Geometric optional dependencies..."
if detect_cuda; then
    echo "Installing PyG extensions with CUDA support..."
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu121.html
else
    echo "Installing PyG extensions for CPU..."
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
fi

# Step 4: Install remaining dependencies
echo "Step 4: Installing remaining dependencies..."
pip install -r requirements.txt

echo "✓ All dependencies installed successfully!"

# Verify installation
echo "Verifying installation..."
python -c "
import torch
import torch_geometric
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch Geometric version: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✓ Installation verified!')
"
