#!/bin/bash

# Meta-Learning Pseudo-Labels Community Detection
# Quick Setup and Test Script

set -e  # Exit on any error

echo "========================================"
echo "Meta-Learning Pseudo-Labels Community Detection"
echo "Quick Setup and Test Script"
echo "========================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

if ! command_exists pip; then
    echo "Error: pip is required but not installed."
    exit 1
fi

echo "✓ Python 3 and pip found"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "Warning: requirements.txt not found. Installing basic dependencies..."
    pip install torch torch-geometric scikit-learn matplotlib pandas numpy networkx
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/raw data/processed results/training results/evaluation results/baselines results/training_logs
mkdir -p logs checkpoints
echo "✓ Directory structure created"

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to run with error handling
run_with_logging() {
    local description="$1"
    local command="$2"
    local log_file="logs/$(echo "$description" | tr ' ' '_' | tr '[:upper:]' '[:lower:]').log"
    
    echo "Running: $description"
    echo "Command: $command"
    echo "Log: $log_file"
    
    if eval "$command" > "$log_file" 2>&1; then
        echo "✓ $description completed successfully"
        return 0
    else
        echo "✗ $description failed. Check log: $log_file"
        echo "Last few lines of log:"
        tail -10 "$log_file"
        return 1
    fi
}

# Test data download
echo ""
echo "Testing data download..."
if run_with_logging "Data download test" "python data/download_datasets.py --dataset Cora --test-mode"; then
    echo "✓ Data download works"
else
    echo "Warning: Data download test failed, but continuing..."
fi

# Test preprocessing
echo ""
echo "Testing data preprocessing..."
if run_with_logging "Data preprocessing test" "python data/preprocess.py --test-mode"; then
    echo "✓ Data preprocessing works"
else
    echo "Warning: Data preprocessing test failed, but continuing..."
fi

# Test model components
echo ""
echo "Testing model components..."
if run_with_logging "Model component test" "python -c \"
import sys
sys.path.append('.')
from models.base_models import GNNModel
from models.pseudo_labels import PseudoLabelGenerator
from models.meta_learning import MAMLTrainer
print('✓ All model components imported successfully')
\""; then
    echo "✓ Model components work"
else
    echo "Warning: Model component test failed"
fi

# Test utilities
echo ""
echo "Testing utilities..."
if run_with_logging "Utilities test" "python -c \"
import sys
sys.path.append('.')
from utils.evaluation import CommunityEvaluator
from utils.visualization import GraphVisualizer
print('✓ All utilities imported successfully')
\""; then
    echo "✓ Utilities work"
else
    echo "Warning: Utilities test failed"
fi

# Quick experiment test
echo ""
echo "Running quick experiment test..."
echo "This will run a minimal experiment to test the complete pipeline..."

# Create quick test config
cat > quick_test_config.json << EOF
{
    "datasets": ["Cora"],
    "meta_learning_methods": ["Meta-GCN"],
    "baseline_methods": ["Louvain"],
    "n_trials": 1,
    "meta_learning_epochs": 50,
    "adaptation_steps": 5,
    "patience": 10,
    "generate_report": true,
    "run_notebooks": false
}
EOF

if run_with_logging "Quick experiment test" "python run_experiments.py --config quick_test_config.json --output-dir results/quick_test"; then
    echo "✓ Quick experiment test completed successfully"
    echo ""
    echo "Results available in: results/quick_test/"
    if [ -f "results/quick_test/EXPERIMENT_REPORT.md" ]; then
        echo "Experiment report:"
        cat "results/quick_test/EXPERIMENT_REPORT.md"
    fi
else
    echo "⚠️  Quick experiment test failed, but setup is complete"
fi

# Test Jupyter notebooks (optional)
if command_exists jupyter; then
    echo ""
    echo "Testing Jupyter notebook setup..."
    if run_with_logging "Jupyter notebook test" "jupyter nbconvert --to html --execute notebooks/data_exploration.ipynb --output test_data_exploration.html --ExecutePreprocessor.timeout=60"; then
        echo "✓ Jupyter notebooks work"
        echo "Test notebook HTML generated: notebooks/test_data_exploration.html"
    else
        echo "Warning: Jupyter notebook test failed"
    fi
else
    echo "Note: Jupyter not installed. To use notebooks, install with: pip install jupyter"
fi

# Summary
echo ""
echo "========================================"
echo "SETUP SUMMARY"
echo "========================================"
echo "✓ Virtual environment created and activated"
echo "✓ Dependencies installed"
echo "✓ Directory structure created"
echo "✓ Basic functionality tested"

echo ""
echo "QUICK START GUIDE:"
echo "===================="
echo ""
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run a quick test:"
echo "   python run_experiments.py --quick"
echo ""
echo "3. Run full experiments:"
echo "   python run_experiments.py"
echo ""
echo "4. Analyze results with notebooks:"
echo "   jupyter notebook notebooks/"
echo ""
echo "5. Generate summary report:"
echo "   python utils/generate_summary.py --results-dir results"
echo ""
echo "CONFIGURATION:"
echo "=============="
echo "- Edit configs/*.yaml to modify experiment parameters"
echo "- Use run_experiments.py --help for all options"
echo "- Check main.py for individual component testing"
echo ""
echo "TROUBLESHOOTING:"
echo "================"
echo "- Check logs/ directory for detailed error logs"
echo "- Ensure CUDA is available if using GPU"
echo "- For installation issues, check requirements.txt"
echo ""
echo "For more information, see README.md"
echo ""

# Cleanup test files
rm -f quick_test_config.json

echo "Setup completed successfully! 🎉"
