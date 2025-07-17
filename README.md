# Meta-Learning for Pseudo-Labels in Community Detection

A research project applying Meta-Learning to improve pseudo-label generation and updating for Community Detection in complex networks.

## Overview

This pipeline combines Meta-Learning with Pseudo-Labels to improve community detection performance on complex graphs. The core idea is to use a meta-learner to learn how to generate and update optimal pseudo-labels for different types of graphs/episodes.

## Overall Architecture

### 1. Base Model
- Backbone: GCN/GAT/GIN (Graph Neural Networks)
- Support: Spectral clustering, modularity-based methods

### 2. Meta-Learner
- MAML (Model-Agnostic Meta-Learning)
- Reptile
- Custom meta-update strategies

### 3. Pseudo-label Module
- Node embedding similarity
- Clustering-based label generation
- Confidence-based label refinement

## Project Structure

```
meta-pseudo-community/
├── data/                     # Data and preprocessing
├── models/                   # Model implementations
├── baselines/               # SOTA baseline models for comparison
├── utils/                   # Utilities and helpers
├── experiments/             # Experiment scripts
├── evaluation/              # Evaluation and metrics
├── configs/                 # Configuration files
├── results/                 # Experiment results
└── notebooks/               # Jupyter notebooks for analysis
```

## Installation

### Automated (Recommended)
```bash
./setup.sh
```

### Manual Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install main dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (may require manual installation)
pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

**Note**: This project has been optimized for Python 3.12. Some advanced dependencies may require manual installation.

## Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone <repository-url>
cd meta-pseudo-community

# Run automated setup
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Download and Prepare Datasets
```bash
# Download supported datasets (Cora, CiteSeer, PubMed, DBLP, Amazon)
python data/download_datasets.py
```

### 3. Quick Test Run
```bash
# Run a quick experiment with minimal configuration
python run_experiments.py --config quick
```

### 4. Train Meta-Learning Model
```bash
# Train with default configuration
python experiments/train_meta_pseudo.py --config configs/meta_gcn.yaml

# Train with custom parameters
python experiments/train_meta_pseudo.py --model gcn --meta_learner maml --epochs 100
```

### 5. Evaluate Model
```bash
# Evaluate a trained model
python experiments/evaluate.py --model_path results/meta_gcn/best_model.pth

# Evaluate on specific datasets
python experiments/evaluate.py --model_path results/meta_gcn/best_model.pth --datasets Cora CiteSeer PubMed

# Compare with baselines
python experiments/compare_baselines.py
```

## Detailed Usage

### Training Configuration

The training can be customized through YAML config files or command-line arguments:

```bash
# Using config file
python experiments/train_meta_pseudo.py --config configs/meta_gcn.yaml

# Using command-line arguments
python experiments/train_meta_pseudo.py \
    --model gcn \
    --meta_learner maml \
    --epochs 200 \
    --meta_lr 0.001 \
    --inner_lr 0.01 \
    --batch_size 32
```

### Supported Models
- **Base Models**: GCN, GAT, GIN, GraphSAGE
- **Meta-Learners**: MAML, Reptile, Custom strategies
- **Pseudo-Label Methods**: Embedding similarity, spectral clustering, modularity

### Evaluation Metrics
- **NMI** (Normalized Mutual Information)
- **ARI** (Adjusted Rand Index) 
- **Modularity**
- **F1-score, Purity, Conductance**

## Datasets

### Supported Datasets (Auto-downloaded):
- **Citation Networks**: Cora, CiteSeer, PubMed
- **E-commerce**: Amazon-Computers  
- **Academic**: DBLP (co-author network)

### Synthetic Datasets:
- LFR benchmark graphs with ground truth
- Stochastic Block Models

**Note**: Large datasets from SNAP (LiveJournal, Orkut, Youtube) are disabled by default but can be manually enabled if needed.

## SOTA Baselines

- **Traditional**: DeepWalk, Node2Vec, Louvain, Leiden
- **Deep Learning**: GraphSAGE, GCN, GAT, DGI, DMoN
- **Self-supervised**: Deep Clustering, GCN-based pseudo-label

## Troubleshooting

### Common Issues

#### 1. Installation Problems
```bash
# If torch-geometric installation fails
pip install torch-geometric torch-scatter torch-sparse torch-cluster --extra-index-url https://data.pyg.org/whl/torch-2.1.0+cpu.html

# For CUDA support (if available)
pip install torch-geometric torch-scatter torch-sparse torch-cluster --extra-index-url https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

#### 2. Memory Issues
```bash
# Run with reduced batch size
python experiments/train_meta_pseudo.py --batch_size 16

# Use gradient accumulation
python experiments/train_meta_pseudo.py --accumulation_steps 4
```

#### 3. Dataset Download Issues
```bash
# Clear cache and retry
rm -rf data/raw data/processed
python data/download_datasets.py
```

#### 4. NaN Values in Embeddings
The codebase includes automatic handling for NaN/infinity values in embeddings, but if you encounter issues:
- Check input data quality
- Reduce learning rates
- Add gradient clipping

### Performance Tips

1. **For Quick Testing**: Use `--config quick` to run with minimal epochs and smaller datasets
2. **For Production**: Use full configuration with proper validation splits
3. **Memory Optimization**: Reduce batch size or use gradient accumulation
4. **Speed**: Use fewer datasets in `run_experiments.py` for development

## Configuration

### Default Configurations
- `configs/meta_gcn.yaml`: GCN with MAML meta-learning
- `configs/meta_gat.yaml`: GAT with meta-learning
- `configs/baseline.yaml`: Non-meta baseline configuration

### Custom Configuration
Create your own YAML config file:
```yaml
model:
  name: "gcn"
  hidden_dim: 64
  num_layers: 2
  
meta_learner:
  name: "maml"
  meta_lr: 0.001
  inner_lr: 0.01
  inner_steps: 5

training:
  epochs: 100
  batch_size: 32
  datasets: ["Cora", "CiteSeer", "PubMed"]
```

## Results and Visualization

After training, results are saved to:
- `results/[experiment_name]/`: Model checkpoints and metrics
- `results/[experiment_name]/plots/`: Visualization plots
- `results/[experiment_name]/logs/`: Training logs

Generate summary reports:
```bash
python utils/generate_summary.py --results_dir results/meta_gcn
```

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation for new functionality
4. Ensure compatibility with Python 3.12

## Author

[Author Name]

## License

MIT License