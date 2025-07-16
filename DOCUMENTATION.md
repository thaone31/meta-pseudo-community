# PROJECT DOCUMENTATION - Meta-Learning Pseudo-Labels Community Detection

## Complete Project Overview

This project implements a comprehensive research framework for **Meta-Learning approaches to Pseudo-Label generation in Graph Community Detection**. The project includes state-of-the-art meta-learning algorithms (MAML, Reptile), advanced pseudo-label generation techniques, extensive baseline comparisons, and thorough experimental analysis.

---

## 🎯 Research Objectives

1. **Develop meta-learning algorithms** for adaptive pseudo-label generation in community detection
2. **Compare performance** against traditional and deep learning baselines
3. **Analyze generalization** across diverse graph structures and domains
4. **Investigate computational efficiency** and scalability
5. **Provide reproducible experimental framework** for future research

---

## 📁 Complete Project Structure

```
meta-pseudo-community/
├── 📄 README.md                          # Main project documentation
├── 📄 requirements.txt                   # Python dependencies
├── 📄 main.py                           # Main entry point and demo
├── 📄 run_experiments.py                # Automated experiment runner
├── 📄 setup.sh                          # Quick setup script
│
├── 📁 data/                             # Data handling
│   ├── 📄 download_datasets.py          # Download standard datasets
│   └── 📄 preprocess.py                 # Data preprocessing & episode generation
│
├── 📁 models/                           # Core model implementations
│   ├── 📄 base_models.py                # GNN backbone models (GCN, GAT, GIN)
│   ├── 📄 pseudo_labels.py              # Pseudo-label generation strategies
│   └── 📄 meta_learning.py              # Meta-learning algorithms (MAML, Reptile)
│
├── 📁 baselines/                        # Baseline method implementations
│   ├── 📄 traditional_methods.py        # Classical algorithms (Louvain, Spectral, etc.)
│   └── 📄 deep_learning_methods.py      # Deep learning baselines (VGAE, DMoN, etc.)
│
├── 📁 utils/                            # Utility functions
│   ├── 📄 evaluation.py                 # Evaluation metrics and analysis
│   ├── 📄 visualization.py              # Plotting and visualization
│   └── 📄 generate_summary.py           # Results summary generation
│
├── 📁 experiments/                      # Experiment scripts
│   ├── 📄 train_meta_pseudo.py          # Meta-learning training
│   ├── 📄 evaluate.py                   # Model evaluation
│   └── 📄 compare_baselines.py          # Baseline comparison
│
├── 📁 configs/                          # Configuration files
│   ├── 📄 meta_gcn.yaml                 # Meta-GCN configuration
│   ├── 📄 meta_gat.yaml                 # Meta-GAT configuration
│   └── 📄 baselines_comparison.yaml     # Baseline comparison config
│
├── 📁 notebooks/                        # Jupyter analysis notebooks
│   ├── 📄 data_exploration.ipynb        # Dataset analysis and visualization
│   ├── 📄 results_analysis.ipynb        # Performance analysis and comparison
│   └── 📄 training_analysis.ipynb       # Training convergence analysis
│
├── 📁 results/                          # Experimental results (generated)
├── 📁 logs/                             # Training and execution logs (generated)
└── 📁 checkpoints/                      # Model checkpoints (generated)
```

---

## 🚀 Quick Start Guide

### 1. Setup Environment
```bash
# Clone/navigate to project directory
cd meta-pseudo-community

# Run automated setup (RECOMMENDED)
chmod +x setup.sh
./setup.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# PyTorch Geometric may need manual installation:
pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

**Setup Script Features:**
- Automatically creates Python virtual environment
- Installs core dependencies with error handling
- Attempts PyTorch Geometric installation with fallback
- Handles Python 3.12 compatibility issues
- Provides clear error messages and next steps

### 2. Quick Test Run
```bash
# Run minimal experiment to test everything works
python run_experiments.py --quick

# Run single component tests
python main.py --demo
```

### 3. Full Experimental Pipeline
```bash
# Run complete experimental pipeline
python run_experiments.py

# Run specific steps
python run_experiments.py --step data      # Data preparation only
python run_experiments.py --step train     # Training only
python run_experiments.py --step baseline  # Baseline comparison only
```

### 4. Analyze Results
```bash
# Generate summary report
python utils/generate_summary.py --results-dir results

# Interactive analysis
jupyter notebook notebooks/
```

---

## 🔬 Experimental Components

### Meta-Learning Algorithms
- **MAML (Model-Agnostic Meta-Learning)**: Gradient-based meta-learning for fast adaptation
- **Reptile**: First-order approximation for computational efficiency  
- **Custom hybrid approaches**: Combining MAML and Reptile strategies

### Pseudo-Label Generation Strategies
- **Spectral Clustering**: Eigenvalue decomposition methods
- **K-means**: Traditional clustering on embeddings
- **Louvain Algorithm**: Modularity optimization
- **Ensemble Methods**: Combining multiple strategies
- **Adaptive Refinement**: Iterative pseudo-label improvement

### Graph Neural Network Backbones
- **GCN (Graph Convolutional Network)**: Basic spectral approach
- **GAT (Graph Attention Network)**: Attention-based aggregation
- **GIN (Graph Isomorphism Network)**: Powerful graph representation

### Baseline Methods

**Traditional Methods:**
- Louvain Community Detection
- Leiden Algorithm  
- Spectral Clustering
- Infomap
- Label Propagation
- Fluid Communities
- Modularity Optimization

**Deep Learning Methods:**
- DeepWalk
- Node2Vec  
- GraphSAGE
- Deep Graph Infomax (DGI)
- Variational Graph Auto-Encoder (VGAE)
- Deep Modularity Networks (DMoN)

### Evaluation Metrics
- **NMI (Normalized Mutual Information)**: Information-theoretic similarity
- **ARI (Adjusted Rand Index)**: Clustering agreement measure
- **Modularity**: Network modularity score
- **Conductance**: Community quality measure
- **F1-Score**: Classification performance
- **Silhouette Score**: Cluster cohesion measure

---

## 📊 Datasets and Benchmarks

### Standard Graph Datasets
- **Citation Networks**: Cora, CiteSeer, PubMed (via PyTorch Geometric)
- **Social Networks**: Reddit (via PyTorch Geometric) 
- **E-commerce Networks**: Amazon-Computers (via PyTorch Geometric)
- **Academic Networks**: DBLP co-author network (via PyTorch Geometric)
- **Large-scale Networks**: LiveJournal, Orkut, Youtube (manual download if needed)

**Note**: Project optimized for reliable datasets. Large SNAP datasets can be added manually if computational resources allow.

### Synthetic Datasets
- **LFR Benchmark**: Configurable community structure
- **Stochastic Block Models**: Theoretical analysis
- **Planted Partition**: Ground truth communities

### Dataset Statistics Analysis
- Node/edge counts and distributions
- Degree distribution analysis
- Community size and modularity
- Small-world properties
- Power-law characteristics

---

## 🧪 Experimental Workflows

### 1. Data Preparation Pipeline
```python
# Download and preprocess datasets
python data/download_datasets.py --all
python data/preprocess.py --create-episodes --save-processed
```

### 2. Meta-Learning Training
```python
# Train meta-learning models
python experiments/train_meta_pseudo.py --config configs/meta_gcn.yaml
python experiments/train_meta_pseudo.py --config configs/meta_gat.yaml
```

### 3. Comprehensive Evaluation
```python
# Evaluate trained models
python experiments/evaluate.py --model-dir checkpoints/

# Compare with baselines
python experiments/compare_baselines.py --config configs/baselines_comparison.yaml
```

### 4. Results Analysis
```python
# Generate analysis notebooks
jupyter nbconvert --execute notebooks/results_analysis.ipynb
jupyter nbconvert --execute notebooks/training_analysis.ipynb

# Summary statistics
python utils/generate_summary.py --results-dir results
```

---

## 📈 Analysis and Visualization

### Performance Analysis
- **Statistical significance testing** (Mann-Whitney U, Wilcoxon)
- **Effect size analysis** (Cohen's d)
- **Convergence analysis** (training curves, stability)
- **Computational efficiency** (training/inference time)

### Visualization Components
- **Interactive dashboards** (Plotly-based)
- **Training convergence plots** (loss, metrics over time)  
- **Performance comparison charts** (box plots, heatmaps)
- **Graph structure visualization** (community overlays)
- **Radar charts** (multi-metric comparison)

### Statistical Analysis
- **Cross-validation** across multiple datasets
- **Confidence intervals** and error bars
- **Ranking analysis** (average ranks across datasets)
- **Ablation studies** (component contribution analysis)

---

## ⚙️ Configuration and Customization

### Configuration Files (YAML)
```yaml
# Example: configs/meta_gcn.yaml
model:
  type: "meta_gcn"
  hidden_dim: 128
  num_layers: 3
  dropout: 0.1

meta_learning:
  algorithm: "maml"
  inner_lr: 0.01
  meta_lr: 0.001
  adaptation_steps: 5

training:
  epochs: 1000
  patience: 100
  batch_size: 32

pseudo_labels:
  strategy: "ensemble"
  refinement_iterations: 3
  consensus_threshold: 0.7
```

### Customizable Components
- **New GNN architectures**: Extend `base_models.py`
- **Custom pseudo-label strategies**: Add to `pseudo_labels.py`
- **Additional baselines**: Implement in `baselines/`
- **New evaluation metrics**: Add to `utils/evaluation.py`
- **Custom datasets**: Integrate with `data/download_datasets.py`

---

## 🔧 Advanced Usage

### Custom Experiments
```python
# Custom meta-learning experiment
from experiments.train_meta_pseudo import MetaPseudoTrainer

trainer = MetaPseudoTrainer(config_file="custom_config.yaml")
trainer.train()
results = trainer.evaluate()
```

### Ablation Studies
```python
# Component ablation
python run_experiments.py --ablation pseudo_label_strategy
python run_experiments.py --ablation meta_algorithm
python run_experiments.py --ablation gnn_architecture
```

### Hyperparameter Optimization
```python
# Grid search or Bayesian optimization
python experiments/hyperparameter_search.py --method bayesian --trials 100
```

### Large-Scale Experiments
```python
# Distributed training
python run_experiments.py --distributed --nodes 4 --gpus-per-node 2
```

---

## 📝 Results and Reporting

### Automated Reports
- **Experiment summary** (JSON/Markdown format)
- **Performance comparison tables** (LaTeX-ready)
- **Statistical significance results** (p-values, effect sizes)
- **Computational efficiency analysis** (time/memory usage)

### Jupyter Notebook Analysis
- **`data_exploration.ipynb`**: Dataset characteristics and visualization
- **`results_analysis.ipynb`**: Performance comparison and statistical tests  
- **`training_analysis.ipynb`**: Training convergence and meta-learning dynamics

### Export Options
- **CSV files**: Raw experimental results
- **JSON reports**: Structured analysis results
- **PDF/HTML**: Publication-ready figures
- **Interactive dashboards**: Web-based exploration

---

## 🐛 Troubleshooting and FAQ

### Common Issues

**1. CUDA/GPU Setup**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torch-geometric -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
```

**2. Connection Timeouts (Dataset Download)**
```bash
# If SNAP datasets fail to download, use alternative approach:
python data/download_datasets.py --pyg-only  # Only PyTorch Geometric datasets

# Or download large datasets manually:
# wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz -P data/raw/snap/
```

**3. Python 3.12 Compatibility Issues**
```bash
# Some packages may not support Python 3.12 yet
# Use requirements.txt (core dependencies) first:
pip install -r requirements.txt

# Then try optional dependencies:
pip install -r requirements-optional.txt  # If this file exists
```

**4. Memory Issues**
```python
# Reduce batch size in configs
batch_size: 16  # Instead of 32

# Use gradient checkpointing
gradient_checkpointing: true
```

**5. Slow Training**
```python
# Enable mixed precision
use_amp: true

# Reduce model complexity
hidden_dim: 64  # Instead of 128
num_layers: 2   # Instead of 3
```

### Performance Tips
- Use SSD storage for datasets
- Enable gradient clipping for stability
- Use learning rate scheduling
- Implement early stopping
- Monitor GPU memory usage

---

## 🤝 Contributing and Extension

### Adding New Methods
1. **New meta-learning algorithm**: Extend `MetaLearner` class
2. **New pseudo-label strategy**: Add to `PseudoLabelGenerator`
3. **New baseline**: Implement in appropriate baseline file
4. **New dataset**: Add loader to `download_datasets.py`

### Code Style
- Follow PEP 8 Python style guide
- Add comprehensive docstrings
- Include type hints where appropriate
- Write unit tests for new components

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific component tests
python -m pytest tests/test_meta_learning.py
```

---

## 📚 References and Background

### Key Papers
1. **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation"
2. **Graph Neural Networks**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks"
3. **Community Detection**: Fortunato & Hric, "Community detection in networks: A user guide"
4. **Pseudo-Labeling**: Lee, "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method"

### Related Work
- Meta-learning for few-shot learning
- Graph representation learning
- Semi-supervised community detection
- Ensemble methods for graph clustering

---

## 📄 License and Citation

### License
This project is licensed under the MIT License. See LICENSE file for details.

### Citation
```bibtex
@software{meta_pseudo_community_detection,
  title={Meta-Learning Pseudo-Labels for Graph Community Detection},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/meta-pseudo-community}
}
```

---

## 🎯 Future Work and Research Directions

### Immediate Extensions
1. **Multi-task meta-learning**: Simultaneous node/graph-level tasks
2. **Online meta-learning**: Continual adaptation to new domains
3. **Federated community detection**: Privacy-preserving distributed learning
4. **Temporal graphs**: Dynamic community evolution

### Research Questions
1. How does meta-learning performance scale with graph size?
2. What graph properties favor meta-learning approaches?
3. Can meta-learning improve computational efficiency?
4. How to handle overlapping community structures?

### Technical Improvements
1. **Memory-efficient meta-learning**: Reduced memory footprint
2. **Faster adaptation**: Fewer gradient steps for convergence
3. **Uncertainty quantification**: Confidence in pseudo-labels
4. **Interpretability**: Understanding learned adaptation strategies

---

## 📞 Support and Contact

For questions, issues, or contributions:
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Community Q&A and ideas
- **Documentation**: Check README.md and notebook tutorials
- **Examples**: See `main.py` and experiment scripts

---

**Happy Researching! 🎉**

This comprehensive framework provides everything needed for cutting-edge research in meta-learning for graph community detection. The modular design allows for easy extension and customization while maintaining reproducibility and scientific rigor.
