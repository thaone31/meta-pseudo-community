"""
Main script để chạy toàn bộ pipeline
"""

import os
import sys
import argparse
import subprocess
import yaml
from typing import List, Dict


def run_command(command: List[str], description: str) -> bool:
    """Run a command và handle errors"""
    print(f"\n{description}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torch_geometric', 'networkx', 'scikit-learn',
        'matplotlib', 'seaborn', 'pandas', 'tqdm', 'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def setup_data():
    """Setup và preprocess data"""
    print("\n" + "="*50)
    print("SETTING UP DATA")
    print("="*50)
    
    # Download datasets
    success = run_command(
        ["python", "data/download_datasets.py"],
        "Downloading datasets..."
    )
    
    if not success:
        print("Warning: Some datasets might not have been downloaded")
    
    # Preprocess data
    success = run_command(
        ["python", "data/preprocess.py"],
        "Preprocessing data and creating episodes..."
    )
    
    return success


def run_baseline_comparison(config_path: str = "configs/baselines_comparison.yaml"):
    """Run baseline methods comparison"""
    print("\n" + "="*50)
    print("RUNNING BASELINE COMPARISON")
    print("="*50)
    
    success = run_command(
        ["python", "experiments/compare_baselines.py", "--config", config_path],
        "Comparing baseline methods..."
    )
    
    return success


def train_meta_model(config_path: str):
    """Train meta-learning model"""
    print("\n" + "="*50)
    print("TRAINING META-LEARNING MODEL")
    print("="*50)
    
    success = run_command(
        ["python", "experiments/train_meta_pseudo.py", "--config", config_path],
        f"Training meta-learning model with config: {config_path}"
    )
    
    return success


def evaluate_meta_model(model_path: str, config_path: str = None):
    """Evaluate trained meta-learning model"""
    print("\n" + "="*50)
    print("EVALUATING META-LEARNING MODEL")
    print("="*50)
    
    command = ["python", "experiments/evaluate.py", "--model_path", model_path]
    if config_path:
        command.extend(["--config", config_path])
    
    success = run_command(
        command,
        f"Evaluating model: {model_path}"
    )
    
    return success


def run_full_pipeline(meta_config: str = "configs/meta_gcn.yaml"):
    """Run complete research pipeline"""
    print("STARTING FULL RESEARCH PIPELINE")
    print("="*80)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies first")
        return False
    
    # Step 1: Setup data
    if not setup_data():
        print("Failed to setup data")
        return False
    
    # Step 2: Run baseline comparison
    if not run_baseline_comparison():
        print("Warning: Baseline comparison failed")
    
    # Step 3: Train meta-learning model
    if not train_meta_model(meta_config):
        print("Failed to train meta-learning model")
        return False
    
    # Step 4: Evaluate meta-learning model
    # Find the best model from training
    config_name = os.path.splitext(os.path.basename(meta_config))[0]
    model_path = f"./results/{config_name}/best_model.pth"
    
    if os.path.exists(model_path):
        if not evaluate_meta_model(model_path, meta_config):
            print("Warning: Model evaluation failed")
    else:
        print(f"Model not found at {model_path}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED!")
    print("="*80)
    print("\nResults can be found in:")
    print("- ./results/baselines_comparison/")
    print(f"- ./results/{config_name}/")
    print("- ./results/evaluation/")
    
    return True


def run_quick_demo():
    """Run a quick demo với smaller dataset"""
    print("\n" + "="*50)
    print("RUNNING QUICK DEMO")
    print("="*50)
    
    # Create demo config
    demo_config = {
        'model': {
            'type': 'meta_pseudo_gcn',
            'base_model': {
                'encoder_type': 'gcn',
                'hidden_dim': 64,
                'embedding_dim': 32,
                'num_layers': 2,
                'dropout': 0.1
            },
            'meta_learner': {
                'algorithm': 'maml',
                'lr_inner': 0.01,
                'lr_outer': 0.001,
                'num_inner_steps': 3,
                'first_order': False
            },
            'pseudo_label': {
                'methods': ['spectral', 'kmeans'],
                'confidence_threshold': 0.8,
                'adaptive_weights': True
            }
        },
        'data': {
            'datasets': ['cora'],
            'episode_batch_size': 4,
            'subgraph_size_range': [50, 150],
            'num_episodes_per_dataset': 20,
            'train_val_split': 0.8
        },
        'training': {
            'num_epochs': 100,
            'meta_batch_size': 2,
            'evaluation_interval': 20,
            'early_stopping_patience': 50,
            'optimizer': {
                'type': 'adam',
                'lr': 0.001,
                'weight_decay': 1e-5
            }
        },
        'logging': {
            'save_dir': './results/demo',
            'tensorboard': False,
            'save_model_every': 50
        },
        'device': 'auto',
        'seed': 42
    }
    
    # Save demo config
    os.makedirs('./configs', exist_ok=True)
    with open('./configs/demo.yaml', 'w') as f:
        yaml.dump(demo_config, f, default_flow_style=False)
    
    # Run demo pipeline
    print("Setting up minimal data...")
    if not setup_data():
        return False
    
    print("Training demo model...")
    if not train_meta_model('./configs/demo.yaml'):
        return False
    
    print("Demo completed! Check ./results/demo/ for outputs")
    return True


def main():
    parser = argparse.ArgumentParser(description='Meta-Learning Community Detection Pipeline')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full pipeline
    full_parser = subparsers.add_parser('full', help='Run full research pipeline')
    full_parser.add_argument('--config', type=str, default='configs/meta_gcn.yaml',
                           help='Meta-learning config file')
    
    # Individual components
    data_parser = subparsers.add_parser('data', help='Setup data only')
    
    baseline_parser = subparsers.add_parser('baselines', help='Run baseline comparison only')
    baseline_parser.add_argument('--config', type=str, default='configs/baselines_comparison.yaml',
                               help='Baseline config file')
    
    train_parser = subparsers.add_parser('train', help='Train meta-learning model only')
    train_parser.add_argument('--config', type=str, required=True,
                            help='Meta-learning config file')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model only')
    eval_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model')
    eval_parser.add_argument('--config', type=str, default=None,
                           help='Config file (optional)')
    
    # Demo
    demo_parser = subparsers.add_parser('demo', help='Run quick demo')
    
    # Dependencies check
    deps_parser = subparsers.add_parser('check', help='Check dependencies')
    
    args = parser.parse_args()
    
    if args.command == 'full':
        run_full_pipeline(args.config)
    
    elif args.command == 'data':
        setup_data()
    
    elif args.command == 'baselines':
        setup_data()
        run_baseline_comparison(args.config)
    
    elif args.command == 'train':
        train_meta_model(args.config)
    
    elif args.command == 'evaluate':
        evaluate_meta_model(args.model, args.config)
    
    elif args.command == 'demo':
        run_quick_demo()
    
    elif args.command == 'check':
        if check_dependencies():
            print("✓ All dependencies are installed")
        else:
            print("✗ Some dependencies are missing")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
