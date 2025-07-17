"""
Evaluation script cho trained meta-learning models
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_models import CommunityDetectionModel
from models.meta_learning import MetaPseudoLabelOptimizer
from data.preprocess import EpisodeDataLoader
from utils.evaluation import compute_all_metrics, evaluate_clustering_stability
from utils.visualization import (
    plot_graph_communities, plot_embeddings_2d, 
    create_result_dashboard, plot_metrics_comparison
)


class MetaEvaluator:
    """Evaluator cho trained meta-learning models"""
    
    def __init__(self, config: Dict, model_path: str):
        self.config = config
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize data loader first
        self.data_loader = EpisodeDataLoader(processed_data_dir="./data/processed")
        
        # Load model
        self.model, self.meta_optimizer = self._load_model()
        
        # Results storage
        self.results = {}
        
    def _load_model(self) -> Tuple:
        """Load trained model"""
        print(f"Loading model from {self.model_path}")
        
        # Check if file exists
        if not os.path.exists(self.model_path):
            # Try to find in common locations
            possible_paths = [
                self.model_path,
                f"results/{self.model_path}",
                f"results/meta_gcn/{os.path.basename(self.model_path)}",
                f"results/training/{os.path.basename(self.model_path)}"
            ]
            
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path:
                self.model_path = found_path
                print(f"Found model at: {self.model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path} or common locations")
        
        # Load with weights_only=False to handle numpy objects
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', self.config)
        
        # Create model
        model_config = config['model']['base_model']
        
        # Extract input dimensions from the checkpoint state_dict
        state_dict = checkpoint['model_state_dict']
        input_dims = []
        
        # Find all feature_projectors in state_dict
        for key in state_dict.keys():
            if key.startswith('feature_projectors.') and key.endswith('.weight'):
                # Extract the dimension from key like "feature_projectors.1433.weight"
                parts = key.split('.')
                if len(parts) >= 3 and parts[1].isdigit():
                    input_dims.append(int(parts[1]))
        
        # Also check for default projector size
        if 'feature_projectors.default.weight' in state_dict:
            default_size = state_dict['feature_projectors.default.weight'].size(0)
        else:
            # Fallback: get from encoder input size
            encoder_key = 'encoder.convs.0.lin.weight'
            if encoder_key in state_dict:
                default_size = state_dict[encoder_key].size(1)
            else:
                default_size = 767  # From error message
        
        # Remove duplicates and sort
        input_dims = sorted(list(set(input_dims)))
        if not input_dims:
            input_dims = [default_size]
        
        print(f"Detected input dimensions: {input_dims}, default: {default_size}")
        
        # Use the default size detected from checkpoint
        default_input_dim = default_size
        
        model = CommunityDetectionModel(
            encoder_type=model_config['encoder_type'],
            input_dim=default_input_dim,
            hidden_dim=model_config['hidden_dim'],
            embedding_dim=model_config['embedding_dim'],
            num_classes=model_config.get('num_classes', 10),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.1),
            heads=model_config.get('heads', 8),
            adaptive_input=True  # Enable adaptive input
        ).to(self.device)
        
        # Load model state with strict=False to handle missing keys
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"Missing keys in model: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")  # Show first 5
        print("Model state loaded successfully with adaptive input handling")
        
        model.eval()
        
        # Create meta-optimizer
        meta_optimizer = MetaPseudoLabelOptimizer(
            base_model=model,
            meta_algorithm=config['model']['meta_learner']['algorithm'],
            pseudo_label_methods=config['model']['pseudo_label']['methods']
        )
        
        # Load meta-optimizer state with strict=False
        if 'meta_optimizer_state_dict' in checkpoint:
            try:
                missing_keys, unexpected_keys = meta_optimizer.meta_learner.load_state_dict(
                    checkpoint['meta_optimizer_state_dict'], strict=False
                )
                if missing_keys:
                    print(f"Missing keys in meta-learner: {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"Unexpected keys in meta-learner: {len(unexpected_keys)} keys")
                print("Meta-learner state loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load meta-learner state: {e}")
        
        if 'meta_params' in checkpoint:
            try:
                meta_optimizer.meta_params.load_state_dict(checkpoint['meta_params'], strict=False)
                print("Meta-params loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load meta-params: {e}")
        
        print(f"Model loaded successfully! (Epoch: {checkpoint.get('epoch', 'unknown')})")
        print(f"Best metric: {checkpoint.get('best_metric', 'unknown')}")
        
        return model, meta_optimizer
    
    def evaluate_dataset(self, dataset_name: str, num_runs: int = 5) -> Dict:
        """Evaluate trên một dataset"""
        print(f"\nEvaluating on {dataset_name}...")
        
        if dataset_name not in self.data_loader.get_available_datasets():
            print(f"Dataset {dataset_name} not found!")
            return {}
        
        episodes = self.data_loader.get_dataset_episodes(dataset_name)
        
        if not episodes:
            print(f"No episodes found for {dataset_name}")
            return {}
        
        all_results = []
        all_embeddings = []
        all_labels = []
        
        # Multiple runs for stability
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            
            run_results = []
            run_embeddings = []
            run_labels = []
            
            with torch.no_grad():
                for episode in episodes:
                    # Move to device
                    episode = episode.to(self.device)
                    
                    # Generate pseudo-labels
                    pseudo_labels = self.meta_optimizer._generate_meta_pseudo_labels(episode)
                    
                    # Get embeddings
                    embeddings = self.model.get_embeddings(episode.x, episode.edge_index)
                    
                    # Compute metrics
                    metrics = compute_all_metrics(
                        pred_labels=pseudo_labels,
                        edge_index=episode.edge_index,
                        embeddings=embeddings,
                        num_nodes=episode.num_nodes
                    )
                    
                    run_results.append(metrics)
                    run_embeddings.append(embeddings.cpu().numpy())
                    run_labels.append(pseudo_labels.cpu().numpy())
            
            all_results.append(run_results)
            all_embeddings.append(run_embeddings)
            all_labels.append(run_labels)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        
        # Store results
        result = {
            'metrics': aggregated_results,
            'embeddings': all_embeddings[0],  # Use first run for visualization
            'labels': all_labels[0],
            'episodes': episodes,
            'num_runs': num_runs
        }
        
        self.results[dataset_name] = result
        
        return result
    
    def _aggregate_results(self, all_results: List[List[Dict]]) -> Dict:
        """Aggregate results across multiple runs và episodes"""
        
        # Flatten all metrics
        all_metrics = []
        for run_results in all_results:
            all_metrics.extend(run_results)
        
        if not all_metrics:
            return {}
        
        # Aggregate by metric name
        aggregated = {}
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values and all(isinstance(v, (int, float)) for v in values):
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
                aggregated[f'{metric_name}_min'] = np.min(values)
                aggregated[f'{metric_name}_max'] = np.max(values)
        
        return aggregated
    
    def evaluate_all_datasets(self) -> Dict:
        """Evaluate trên tất cả available datasets"""
        datasets = self.config.get('evaluation', {}).get('datasets', 
                                  self.data_loader.get_available_datasets())
        
        num_runs = self.config.get('evaluation', {}).get('num_runs', 5)
        
        all_results = {}
        
        for dataset_name in datasets:
            if dataset_name in self.data_loader.get_available_datasets():
                result = self.evaluate_dataset(dataset_name, num_runs)
                if result:
                    all_results[dataset_name] = result
        
        return all_results
    
    def create_visualizations(self, save_dir: str):
        """Create visualizations cho results"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nCreating visualizations in {save_dir}...")
        
        for dataset_name, result in self.results.items():
            dataset_save_dir = f"{save_dir}/{dataset_name}"
            os.makedirs(dataset_save_dir, exist_ok=True)
            
            episodes = result['episodes']
            embeddings_list = result['embeddings']
            labels_list = result['labels']
            
            # Plot a few episodes
            num_plots = min(3, len(episodes))
            
            for i in range(num_plots):
                episode = episodes[i]
                embeddings = embeddings_list[i]
                labels = labels_list[i]
                
                # Graph visualization
                plot_graph_communities(
                    episode, labels,
                    title=f"{dataset_name} Episode {i+1} Communities",
                    save_path=f"{dataset_save_dir}/episode_{i+1}_graph.png"
                )
                
                # Embeddings visualization
                plot_embeddings_2d(
                    embeddings, labels, method='tsne',
                    title=f"{dataset_name} Episode {i+1} t-SNE",
                    save_path=f"{dataset_save_dir}/episode_{i+1}_tsne.png"
                )
                
                plot_embeddings_2d(
                    embeddings, labels, method='pca',
                    title=f"{dataset_name} Episode {i+1} PCA",
                    save_path=f"{dataset_save_dir}/episode_{i+1}_pca.png"
                )
    
    def compare_with_baselines(self, baseline_results_path: Optional[str] = None) -> Dict:
        """Compare với baseline methods"""
        if not baseline_results_path or not os.path.exists(baseline_results_path):
            print("No baseline results found for comparison")
            return {}
        
        # Load baseline results
        with open(baseline_results_path, 'rb') as f:
            baseline_results = pickle.load(f)
        
        comparison = {}
        
        for dataset_name in self.results.keys():
            if dataset_name in baseline_results:
                comparison[dataset_name] = {
                    'meta_learning': self.results[dataset_name]['metrics'],
                    'baselines': baseline_results[dataset_name]
                }
        
        return comparison
    
    def save_results(self, save_dir: str):
        """Save evaluation results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed results
        with open(f"{save_dir}/evaluation_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary metrics
        summary = {}
        for dataset_name, result in self.results.items():
            summary[dataset_name] = result['metrics']
        
        with open(f"{save_dir}/summary_metrics.pkl", 'wb') as f:
            pickle.dump(summary, f)
        
        # Save as CSV
        try:
            import pandas as pd
            
            rows = []
            for dataset_name, result in self.results.items():
                row = {'dataset': dataset_name}
                row.update(result['metrics'])
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(f"{save_dir}/evaluation_summary.csv", index=False)
            print(f"CSV summary saved to {save_dir}/evaluation_summary.csv")
        except ImportError:
            print("Pandas not available, skipping CSV export")
        
        print(f"Results saved to {save_dir}")
        
        # Print summary
        print("\nEvaluation Summary:")
        print("=" * 60)
        for dataset_name, result in self.results.items():
            metrics = result['metrics']
            print(f"\n{dataset_name}:")
            for metric_name, value in metrics.items():
                if metric_name.endswith('_mean'):
                    base_name = metric_name[:-5]
                    std_name = f"{base_name}_std"
                    if std_name in metrics:
                        print(f"  {base_name}: {value:.4f} ± {metrics[std_name]:.4f}")
    
    def run_evaluation(self, save_dir: str):
        """Run complete evaluation pipeline"""
        print("Starting evaluation...")
        
        # Evaluate all datasets
        self.evaluate_all_datasets()
        
        # Create visualizations
        self.create_visualizations(f"{save_dir}/visualizations")
        
        # Save results
        self.save_results(save_dir)
        
        print("Evaluation completed!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Meta-Learning Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional)')
    parser.add_argument('--save_dir', type=str, default="./results/evaluation",
                       help='Directory to save evaluation results')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to evaluate (optional)')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs for stability evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default config
        config = {
            'evaluation': {
                'datasets': args.datasets,
                'num_runs': args.num_runs
            }
        }
    
    # Override config with command line arguments
    if args.datasets:
        config.setdefault('evaluation', {})['datasets'] = args.datasets
    if args.num_runs:
        config.setdefault('evaluation', {})['num_runs'] = args.num_runs
    
    # Create evaluator
    evaluator = MetaEvaluator(config, args.model_path)
    
    # Run evaluation
    evaluator.run_evaluation(args.save_dir)


if __name__ == "__main__":
    main()
