"""
Comprehensive comparison with baseline methods
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pickle
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import EpisodeDataLoader
from baselines.traditional_methods import TraditionalMethodsEvaluator
from baselines.deep_learning_methods import DeepLearningMethodsEvaluator
from utils.evaluation import compute_all_metrics, evaluate_clustering_stability
from utils.visualization import (
    compare_methods, create_result_dashboard, 
    compute_runtime_comparison, save_results_to_csv
)


class BaselineComparator:
    """Comprehensive baseline comparison"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize evaluators
        self.traditional_evaluator = TraditionalMethodsEvaluator()
        self.deep_learning_evaluator = DeepLearningMethodsEvaluator(self.device)
        
        # Initialize data loader
        self.data_loader = EpisodeDataLoader(processed_data_dir="./data/processed")
        
        # Results storage
        self.results = {}
        
    def evaluate_traditional_methods(self, data, dataset_name: str, 
                                   ground_truth=None) -> Dict[str, Dict]:
        """Evaluate traditional community detection methods"""
        
        print(f"Evaluating traditional methods on {dataset_name}...")
        
        # Get methods từ config
        methods_config = self.config.get('baselines', {}).get('traditional', [])
        
        if not methods_config:
            # Use all available methods với default parameters
            return self.traditional_evaluator.evaluate_all_methods(data, ground_truth)
        
        # Evaluate specific methods
        results = {}
        
        for method_config in methods_config:
            method_name = method_config['name']
            params = method_config.get('params', {})
            
            if method_name in self.traditional_evaluator.methods:
                print(f"  Evaluating {method_name}...")
                
                try:
                    # Update method parameters
                    method = self.traditional_evaluator.methods[method_name]
                    for param_name, param_value in params.items():
                        if hasattr(method, param_name):
                            setattr(method, param_name, param_value)
                    
                    # Evaluate method
                    result = method.detect_communities(data)
                    
                    # Add evaluation metrics nếu có ground truth
                    if ground_truth is not None:
                        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
                        predicted_labels = result['labels'].numpy()
                        true_labels = ground_truth.numpy()
                        
                        result['nmi'] = normalized_mutual_info_score(true_labels, predicted_labels)
                        result['ari'] = adjusted_rand_score(true_labels, predicted_labels)
                    
                    results[method_name] = result
                    
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    results[method_name] = {
                        'error': str(e),
                        'labels': torch.zeros(data.num_nodes, dtype=torch.long),
                        'modularity': 0.0,
                        'num_communities': 1,
                        'runtime': 0.0
                    }
        
        return results
    
    def evaluate_deep_learning_methods(self, data, dataset_name: str) -> Dict[str, Dict]:
        """Evaluate deep learning methods"""
        
        print(f"Evaluating deep learning methods on {dataset_name}...")
        
        # Get methods từ config
        methods_config = self.config.get('baselines', {}).get('deep_learning', [])
        
        results = {}
        
        for method_config in methods_config:
            method_name = method_config['name']
            params = method_config.get('params', {})
            
            print(f"  Evaluating {method_name}...")
            
            try:
                if method_name == 'deepwalk':
                    from baselines.deep_learning_methods import DeepWalk
                    method = DeepWalk(**params)
                    result = method.detect_communities(data)
                    
                elif method_name == 'node2vec':
                    from baselines.deep_learning_methods import Node2VecMethod
                    method = Node2VecMethod(**params)
                    result = method.detect_communities(data)
                    
                elif method_name == 'dgi':
                    result = self.deep_learning_evaluator.train_and_evaluate_dgi(data, **params)
                    
                elif method_name == 'vgae':
                    result = self.deep_learning_evaluator.train_and_evaluate_vgae(data, **params)
                    
                elif method_name == 'dmon':
                    result = self.deep_learning_evaluator.train_and_evaluate_dmon(data, **params)
                
                else:
                    print(f"    Unknown method: {method_name}")
                    continue
                
                results[method_name] = result
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                results[method_name] = {
                    'error': str(e),
                    'labels': torch.zeros(data.num_nodes, dtype=torch.long),
                    'modularity': 0.0,
                    'num_communities': 1,
                    'runtime': 0.0
                }
        
        return results
    
    def evaluate_dataset(self, dataset_name: str, num_runs: int = 1) -> Dict:
        """Evaluate all methods trên một dataset"""
        
        print(f"\n{'='*50}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*50}")
        
        if dataset_name not in self.data_loader.get_available_datasets():
            print(f"Dataset {dataset_name} not found!")
            return {}
        
        # Get dataset
        dataset_info = self.data_loader.datasets[dataset_name]
        original_graph = dataset_info['original_graph'].to(self.device)
        
        # Get ground truth nếu có
        ground_truth = None
        if 'ground_truth' in dataset_info and dataset_info['ground_truth']:
            # Convert ground truth communities to labels
            communities = dataset_info['ground_truth']
            ground_truth = torch.zeros(original_graph.num_nodes, dtype=torch.long)
            for comm_id, community in enumerate(communities):
                for node in community:
                    if node < original_graph.num_nodes:
                        ground_truth[node] = comm_id
        
        all_results = {}
        
        # Multiple runs for stability
        for run in range(num_runs):
            if num_runs > 1:
                print(f"\nRun {run + 1}/{num_runs}")
            
            run_results = {}
            
            # Evaluate traditional methods
            traditional_results = self.evaluate_traditional_methods(
                original_graph, dataset_name, ground_truth
            )
            run_results.update({f"traditional_{k}": v for k, v in traditional_results.items()})
            
            # Evaluate deep learning methods (only if graph has features)
            if original_graph.x is not None:
                deep_results = self.evaluate_deep_learning_methods(original_graph, dataset_name)
                run_results.update({f"deep_{k}": v for k, v in deep_results.items()})
            else:
                print("  Skipping deep learning methods (no node features)")
            
            # Store run results
            for method_name, result in run_results.items():
                if method_name not in all_results:
                    all_results[method_name] = []
                all_results[method_name].append(result)
        
        # Aggregate results across runs
        aggregated_results = {}
        for method_name, run_results in all_results.items():
            if len(run_results) == 1:
                aggregated_results[method_name] = run_results[0]
            else:
                # Average metrics across runs
                aggregated = self._aggregate_run_results(run_results)
                aggregated_results[method_name] = aggregated
        
        return aggregated_results
    
    def _aggregate_run_results(self, run_results: List[Dict]) -> Dict:
        """Aggregate results across multiple runs"""
        
        if not run_results:
            return {}
        
        # Check for errors
        if any('error' in result for result in run_results):
            return run_results[0]  # Return first result if there are errors
        
        aggregated = {}
        
        # Get numeric metrics
        numeric_metrics = []
        for result in run_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['num_communities']:
                    if key not in numeric_metrics:
                        numeric_metrics.append(key)
        
        # Aggregate numeric metrics
        for metric in numeric_metrics:
            values = [result.get(metric, 0) for result in run_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        
        # Take other values from first run
        for key, value in run_results[0].items():
            if key not in numeric_metrics:
                aggregated[key] = value
        
        return aggregated
    
    def evaluate_all_datasets(self) -> Dict:
        """Evaluate all methods trên tất cả datasets"""
        
        datasets = self.config.get('data', {}).get('datasets', 
                                   self.data_loader.get_available_datasets())
        
        num_runs = self.config.get('evaluation', {}).get('num_runs', 1)
        
        all_results = {}
        
        for dataset_name in datasets:
            if dataset_name in self.data_loader.get_available_datasets():
                result = self.evaluate_dataset(dataset_name, num_runs)
                if result:
                    all_results[dataset_name] = result
            else:
                print(f"Warning: Dataset {dataset_name} not found")
        
        self.results = all_results
        return all_results
    
    def create_comparison_visualizations(self, save_dir: str):
        """Create comparison visualizations"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nCreating comparison visualizations in {save_dir}...")
        
        for dataset_name, dataset_results in self.results.items():
            dataset_save_dir = f"{save_dir}/{dataset_name}"
            os.makedirs(dataset_save_dir, exist_ok=True)
            
            # Method comparison plots
            metrics = ['modularity', 'nmi', 'ari', 'runtime']
            
            # Filter available metrics
            available_metrics = set()
            for result in dataset_results.values():
                available_metrics.update(result.keys())
            
            # Adjust metrics based on what's available
            comparison_metrics = []
            for metric in metrics:
                if metric in available_metrics:
                    comparison_metrics.append(metric)
                elif f"{metric}_mean" in available_metrics:
                    comparison_metrics.append(f"{metric}_mean")
            
            if comparison_metrics:
                compare_methods(
                    dataset_results, 
                    metrics=comparison_metrics[:3],  # Limit to 3 metrics for readability
                    save_path=f"{dataset_save_dir}/comparison.png"
                )
            
            # Runtime comparison
            compute_runtime_comparison(
                dataset_results,
                save_path=f"{dataset_save_dir}/runtime_comparison.png"
            )
    
    def save_results(self, save_dir: str):
        """Save comparison results"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed results
        with open(f"{save_dir}/baseline_comparison_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save CSV for each dataset
        for dataset_name, dataset_results in self.results.items():
            save_results_to_csv(
                dataset_results, 
                f"{save_dir}/{dataset_name}_comparison.csv"
            )
        
        # Create summary
        self._create_summary_report(save_dir)
        
        print(f"Results saved to {save_dir}")
    
    def _create_summary_report(self, save_dir: str):
        """Create summary report"""
        
        with open(f"{save_dir}/summary_report.txt", 'w') as f:
            f.write("BASELINE COMPARISON SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset_name, dataset_results in self.results.items():
                f.write(f"Dataset: {dataset_name}\n")
                f.write("-" * 30 + "\n")
                
                # Find best methods for each metric
                metrics = ['modularity', 'nmi', 'ari']
                
                for metric in metrics:
                    best_method = None
                    best_score = -float('inf')
                    
                    for method_name, result in dataset_results.items():
                        if 'error' not in result:
                            score = result.get(metric, result.get(f"{metric}_mean", -float('inf')))
                            if score > best_score:
                                best_score = score
                                best_method = method_name
                    
                    if best_method:
                        f.write(f"Best {metric}: {best_method} ({best_score:.4f})\n")
                
                f.write("\n")
        
        print("Summary report created")
    
    def run_comparison(self, save_dir: str):
        """Run complete baseline comparison"""
        
        print("Starting baseline comparison...")
        print(f"Using device: {self.device}")
        
        # Evaluate all datasets
        self.evaluate_all_datasets()
        
        # Create visualizations
        if self.config.get('evaluation', {}).get('create_comparison_plots', True):
            self.create_comparison_visualizations(f"{save_dir}/visualizations")
        
        # Save results
        self.save_results(save_dir)
        
        # Create dashboard if requested
        if self.config.get('output', {}).get('create_dashboard', False):
            print("\nCreating dashboard...")
            # Create a simplified dashboard for baselines
            for dataset_name, dataset_results in self.results.items():
                # Get first successful result's data for visualization
                sample_data = None
                for method_name, result in dataset_results.items():
                    if 'error' not in result and 'labels' in result:
                        # Would need to load original data for visualization
                        break
        
        print("Baseline comparison completed!")


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline Methods')
    parser.add_argument('--config', type=str, 
                       default='configs/baselines_comparison.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, 
                       default="./results/baselines_comparison",
                       help='Directory to save comparison results')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to evaluate')
    parser.add_argument('--num_runs', type=int, default=None,
                       help='Number of runs for stability evaluation')
    parser.add_argument('--methods', nargs='+', default=None,
                       help='Specific methods to evaluate')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config với command line arguments
    if args.datasets:
        config.setdefault('data', {})['datasets'] = args.datasets
    
    if args.num_runs:
        config.setdefault('evaluation', {})['num_runs'] = args.num_runs
    
    if args.methods:
        # Filter methods based on command line input
        traditional_methods = config.get('baselines', {}).get('traditional', [])
        deep_methods = config.get('baselines', {}).get('deep_learning', [])
        
        filtered_traditional = [m for m in traditional_methods if m['name'] in args.methods]
        filtered_deep = [m for m in deep_methods if m['name'] in args.methods]
        
        config['baselines']['traditional'] = filtered_traditional
        config['baselines']['deep_learning'] = filtered_deep
    
    # Create comparator
    comparator = BaselineComparator(config)
    
    # Run comparison
    comparator.run_comparison(args.save_dir)


if __name__ == "__main__":
    main()
