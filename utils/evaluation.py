"""
Utility functions cho community detection evaluation
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score, 
    silhouette_score, homogeneity_score, completeness_score,
    v_measure_score, fowlkes_mallows_score
)
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def compute_nmi(true_labels: Union[torch.Tensor, np.ndarray], 
                pred_labels: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute Normalized Mutual Information"""
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    
    return normalized_mutual_info_score(true_labels, pred_labels)


def compute_ari(true_labels: Union[torch.Tensor, np.ndarray], 
                pred_labels: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute Adjusted Rand Index"""
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    
    return adjusted_rand_score(true_labels, pred_labels)


def compute_modularity(edge_index: torch.Tensor, labels: Union[torch.Tensor, np.ndarray],
                      num_nodes: int) -> float:
    """Compute modularity score"""
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Create NetworkX graph
    edges = edge_index.t().cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    # Create communities
    communities = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        community = np.where(labels == label)[0].tolist()
        if len(community) > 0:
            communities.append(community)
    
    if len(communities) <= 1:
        return 0.0
    
    try:
        modularity = nx.community.modularity(G, communities)
    except:
        modularity = 0.0
    
    return modularity


def compute_conductance(edge_index: torch.Tensor, labels: Union[torch.Tensor, np.ndarray],
                       num_nodes: int) -> float:
    """Compute average conductance score"""
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Create NetworkX graph
    edges = edge_index.t().cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    # Compute conductance for each community
    conductances = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        community = np.where(labels == label)[0].tolist()
        if len(community) > 1:
            try:
                conductance = nx.conductance(G, community)
                conductances.append(conductance)
            except:
                continue
    
    return np.mean(conductances) if conductances else 1.0


def compute_silhouette_score(embeddings: Union[torch.Tensor, np.ndarray],
                           labels: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute silhouette score"""
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Need at least 2 unique labels
    if len(np.unique(labels)) < 2:
        return 0.0
    
    try:
        score = silhouette_score(embeddings, labels)
    except:
        score = 0.0
    
    return score


def compute_f1_score(true_labels: Union[torch.Tensor, np.ndarray],
                    pred_labels: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute F1 score for community detection (macro-averaged)"""
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    
    from sklearn.metrics import f1_score
    
    try:
        score = f1_score(true_labels, pred_labels, average='macro')
    except:
        score = 0.0
    
    return score


def compute_purity(true_labels: Union[torch.Tensor, np.ndarray],
                  pred_labels: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute purity score"""
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    
    # Create confusion matrix
    true_clusters = np.unique(true_labels)
    pred_clusters = np.unique(pred_labels)
    
    confusion_matrix = np.zeros((len(true_clusters), len(pred_clusters)))
    
    for i, true_cluster in enumerate(true_clusters):
        for j, pred_cluster in enumerate(pred_clusters):
            confusion_matrix[i, j] = np.sum((true_labels == true_cluster) & 
                                          (pred_labels == pred_cluster))
    
    # Purity = sum of max values in each column / total points
    purity = np.sum(np.max(confusion_matrix, axis=0)) / len(true_labels)
    
    return purity


def compute_all_metrics(true_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
                       pred_labels: Union[torch.Tensor, np.ndarray] = None,
                       edge_index: Optional[torch.Tensor] = None,
                       embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None,
                       num_nodes: Optional[int] = None) -> Dict[str, float]:
    """Compute all evaluation metrics"""
    
    metrics = {}
    
    # Supervised metrics (require ground truth)
    if true_labels is not None and pred_labels is not None:
        metrics['nmi'] = compute_nmi(true_labels, pred_labels)
        metrics['ari'] = compute_ari(true_labels, pred_labels)
        metrics['f1_score'] = compute_f1_score(true_labels, pred_labels)
        metrics['purity'] = compute_purity(true_labels, pred_labels)
        metrics['homogeneity'] = homogeneity_score(
            true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else true_labels,
            pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else pred_labels
        )
        metrics['completeness'] = completeness_score(
            true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else true_labels,
            pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else pred_labels
        )
        metrics['v_measure'] = v_measure_score(
            true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else true_labels,
            pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else pred_labels
        )
    
    # Unsupervised metrics
    if pred_labels is not None and edge_index is not None and num_nodes is not None:
        metrics['modularity'] = compute_modularity(edge_index, pred_labels, num_nodes)
        metrics['conductance'] = compute_conductance(edge_index, pred_labels, num_nodes)
    
    # Embedding-based metrics
    if embeddings is not None and pred_labels is not None:
        metrics['silhouette'] = compute_silhouette_score(embeddings, pred_labels)
    
    return metrics


def evaluate_clustering_stability(clustering_func: callable, data, 
                                 num_runs: int = 10, random_seeds: Optional[List[int]] = None) -> Dict:
    """Evaluate stability of clustering algorithm"""
    
    if random_seeds is None:
        random_seeds = list(range(num_runs))
    
    all_labels = []
    all_metrics = []
    
    for seed in random_seeds:
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Run clustering
        result = clustering_func(data)
        labels = result['labels'] if isinstance(result, dict) else result
        
        all_labels.append(labels)
        
        # Compute metrics if available
        if isinstance(result, dict):
            all_metrics.append(result)
    
    # Compute stability metrics
    stability_metrics = {}
    
    # ARI between different runs
    ari_scores = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari = compute_ari(all_labels[i], all_labels[j])
            ari_scores.append(ari)
    
    stability_metrics['mean_ari_between_runs'] = np.mean(ari_scores)
    stability_metrics['std_ari_between_runs'] = np.std(ari_scores)
    
    # Standard deviation of metrics across runs
    if all_metrics and len(all_metrics) > 1:
        metric_names = all_metrics[0].keys()
        for metric_name in metric_names:
            if isinstance(all_metrics[0][metric_name], (int, float)):
                values = [metrics[metric_name] for metrics in all_metrics]
                stability_metrics[f'mean_{metric_name}'] = np.mean(values)
                stability_metrics[f'std_{metric_name}'] = np.std(values)
    
    return stability_metrics


def compare_methods(results: Dict[str, Dict], 
                   metrics: List[str] = ['modularity', 'nmi', 'ari'],
                   save_path: Optional[str] = None) -> None:
    """Compare different methods và visualize results"""
    
    # Prepare data for plotting
    method_names = list(results.keys())
    
    # Create comparison table
    comparison_data = defaultdict(list)
    
    for method_name, result in results.items():
        for metric in metrics:
            if metric in result:
                comparison_data[metric].append(result[metric])
            else:
                comparison_data[metric].append(0.0)
    
    # Create plots
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = comparison_data[metric]
        
        # Bar plot
        bars = axes[i].bar(method_names, values, alpha=0.7)
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print comparison table
    print("\nMethod Comparison:")
    print("-" * 60)
    print(f"{'Method':<20}", end="")
    for metric in metrics:
        print(f"{metric.upper():<15}", end="")
    print()
    print("-" * 60)
    
    for method_name in method_names:
        print(f"{method_name:<20}", end="")
        for metric in metrics:
            value = results[method_name].get(metric, 0.0)
            print(f"{value:<15.4f}", end="")
        print()


def create_confusion_matrix(true_labels: Union[torch.Tensor, np.ndarray],
                          pred_labels: Union[torch.Tensor, np.ndarray],
                          save_path: Optional[str] = None) -> np.ndarray:
    """Create và visualize confusion matrix"""
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cm


def analyze_community_sizes(labels: Union[torch.Tensor, np.ndarray],
                          save_path: Optional[str] = None) -> Dict:
    """Analyze community size distribution"""
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Compute community sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    stats = {
        'num_communities': len(unique_labels),
        'mean_size': np.mean(counts),
        'std_size': np.std(counts),
        'min_size': np.min(counts),
        'max_size': np.max(counts),
        'median_size': np.median(counts)
    }
    
    # Visualize distribution
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(counts, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Community Size')
    plt.ylabel('Frequency')
    plt.title('Community Size Distribution')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(counts)
    plt.ylabel('Community Size')
    plt.title('Community Size Boxplot')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return stats


def compute_runtime_comparison(methods_results: Dict[str, Dict],
                             save_path: Optional[str] = None) -> None:
    """Compare runtime của các methods"""
    
    method_names = []
    runtimes = []
    
    for method_name, result in methods_results.items():
        if 'runtime' in result:
            method_names.append(method_name)
            runtimes.append(result['runtime'])
    
    if not runtimes:
        print("No runtime information available")
        return
    
    # Visualize
    plt.figure(figsize=(10, 6))
    bars = plt.bar(method_names, runtimes, alpha=0.7)
    plt.xlabel('Methods')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, runtime in zip(bars, runtimes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{runtime:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print runtime statistics
    print("\nRuntime Statistics:")
    print("-" * 40)
    for method_name, runtime in zip(method_names, runtimes):
        print(f"{method_name:<25}: {runtime:.4f}s")
    
    print(f"\nFastest method: {method_names[np.argmin(runtimes)]} ({min(runtimes):.4f}s)")
    print(f"Slowest method: {method_names[np.argmax(runtimes)]} ({max(runtimes):.4f}s)")


def save_results_to_csv(results: Dict[str, Dict], filename: str) -> None:
    """Save results to CSV file"""
    
    import pandas as pd
    
    # Flatten results
    flattened_data = []
    
    for method_name, result in results.items():
        row = {'method': method_name}
        row.update(result)
        flattened_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # Test evaluation utilities
    
    # Create test data
    true_labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    pred_labels = np.array([0, 0, 1, 2, 2, 2, 0, 1, 1, 0])
    
    # Test individual metrics
    print("Testing evaluation metrics:")
    print(f"NMI: {compute_nmi(true_labels, pred_labels):.4f}")
    print(f"ARI: {compute_ari(true_labels, pred_labels):.4f}")
    print(f"Purity: {compute_purity(true_labels, pred_labels):.4f}")
    
    # Test all metrics
    edge_index = torch.randint(0, 10, (2, 20))
    embeddings = np.random.randn(10, 64)
    
    all_metrics = compute_all_metrics(
        true_labels=true_labels,
        pred_labels=pred_labels,
        edge_index=edge_index,
        embeddings=embeddings,
        num_nodes=10
    )
    
    print("\nAll metrics:")
    for metric, value in all_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test community size analysis
    print("\nCommunity size analysis:")
    size_stats = analyze_community_sizes(pred_labels)
    for stat, value in size_stats.items():
        print(f"{stat}: {value}")
    
    print("\nEvaluation utilities test completed!")
