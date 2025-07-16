"""
Visualization utilities cho community detection
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd


def plot_graph_communities(data: Data, labels: Union[torch.Tensor, np.ndarray],
                          layout: str = 'spring', figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None, title: str = "Graph Communities") -> None:
    """Visualize graph với community assignments"""
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get unique communities và assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # Draw nodes by community
    for i, community_id in enumerate(unique_labels):
        community_nodes = [node for node, label in zip(range(len(labels)), labels) 
                          if label == community_id]
        nx.draw_networkx_nodes(G, pos, nodelist=community_nodes, 
                             node_color=[colors[i]], node_size=100, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color='gray')
    
    # Add labels (optional, only for small graphs)
    if len(G.nodes()) < 50:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], markersize=10,
                                 label=f'Community {community_id}')
                      for i, community_id in enumerate(unique_labels)]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_embeddings_2d(embeddings: Union[torch.Tensor, np.ndarray], 
                       labels: Union[torch.Tensor, np.ndarray],
                       method: str = 'tsne', figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None, title: str = "2D Embeddings") -> None:
    """Visualize embeddings trong 2D space"""
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    # Plot
    plt.figure(figsize=figsize)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=f'Community {label}', alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_interactive_graph(data: Data, labels: Union[torch.Tensor, np.ndarray],
                          layout: str = 'spring', save_path: Optional[str] = None,
                          title: str = "Interactive Graph Communities") -> go.Figure:
    """Create interactive graph visualization using Plotly"""
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    
    # Layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Extract coordinates
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    unique_labels = np.unique(labels)
    color_map = px.colors.qualitative.Set3
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Node {node}<br>Community {labels[node]}')
        
        # Color by community
        community_idx = np.where(unique_labels == labels[node])[0][0]
        node_colors.append(color_map[community_idx % len(color_map)])
    
    # Extract edges
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create traces
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=0.5, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers',
                           hoverinfo='text',
                           text=node_text,
                           marker=dict(size=10,
                                     color=node_colors,
                                     line=dict(width=2, color='black')))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(title=title,
                                   titlefont_size=16,
                                   showlegend=False,
                                   hovermode='closest',
                                   margin=dict(b=20,l=5,r=5,t=40),
                                   annotations=[ dict(
                                       text="Community detection visualization",
                                       showarrow=False,
                                       xref="paper", yref="paper",
                                       x=0.005, y=-0.002,
                                       xanchor='left', yanchor='bottom',
                                       font=dict(size=12)
                                   )],
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_metrics_comparison(results: Dict[str, Dict], 
                           metrics: List[str] = ['modularity', 'nmi', 'ari'],
                           save_path: Optional[str] = None) -> go.Figure:
    """Create interactive comparison plot using Plotly"""
    
    methods = list(results.keys())
    
    fig = make_subplots(rows=1, cols=len(metrics),
                       subplot_titles=metrics,
                       specs=[[{'secondary_y': False} for _ in metrics]])
    
    colors = px.colors.qualitative.Set1
    
    for i, metric in enumerate(metrics):
        values = [results[method].get(metric, 0) for method in methods]
        
        fig.add_trace(
            go.Bar(x=methods, y=values, name=metric.upper(),
                  marker_color=colors[i % len(colors)],
                  text=[f'{v:.3f}' for v in values],
                  textposition='auto'),
            row=1, col=i+1
        )
        
        fig.update_xaxes(title_text="Methods", row=1, col=i+1)
        fig.update_yaxes(title_text=metric.upper(), row=1, col=i+1)
    
    fig.update_layout(title="Methods Comparison",
                     showlegend=False,
                     height=500)
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_training_curves(train_losses: List[float], val_losses: Optional[List[float]] = None,
                        train_metrics: Optional[Dict[str, List[float]]] = None,
                        val_metrics: Optional[Dict[str, List[float]]] = None,
                        save_path: Optional[str] = None) -> None:
    """Plot training curves"""
    
    num_plots = 1 + (1 if train_metrics else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='Train Loss', color='blue')
    
    if val_losses:
        axes[0].plot(epochs, val_losses, label='Val Loss', color='red')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training/Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics
    if train_metrics and num_plots > 1:
        for metric_name, values in train_metrics.items():
            axes[1].plot(epochs, values, label=f'Train {metric_name}', linestyle='-')
            
            if val_metrics and metric_name in val_metrics:
                axes[1].plot(epochs, val_metrics[metric_name], 
                           label=f'Val {metric_name}', linestyle='--')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Training/Validation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix_interactive(true_labels: Union[torch.Tensor, np.ndarray],
                                    pred_labels: Union[torch.Tensor, np.ndarray],
                                    save_path: Optional[str] = None) -> go.Figure:
    """Create interactive confusion matrix using Plotly"""
    
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Create heatmap
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="True", color="Count"),
                   x=[f"Pred {i}" for i in range(cm.shape[1])],
                   y=[f"True {i}" for i in range(cm.shape[0])],
                   color_continuous_scale='Blues',
                   text_auto=True)
    
    fig.update_layout(title="Confusion Matrix",
                     xaxis_title="Predicted Labels",
                     yaxis_title="True Labels")
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_community_statistics(labels: Union[torch.Tensor, np.ndarray],
                             save_path: Optional[str] = None) -> None:
    """Plot community statistics"""
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Community size distribution
    axes[0, 0].hist(counts, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Community Size')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Community Size Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot of sizes
    axes[0, 1].boxplot(counts)
    axes[0, 1].set_ylabel('Community Size')
    axes[0, 1].set_title('Community Size Boxplot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Community sizes bar plot
    axes[1, 0].bar(range(len(counts)), counts, alpha=0.7)
    axes[1, 0].set_xlabel('Community ID')
    axes[1, 0].set_ylabel('Size')
    axes[1, 0].set_title('Individual Community Sizes')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_counts = np.sort(counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[1, 1].plot(sorted_counts, cumulative, marker='o', markersize=4)
    axes[1, 1].set_xlabel('Community Size')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution of Sizes')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print("Community Statistics:")
    print(f"Number of communities: {len(unique_labels)}")
    print(f"Mean size: {np.mean(counts):.2f}")
    print(f"Std size: {np.std(counts):.2f}")
    print(f"Min size: {np.min(counts)}")
    print(f"Max size: {np.max(counts)}")
    print(f"Median size: {np.median(counts):.2f}")


def create_result_dashboard(results: Dict[str, Dict], 
                          data: Data, 
                          embeddings: Optional[Dict[str, np.ndarray]] = None,
                          save_dir: Optional[str] = "./results/dashboard/") -> None:
    """Create comprehensive dashboard cho results"""
    
    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Methods comparison
    metrics = ['modularity', 'nmi', 'ari', 'runtime']
    available_metrics = set()
    for result in results.values():
        available_metrics.update(result.keys())
    
    available_metrics = [m for m in metrics if m in available_metrics]
    
    if available_metrics:
        fig_comparison = plot_metrics_comparison(results, available_metrics)
        if save_dir:
            fig_comparison.write_html(f"{save_dir}/methods_comparison.html")
    
    # 2. Graph visualizations for each method
    for method_name, result in results.items():
        if 'labels' in result:
            # 2D graph plot
            plot_graph_communities(
                data, result['labels'], 
                title=f"{method_name} Communities",
                save_path=f"{save_dir}/{method_name}_graph.png" if save_dir else None
            )
            
            # Interactive plot
            fig_interactive = plot_interactive_graph(
                data, result['labels'],
                title=f"{method_name} Interactive Communities"
            )
            if save_dir:
                fig_interactive.write_html(f"{save_dir}/{method_name}_interactive.html")
            
            # Community statistics
            plot_community_statistics(
                result['labels'],
                save_path=f"{save_dir}/{method_name}_stats.png" if save_dir else None
            )
    
    # 3. Embeddings visualization (if available)
    if embeddings:
        for method_name, emb in embeddings.items():
            if method_name in results and 'labels' in results[method_name]:
                # t-SNE plot
                plot_embeddings_2d(
                    emb, results[method_name]['labels'], 
                    method='tsne',
                    title=f"{method_name} t-SNE Embeddings",
                    save_path=f"{save_dir}/{method_name}_tsne.png" if save_dir else None
                )
                
                # PCA plot
                plot_embeddings_2d(
                    emb, results[method_name]['labels'], 
                    method='pca',
                    title=f"{method_name} PCA Embeddings",
                    save_path=f"{save_dir}/{method_name}_pca.png" if save_dir else None
                )
    
    print(f"Dashboard created successfully!" + (f" Saved to {save_dir}" if save_dir else ""))


if __name__ == "__main__":
    # Test visualization utilities
    
    # Create test data
    num_nodes = 50
    edge_index = torch.randint(0, num_nodes, (2, 100))
    x = torch.randn(num_nodes, 64)
    data = Data(x=x, edge_index=edge_index)
    
    # Create test labels
    labels = torch.randint(0, 5, (num_nodes,))
    
    # Test 2D embeddings plot
    print("Testing embeddings visualization...")
    embeddings = torch.randn(num_nodes, 32)
    plot_embeddings_2d(embeddings, labels, method='pca')
    
    # Test graph communities plot
    print("Testing graph communities visualization...")
    plot_graph_communities(data, labels, layout='spring')
    
    # Test community statistics
    print("Testing community statistics...")
    plot_community_statistics(labels)
    
    # Test interactive features
    print("Testing interactive plots...")
    fig_interactive = plot_interactive_graph(data, labels)
    
    # Create dummy results for comparison
    results = {
        'Method A': {'modularity': 0.5, 'nmi': 0.7, 'ari': 0.6, 'runtime': 1.2},
        'Method B': {'modularity': 0.6, 'nmi': 0.8, 'ari': 0.7, 'runtime': 2.1},
        'Method C': {'modularity': 0.4, 'nmi': 0.6, 'ari': 0.5, 'runtime': 0.8}
    }
    
    fig_comparison = plot_metrics_comparison(results)
    
    print("Visualization utilities test completed!")
