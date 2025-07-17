"""
Pseudo-label generat        self.methods = {
            'spectral': self._spectral_clustering,
            'kmeans': self._kmeans_clustering, 
            'dbscan': self._dbscan_clustering,
            'louvain': self._louvain_clustering,
            'leiden': self._leiden_clustering,
            'modularity': self._modularity_clustering,
            'fallback': self._simple_fallback_clustering
        }efinement module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import networkx as nx
from torch_geometric.utils import to_networkx


class PseudoLabelGenerator:
    """Generate pseudo-labels cho community detection"""
    
    def __init__(self, method: str = 'spectral', confidence_threshold: float = 0.8):
        self.method = method
        self.confidence_threshold = confidence_threshold
        
        # Available methods
        self.methods = {
            'spectral': self._spectral_clustering,
            'kmeans': self._kmeans_clustering,
            'dbscan': self._dbscan_clustering,
            'louvain': self._louvain_clustering,
            'leiden': self._leiden_clustering,
            'modularity': self._modularity_clustering,
            'fallback': self._simple_fallback_clustering
        }
        
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
    
    def generate(self, data: Data, embeddings: Optional[torch.Tensor] = None,
                num_clusters: Optional[int] = None) -> Dict:
        """Generate pseudo-labels cho graph"""
        
        # Estimate number of clusters if not provided
        if num_clusters is None:
            num_clusters = self._estimate_num_clusters(data, embeddings)
        
        # Generate labels
        labels, confidence_scores = self.methods[self.method](data, embeddings, num_clusters)
        
        # Filter high-confidence labels
        high_conf_mask = confidence_scores >= self.confidence_threshold
        
        result = {
            'labels': labels,
            'confidence_scores': confidence_scores,
            'high_confidence_mask': high_conf_mask,
            'num_clusters': num_clusters,
            'num_high_conf': high_conf_mask.sum().item()
        }
        
        return result
    
    def _estimate_num_clusters(self, data: Data, embeddings: Optional[torch.Tensor] = None) -> int:
        """Estimate optimal number of clusters"""
        
        # Conservative estimation - ensure it doesn't exceed model capacity
        min_clusters = 2
        max_clusters = min(10, data.num_nodes // 5, data.num_nodes - 1)  # More conservative
        
        # Ensure at least 2 clusters but not more than reasonable
        if max_clusters < min_clusters:
            max_clusters = min_clusters
        
        if embeddings is not None:
            # Use embeddings for estimation
            X = embeddings.detach().cpu().numpy()
        else:
            # Use adjacency matrix
            edge_index = data.edge_index
            adj_matrix = torch.sparse_coo_tensor(
                edge_index,
                torch.ones(edge_index.size(1)),
                (data.num_nodes, data.num_nodes)
            ).to_dense().numpy()
            X = adj_matrix
        
        # Try different number of clusters and find best silhouette score
        best_score = -1
        best_k = min_clusters
        
        for k in range(min_clusters, min(max_clusters + 1, X.shape[0])):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(X, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        return best_k
    
    def _spectral_clustering(self, data: Data, embeddings: Optional[torch.Tensor],
                           num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Spectral clustering on adjacency matrix"""
        
        # Handle edge case: if we have fewer nodes than clusters
        if data.num_nodes < num_clusters:
            labels = torch.arange(data.num_nodes, dtype=torch.long)
            confidence_scores = torch.ones(data.num_nodes, dtype=torch.float)
            return labels, confidence_scores
        
        # Handle case with no edges
        if data.edge_index.size(1) == 0:
            labels = torch.arange(data.num_nodes, dtype=torch.long) % num_clusters
            confidence_scores = torch.ones(data.num_nodes, dtype=torch.float) * 0.5
            return labels, confidence_scores
        
        edge_index = data.edge_index
        
        # Build adjacency matrix
        adj_matrix = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1)),
            (data.num_nodes, data.num_nodes)
        ).to_dense().numpy()
        
        try:
            # Spectral clustering
            clustering = SpectralClustering(
                n_clusters=num_clusters,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42
            )
            
            labels = clustering.fit_predict(adj_matrix)
            
            # Confidence scores (simplified as spectral clustering doesn't provide them directly)
            confidence_scores = np.ones(len(labels)) * 0.8
            
            return torch.LongTensor(labels), torch.FloatTensor(confidence_scores)
            
        except Exception as e:
            print(f"Warning: Spectral clustering failed ({e}), using fallback clustering")
            # Fallback: assign sequential labels
            labels = torch.arange(data.num_nodes, dtype=torch.long) % num_clusters
            confidence_scores = torch.ones(data.num_nodes, dtype=torch.float) * 0.5
            return labels, confidence_scores
            
            # Compute confidence scores based on eigenvalues
            eigenvals = clustering.eigenvalues_
            if len(eigenvals) > 1:
                gap = eigenvals[-2] - eigenvals[-1]  # Spectral gap
                confidence_scores = torch.ones(data.num_nodes) * min(gap, 1.0)
            else:
                confidence_scores = torch.ones(data.num_nodes) * 0.5
                
        except:
            # Fallback to random
            labels = np.random.randint(0, num_clusters, data.num_nodes)
            confidence_scores = torch.ones(data.num_nodes) * 0.1
        
        return torch.LongTensor(labels), confidence_scores
    
    def _kmeans_clustering(self, data: Data, embeddings: Optional[torch.Tensor],
                         num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """K-means clustering"""
        
        # Handle edge case: if we have fewer nodes than clusters
        if data.num_nodes < num_clusters:
            # Each node is its own cluster
            labels = torch.arange(data.num_nodes, dtype=torch.long)
            confidence_scores = torch.ones(data.num_nodes, dtype=torch.float)
            return labels, confidence_scores
        
        if embeddings is not None:
            X = embeddings.detach().cpu().numpy()
        else:
            # Use node features hoặc adjacency
            if data.x is not None:
                X = data.x.cpu().numpy()
            else:
                edge_index = data.edge_index
                adj_matrix = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(edge_index.size(1)),
                    (data.num_nodes, data.num_nodes)
                ).to_dense().numpy()
                X = adj_matrix
        
        # Additional check for X shape
        if X.shape[0] < num_clusters:
            labels = torch.arange(X.shape[0], dtype=torch.long)
            confidence_scores = torch.ones(X.shape[0], dtype=torch.float)
            return labels, confidence_scores
        
        try:
            # K-means
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Confidence scores based on distance to cluster centers
            distances = kmeans.transform(X)
            min_distances = np.min(distances, axis=1)
            max_distance = np.max(min_distances)
            confidence_scores = 1.0 - (min_distances / (max_distance + 1e-8))
            
            return torch.LongTensor(labels), torch.FloatTensor(confidence_scores)
            
        except Exception as e:
            print(f"Warning: K-means failed ({e}), using fallback clustering")
            # Fallback: assign sequential labels
            labels = torch.arange(data.num_nodes, dtype=torch.long) % num_clusters
            confidence_scores = torch.ones(data.num_nodes, dtype=torch.float) * 0.5
            return labels, confidence_scores
    
    def _dbscan_clustering(self, data: Data, embeddings: Optional[torch.Tensor],
                         num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """DBSCAN clustering"""
        
        # Handle edge case: if we have very few nodes
        if data.num_nodes < 2:
            labels = torch.zeros(data.num_nodes, dtype=torch.long)
            confidence_scores = torch.ones(data.num_nodes, dtype=torch.float)
            return labels, confidence_scores
        
        if embeddings is not None:
            X = embeddings.detach().cpu().numpy()
        else:
            if data.x is not None:
                X = data.x.cpu().numpy()
            else:
                # Use adjacency representation
                edge_index = data.edge_index
                adj_matrix = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(edge_index.size(1)),
                    (data.num_nodes, data.num_nodes)
                ).to_dense().numpy()
                X = adj_matrix
        
        try:
            # DBSCAN
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            labels = dbscan.fit_predict(X)
            
            # Handle noise points (label -1)
            unique_labels = np.unique(labels)
            if -1 in unique_labels:
                # Reassign noise points to nearest cluster
                noise_mask = labels == -1
                if np.any(~noise_mask):  # If there are non-noise points
                    # Simple reassignment based on majority cluster
                    valid_labels = labels[~noise_mask]
                    most_common = np.bincount(valid_labels).argmax()
                    labels[noise_mask] = most_common
                else:
                    # All points are noise, assign to single cluster
                    labels = np.zeros_like(labels)
            
            # Confidence scores (simplified)
            confidence_scores = np.ones(len(labels)) * 0.7  # DBSCAN doesn't provide natural confidence
            
            return torch.LongTensor(labels), torch.FloatTensor(confidence_scores)
            
        except Exception as e:
            print(f"Warning: DBSCAN failed ({e}), using fallback clustering")
            # Fallback: assign sequential labels
            labels = torch.arange(data.num_nodes, dtype=torch.long) % max(1, num_clusters)
            confidence_scores = torch.ones(data.num_nodes, dtype=torch.float) * 0.5
            return labels, confidence_scores
    
    def _louvain_clustering(self, data: Data, embeddings: Optional[torch.Tensor],
                          num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Louvain community detection"""
        
        try:
            import community as community_louvain
            
            # Convert to NetworkX
            G = to_networkx(data, to_undirected=True)
            
            # Louvain algorithm
            partition = community_louvain.best_partition(G)
            
            # Extract labels
            labels = np.array([partition[node] for node in sorted(G.nodes())])
            
            # Confidence scores based on modularity
            modularity = community_louvain.modularity(partition, G)
            confidence_scores = torch.ones(data.num_nodes) * max(0.1, modularity)
            
        except ImportError:
            print("Warning: python-louvain not installed, using spectral clustering")
            return self._spectral_clustering(data, embeddings, num_clusters)
        except:
            # Fallback
            labels = np.random.randint(0, num_clusters, data.num_nodes)
            confidence_scores = torch.ones(data.num_nodes) * 0.1
        
        return torch.LongTensor(labels), confidence_scores
    
    def _leiden_clustering(self, data: Data, embeddings: Optional[torch.Tensor],
                         num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Leiden community detection"""
        
        try:
            import leidenalg
            import igraph as ig
            
            # Convert to igraph
            edge_list = data.edge_index.t().cpu().numpy()
            G = ig.Graph(n=data.num_nodes, edges=edge_list, directed=False)
            
            # Leiden algorithm
            partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
            
            # Extract labels
            labels = np.array(partition.membership)
            
            # Confidence scores based on modularity
            modularity = partition.modularity
            confidence_scores = torch.ones(data.num_nodes) * max(0.1, modularity)
            
        except ImportError:
            print("Warning: leidenalg not installed, using spectral clustering")
            return self._spectral_clustering(data, embeddings, num_clusters)
        except:
            # Fallback
            labels = np.random.randint(0, num_clusters, data.num_nodes)
            confidence_scores = torch.ones(data.num_nodes) * 0.1
        
        return torch.LongTensor(labels), confidence_scores
    
    def _modularity_clustering(self, data: Data, embeddings: Optional[torch.Tensor],
                             num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Modularity-based clustering using NetworkX"""
        
        try:
            # Convert to NetworkX
            G = to_networkx(data, to_undirected=True)
            
            # Greedy modularity communities
            communities = nx.community.greedy_modularity_communities(G)
            
            # Create labels
            labels = np.zeros(data.num_nodes)
            for i, community in enumerate(communities):
                for node in community:
                    labels[node] = i
            
            # Modularity score as confidence
            modularity = nx.community.modularity(G, communities)
            confidence_scores = torch.ones(data.num_nodes) * max(0.1, modularity)
            
        except:
            # Fallback to spectral
            return self._spectral_clustering(data, embeddings, num_clusters)
        
        return torch.LongTensor(labels), confidence_scores
    
    def _simple_fallback_clustering(self, data: Data, embeddings: Optional[torch.Tensor],
                                   num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple fallback clustering for edge cases"""
        
        # Very simple clustering: just assign labels in round-robin fashion
        if data.num_nodes <= num_clusters:
            # Each node gets its own cluster (or sequential assignment)
            labels = torch.arange(data.num_nodes, dtype=torch.long)
        else:
            # Round-robin assignment
            labels = torch.arange(data.num_nodes, dtype=torch.long) % num_clusters
        
        # Low confidence for fallback method
        confidence_scores = torch.ones(data.num_nodes, dtype=torch.float) * 0.3
        
        return labels, confidence_scores


class PseudoLabelRefiner(nn.Module):
    """Neural module to refine pseudo-labels"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.refiner = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # Node + cluster center embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),  # Confidence score
            nn.Sigmoid()
        )
    
    def forward(self, node_embeddings: torch.Tensor, 
                cluster_centers: torch.Tensor,
                assignments: torch.Tensor) -> torch.Tensor:
        """Refine confidence scores cho pseudo-labels"""
        
        # Get assigned cluster centers for each node
        assigned_centers = cluster_centers[assignments]
        
        # Concatenate node and cluster embeddings
        combined = torch.cat([node_embeddings, assigned_centers], dim=1)
        
        # Predict refined confidence scores
        refined_confidence = self.refiner(combined).squeeze(-1)
        
        return refined_confidence


class AdaptivePseudoLabelGenerator:
    """Adaptive pseudo-label generator that can switch between different methods"""
    
    def __init__(self, methods: List[str] = ['spectral', 'louvain', 'kmeans']):
        self.generators = {method: PseudoLabelGenerator(method) for method in methods}
        self.method_weights = {method: 1.0 for method in methods}
    
    def generate_ensemble(self, data: Data, embeddings: Optional[torch.Tensor] = None,
                         num_clusters: Optional[int] = None) -> Dict:
        """Generate ensemble pseudo-labels from multiple methods"""
        
        all_results = {}
        all_labels = []
        all_confidences = []
        successful_methods = []
        
        for method, generator in self.generators.items():
            try:
                result = generator.generate(data, embeddings, num_clusters)
                all_results[method] = result
                all_labels.append(result['labels'])
                all_confidences.append(result['confidence_scores'])
                successful_methods.append(method)
            except Exception as e:
                print(f"Warning: Method {method} failed ({e}), skipping...")
                continue
        
        # If no methods succeeded, use fallback
        if not successful_methods:
            print("Warning: All methods failed, using fallback clustering")
            fallback_generator = PseudoLabelGenerator('fallback')
            result = fallback_generator.generate(data, embeddings, num_clusters)
            all_results['fallback'] = result
            all_labels = [result['labels']]
            all_confidences = [result['confidence_scores']]
            successful_methods = ['fallback']
        
        # Weighted ensemble (only use successful methods)
        active_weights = [self.method_weights.get(method, 1.0) for method in successful_methods]
        ensemble_labels, ensemble_confidence = self._ensemble_labels(
            all_labels, all_confidences, active_weights
        )
        
        ensemble_result = {
            'labels': ensemble_labels,
            'confidence_scores': ensemble_confidence,
            'individual_results': all_results,
            'successful_methods': successful_methods,
            'num_clusters': num_clusters or all_results[successful_methods[0]]['num_clusters']
        }
        
        return ensemble_result
    
    def _ensemble_labels(self, all_labels: List[torch.Tensor], 
                        all_confidences: List[torch.Tensor],
                        weights: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble multiple label predictions"""
        
        # Handle empty input
        if not all_labels or len(all_labels) == 0:
            # Return fallback labels
            num_nodes = 2  # Minimum 
            labels = torch.zeros(num_nodes, dtype=torch.long)
            confidences = torch.ones(num_nodes, dtype=torch.float) * 0.1
            return labels, confidences
        
        # Check if all tensors are valid
        valid_indices = []
        for i, (labels, confidences) in enumerate(zip(all_labels, all_confidences)):
            if labels.numel() > 0 and confidences.numel() > 0:
                valid_indices.append(i)
        
        if not valid_indices:
            # No valid predictions, use fallback
            num_nodes = 2
            labels = torch.zeros(num_nodes, dtype=torch.long)
            confidences = torch.ones(num_nodes, dtype=torch.float) * 0.1
            return labels, confidences
        
        # Filter to only valid predictions
        all_labels = [all_labels[i] for i in valid_indices]
        all_confidences = [all_confidences[i] for i in valid_indices]
        weights = [weights[i] for i in valid_indices]
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted voting
        num_nodes = all_labels[0].size(0)
        
        # Handle empty nodes case
        if num_nodes == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)
        
        # Collect all unique labels
        all_unique_labels = set()
        for labels in all_labels:
            if labels.numel() > 0:
                all_unique_labels.update(labels.tolist())
        
        # Handle case with no valid labels
        if not all_unique_labels:
            labels = torch.zeros(num_nodes, dtype=torch.long)
            confidences = torch.ones(num_nodes, dtype=torch.float) * 0.1
            return labels, confidences
        
        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_unique_labels))}
        num_classes = len(label_to_idx)
        
        # Ensure we have valid classes
        if num_classes == 0:
            labels = torch.zeros(num_nodes, dtype=torch.long)
            confidences = torch.ones(num_nodes, dtype=torch.float) * 0.1
            return labels, confidences
        
        # Vote matrix
        votes = torch.zeros(num_nodes, num_classes)
        
        for i, (labels, confidences, weight) in enumerate(zip(all_labels, all_confidences, weights)):
            for node_idx in range(min(num_nodes, labels.size(0))):
                label = labels[node_idx].item()
                conf = confidences[node_idx].item()
                if label in label_to_idx:
                    class_idx = label_to_idx[label]
                    votes[node_idx, class_idx] += weight * conf
        
        # Check votes validity before max operation
        if votes.size(1) == 0:
            labels = torch.zeros(num_nodes, dtype=torch.long)
            confidences = torch.ones(num_nodes, dtype=torch.float) * 0.1
            return labels, confidences
        
        # Final labels and confidence
        ensemble_confidence, ensemble_label_indices = torch.max(votes, dim=1)
        
        # Map back to labels
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        ensemble_labels = torch.LongTensor([idx_to_label[idx.item()] for idx in ensemble_label_indices])
        
        # Normalize confidence scores
        max_conf = ensemble_confidence.max()
        if max_conf > 0:
            ensemble_confidence = ensemble_confidence / max_conf
        else:
            ensemble_confidence = torch.ones_like(ensemble_confidence) * 0.1
        
        return ensemble_labels, ensemble_confidence
    
    def update_method_weights(self, performance_scores: Dict[str, float]):
        """Update method weights based on performance"""
        
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for method in self.method_weights:
                if method in performance_scores:
                    self.method_weights[method] = performance_scores[method] / total_score
                else:
                    self.method_weights[method] = 0.1  # Small default weight


if __name__ == "__main__":
    # Test pseudo-label generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    num_nodes = 100
    edge_index = torch.randint(0, num_nodes, (2, 200))
    x = torch.randn(num_nodes, 64)
    data = Data(x=x, edge_index=edge_index)
    
    # Test generator
    generator = PseudoLabelGenerator(method='spectral')
    result = generator.generate(data)
    
    print(f"Generated {result['num_clusters']} clusters")
    print(f"High confidence nodes: {result['num_high_conf']}/{num_nodes}")
    print(f"Labels shape: {result['labels'].shape}")
    print(f"Confidence shape: {result['confidence_scores'].shape}")
