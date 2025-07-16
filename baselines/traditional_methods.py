"""
Traditional community detection methods
"""

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from typing import Dict, List, Tuple, Optional
import time


class LouvainMethod:
    """Louvain community detection algorithm"""
    
    def __init__(self, resolution: float = 1.0, random_state: int = 42):
        self.resolution = resolution
        self.random_state = random_state
        self.name = "Louvain"
    
    def detect_communities(self, data: Data) -> Dict:
        """Detect communities using Louvain algorithm"""
        
        try:
            import community as community_louvain
            
            start_time = time.time()
            
            # Convert to NetworkX
            G = to_networkx(data, to_undirected=True)
            
            # Apply Louvain algorithm
            partition = community_louvain.best_partition(
                G, resolution=self.resolution, random_state=self.random_state
            )
            
            # Extract labels
            labels = np.array([partition[node] for node in sorted(G.nodes())])
            
            # Compute modularity
            modularity = community_louvain.modularity(partition, G)
            
            end_time = time.time()
            
            return {
                'labels': torch.LongTensor(labels),
                'modularity': modularity,
                'num_communities': len(set(labels)),
                'runtime': end_time - start_time
            }
            
        except ImportError:
            raise ImportError("Please install python-louvain: pip install python-louvain")


class LeidenMethod:
    """Leiden community detection algorithm"""
    
    def __init__(self, resolution: float = 1.0, random_state: int = 42):
        self.resolution = resolution
        self.random_state = random_state
        self.name = "Leiden"
    
    def detect_communities(self, data: Data) -> Dict:
        """Detect communities using Leiden algorithm"""
        
        try:
            import leidenalg
            import igraph as ig
            
            start_time = time.time()
            
            # Convert to igraph
            edge_list = data.edge_index.t().cpu().numpy()
            G = ig.Graph(n=data.num_nodes, edges=edge_list, directed=False)
            
            # Apply Leiden algorithm
            partition = leidenalg.find_partition(
                G, leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.resolution,
                seed=self.random_state
            )
            
            # Extract labels
            labels = np.array(partition.membership)
            
            # Compute modularity
            modularity = partition.modularity
            
            end_time = time.time()
            
            return {
                'labels': torch.LongTensor(labels),
                'modularity': modularity,
                'num_communities': len(set(labels)),
                'runtime': end_time - start_time
            }
            
        except ImportError:
            raise ImportError("Please install leidenalg: pip install leidenalg")


class InfomapMethod:
    """Infomap community detection algorithm"""
    
    def __init__(self, num_trials: int = 10, random_state: int = 42):
        self.num_trials = num_trials
        self.random_state = random_state
        self.name = "Infomap"
    
    def detect_communities(self, data: Data) -> Dict:
        """Detect communities using Infomap algorithm"""
        
        try:
            import infomap
            
            start_time = time.time()
            
            # Create Infomap instance
            im = infomap.Infomap(f"--num-trials {self.num_trials} --seed {self.random_state}")
            
            # Add edges
            edge_list = data.edge_index.t().cpu().numpy()
            for edge in edge_list:
                im.add_link(int(edge[0]), int(edge[1]))
            
            # Run algorithm
            im.run()
            
            # Extract communities
            labels = np.zeros(data.num_nodes, dtype=int)
            for node in im.tree:
                if node.is_leaf:
                    labels[node.node_id] = node.module_id
            
            # Compute modularity
            G = to_networkx(data, to_undirected=True)
            communities = []
            for comm_id in np.unique(labels):
                community = np.where(labels == comm_id)[0].tolist()
                communities.append(community)
            
            modularity = nx.community.modularity(G, communities)
            
            end_time = time.time()
            
            return {
                'labels': torch.LongTensor(labels),
                'modularity': modularity,
                'num_communities': len(set(labels)),
                'codelength': im.codelength,
                'runtime': end_time - start_time
            }
            
        except ImportError:
            raise ImportError("Please install infomap: pip install infomap")


class SpectralClusteringMethod:
    """Spectral clustering for community detection"""
    
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.name = "SpectralClustering"
    
    def detect_communities(self, data: Data) -> Dict:
        """Detect communities using spectral clustering"""
        
        start_time = time.time()
        
        # Build adjacency matrix
        adj_matrix = torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones(data.edge_index.size(1)),
            (data.num_nodes, data.num_nodes)
        ).to_dense().numpy()
        
        # Estimate number of clusters if not provided
        if self.n_clusters is None:
            n_clusters = self._estimate_num_clusters(adj_matrix)
        else:
            n_clusters = self.n_clusters
        
        # Apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=self.random_state
        )
        
        labels = clustering.fit_predict(adj_matrix)
        
        # Compute modularity
        G = to_networkx(data, to_undirected=True)
        communities = []
        for comm_id in np.unique(labels):
            community = np.where(labels == comm_id)[0].tolist()
            communities.append(community)
        
        modularity = nx.community.modularity(G, communities)
        
        end_time = time.time()
        
        return {
            'labels': torch.LongTensor(labels),
            'modularity': modularity,
            'num_communities': len(set(labels)),
            'runtime': end_time - start_time
        }
    
    def _estimate_num_clusters(self, adj_matrix: np.ndarray, max_clusters: int = 20) -> int:
        """Estimate optimal number of clusters using eigenvalues"""
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(adj_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
        
        # Find largest gap
        gaps = np.diff(eigenvals[:max_clusters])
        optimal_k = np.argmax(gaps) + 1
        
        return max(2, min(optimal_k, max_clusters))


class ModularityMethod:
    """Modularity optimization using NetworkX"""
    
    def __init__(self, resolution: float = 1.0, cutoff: int = 1):
        self.resolution = resolution
        self.cutoff = cutoff
        self.name = "ModularityOptimization"
    
    def detect_communities(self, data: Data) -> Dict:
        """Detect communities using greedy modularity optimization"""
        
        start_time = time.time()
        
        # Convert to NetworkX
        G = to_networkx(data, to_undirected=True)
        
        # Greedy modularity communities
        communities = nx.community.greedy_modularity_communities(
            G, cutoff=self.cutoff, best_n=None
        )
        
        # Create labels
        labels = np.zeros(data.num_nodes, dtype=int)
        for comm_id, community in enumerate(communities):
            for node in community:
                labels[node] = comm_id
        
        # Compute modularity
        modularity = nx.community.modularity(G, communities, resolution=self.resolution)
        
        end_time = time.time()
        
        return {
            'labels': torch.LongTensor(labels),
            'modularity': modularity,
            'num_communities': len(communities),
            'runtime': end_time - start_time
        }


class LabelPropagationMethod:
    """Label propagation algorithm"""
    
    def __init__(self, max_iter: int = 100, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.name = "LabelPropagation"
    
    def detect_communities(self, data: Data) -> Dict:
        """Detect communities using label propagation"""
        
        start_time = time.time()
        
        # Convert to NetworkX
        G = to_networkx(data, to_undirected=True)
        
        # Apply label propagation
        np.random.seed(self.random_state)
        communities = nx.community.label_propagation_communities(G)
        
        # Create labels
        labels = np.zeros(data.num_nodes, dtype=int)
        for comm_id, community in enumerate(communities):
            for node in community:
                labels[node] = comm_id
        
        # Compute modularity
        modularity = nx.community.modularity(G, communities)
        
        end_time = time.time()
        
        return {
            'labels': torch.LongTensor(labels),
            'modularity': modularity,
            'num_communities': len(communities),
            'runtime': end_time - start_time
        }


class FluidCommunitiesMethod:
    """Fluid communities algorithm"""
    
    def __init__(self, k: Optional[int] = None, max_iter: int = 100, random_state: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.name = "FluidCommunities"
    
    def detect_communities(self, data: Data) -> Dict:
        """Detect communities using fluid communities algorithm"""
        
        start_time = time.time()
        
        # Convert to NetworkX
        G = to_networkx(data, to_undirected=True)
        
        # Estimate k if not provided
        if self.k is None:
            k = max(2, min(20, data.num_nodes // 50))
        else:
            k = self.k
        
        # Apply fluid communities
        np.random.seed(self.random_state)
        try:
            communities = nx.community.asyn_fluidc(G, k, max_iter=self.max_iter, seed=self.random_state)
            communities = list(communities)
        except:
            # Fallback if algorithm fails
            communities = [set(range(data.num_nodes))]
        
        # Create labels
        labels = np.zeros(data.num_nodes, dtype=int)
        for comm_id, community in enumerate(communities):
            for node in community:
                labels[node] = comm_id
        
        # Compute modularity
        modularity = nx.community.modularity(G, communities)
        
        end_time = time.time()
        
        return {
            'labels': torch.LongTensor(labels),
            'modularity': modularity,
            'num_communities': len(communities),
            'runtime': end_time - start_time
        }


class TraditionalMethodsEvaluator:
    """Evaluator cho traditional community detection methods"""
    
    def __init__(self):
        self.methods = {
            'louvain': LouvainMethod(),
            'leiden': LeidenMethod(),
            'infomap': InfomapMethod(),
            'spectral': SpectralClusteringMethod(),
            'modularity': ModularityMethod(),
            'label_propagation': LabelPropagationMethod(),
            'fluid_communities': FluidCommunitiesMethod()
        }
    
    def evaluate_all_methods(self, data: Data, ground_truth: Optional[torch.Tensor] = None) -> Dict:
        """Evaluate all traditional methods trên dataset"""
        
        results = {}
        
        for method_name, method in self.methods.items():
            print(f"Evaluating {method_name}...")
            
            try:
                result = method.detect_communities(data)
                
                # Add evaluation metrics nếu có ground truth
                if ground_truth is not None:
                    predicted_labels = result['labels'].numpy()
                    true_labels = ground_truth.numpy()
                    
                    result['nmi'] = normalized_mutual_info_score(true_labels, predicted_labels)
                    result['ari'] = adjusted_rand_score(true_labels, predicted_labels)
                
                results[method_name] = result
                
            except Exception as e:
                print(f"Error evaluating {method_name}: {str(e)}")
                results[method_name] = {
                    'error': str(e),
                    'labels': torch.zeros(data.num_nodes, dtype=torch.long),
                    'modularity': 0.0,
                    'num_communities': 1,
                    'runtime': 0.0
                }
        
        return results
    
    def get_best_method(self, results: Dict, metric: str = 'modularity') -> Tuple[str, Dict]:
        """Get best performing method based on specified metric"""
        
        best_method = None
        best_score = -float('inf')
        
        for method_name, result in results.items():
            if 'error' not in result and metric in result:
                score = result[metric]
                if score > best_score:
                    best_score = score
                    best_method = method_name
        
        return best_method, results.get(best_method, {})


if __name__ == "__main__":
    # Test traditional methods
    
    # Create test data
    num_nodes = 100
    edge_index = torch.randint(0, num_nodes, (2, 200))
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    
    # Test evaluator
    evaluator = TraditionalMethodsEvaluator()
    
    # Test single method
    louvain = LouvainMethod()
    try:
        result = louvain.detect_communities(data)
        print(f"Louvain results: {result}")
    except ImportError as e:
        print(f"Louvain not available: {e}")
    
    # Test all methods (some may fail due to missing dependencies)
    print("\nTesting all available methods...")
    all_results = evaluator.evaluate_all_methods(data)
    
    for method_name, result in all_results.items():
        if 'error' not in result:
            print(f"{method_name}: {result['num_communities']} communities, "
                  f"modularity: {result['modularity']:.3f}, "
                  f"runtime: {result['runtime']:.3f}s")
        else:
            print(f"{method_name}: Error - {result['error']}")
    
    # Find best method
    best_method, best_result = evaluator.get_best_method(all_results)
    print(f"\nBest method: {best_method} with modularity {best_result.get('modularity', 0):.3f}")
