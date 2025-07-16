"""
Data preprocessing và episode generation cho meta-learning
"""

import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx, from_networkx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
import pickle
from typing import List, Dict, Tuple, Optional
import random


class GraphPreprocessor:
    """Xử lý và chuẩn bị dữ liệu graph cho meta-learning"""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(processed_data_dir, exist_ok=True)
        
    def preprocess_all(self):
        """Preprocess tất cả datasets"""
        print("Preprocessing datasets...")
        
        # Process PyG datasets
        self.process_pyg_datasets()
        
        # Process SNAP datasets
        self.process_snap_datasets()
        
        # Process LFR datasets
        self.process_lfr_datasets()
        
        print("All datasets preprocessed successfully!")
    
    def process_pyg_datasets(self):
        """Process PyTorch Geometric datasets"""
        from torch_geometric.datasets import Planetoid, Reddit, Amazon, DBLP
        
        datasets = {
            'Cora': Planetoid(root=f"{self.raw_data_dir}/pyg", name='Cora'),
            'CiteSeer': Planetoid(root=f"{self.raw_data_dir}/pyg", name='CiteSeer'),
            'PubMed': Planetoid(root=f"{self.raw_data_dir}/pyg", name='PubMed'),
            'Reddit': Reddit(root=f"{self.raw_data_dir}/pyg/Reddit"),
            'Amazon': Amazon(root=f"{self.raw_data_dir}/pyg/Amazon", name="Computers"),
            'DBLP': DBLP(root=f"{self.raw_data_dir}/pyg/DBLP")
        }
        
        for name, dataset in datasets.items():
            print(f"Processing {name}...")
            data = dataset[0]
            
            # Chuẩn hóa features nếu có
            if data.x is not None:
                scaler = StandardScaler()
                data.x = torch.FloatTensor(scaler.fit_transform(data.x.numpy()))
            
            # Tạo episodes từ dataset
            episodes = self.create_episodes_from_graph(data, num_episodes=50)
            
            # Lưu processed data
            processed_data = {
                'original_graph': data,
                'episodes': episodes,
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'num_features': data.x.size(1) if data.x is not None else 0
            }
            
            with open(f"{self.processed_data_dir}/{name.lower()}.pkl", 'wb') as f:
                pickle.dump(processed_data, f)
    
    def process_snap_datasets(self):
        """Process SNAP datasets"""
        snap_files = {
            'LiveJournal': f"{self.raw_data_dir}/snap/LiveJournal.txt.gz",
            'Orkut': f"{self.raw_data_dir}/snap/Orkut.txt.gz",
            'Youtube': f"{self.raw_data_dir}/snap/Youtube.txt.gz"
        }
        
        for name, file_path in snap_files.items():
            if os.path.exists(file_path):
                print(f"Processing {name}...")
                
                # Load graph from edgelist
                G = nx.read_edgelist(file_path, nodetype=int)
                
                # Convert to largest connected component
                if not nx.is_connected(G):
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                
                # Convert to PyG format
                data = from_networkx(G)
                
                # Tạo synthetic node features
                data.x = torch.randn(data.num_nodes, 128)  # 128-dim random features
                
                # Tạo episodes
                episodes = self.create_episodes_from_graph(data, num_episodes=30)
                
                processed_data = {
                    'original_graph': data,
                    'episodes': episodes,
                    'num_nodes': data.num_nodes,
                    'num_edges': data.num_edges,
                    'num_features': 128
                }
                
                with open(f"{self.processed_data_dir}/{name.lower()}.pkl", 'wb') as f:
                    pickle.dump(processed_data, f)
    
    def process_lfr_datasets(self):
        """Process LFR benchmark datasets"""
        lfr_dir = f"{self.raw_data_dir}/lfr"
        
        if not os.path.exists(lfr_dir):
            print("LFR datasets not found, skipping...")
            return
        
        for lfr_folder in os.listdir(lfr_dir):
            if lfr_folder.startswith('lfr_'):
                print(f"Processing {lfr_folder}...")
                
                edges_file = f"{lfr_dir}/{lfr_folder}/edges.txt"
                communities_file = f"{lfr_dir}/{lfr_folder}/communities.txt"
                
                if os.path.exists(edges_file):
                    # Load graph
                    G = nx.read_edgelist(edges_file, nodetype=int)
                    data = from_networkx(G)
                    
                    # Tạo synthetic features
                    data.x = torch.randn(data.num_nodes, 64)
                    
                    # Load ground truth communities nếu có
                    ground_truth = None
                    if os.path.exists(communities_file):
                        ground_truth = self._load_communities(communities_file)
                    
                    # Tạo episodes
                    episodes = self.create_episodes_from_graph(data, num_episodes=40)
                    
                    processed_data = {
                        'original_graph': data,
                        'episodes': episodes,
                        'ground_truth': ground_truth,
                        'num_nodes': data.num_nodes,
                        'num_edges': data.num_edges,
                        'num_features': 64
                    }
                    
                    with open(f"{self.processed_data_dir}/{lfr_folder}.pkl", 'wb') as f:
                        pickle.dump(processed_data, f)
    
    def create_episodes_from_graph(self, data: Data, num_episodes: int = 50, 
                                 subgraph_size_range: Tuple[int, int] = (100, 500)) -> List[Data]:
        """Tạo episodes (subgraphs) từ graph lớn cho meta-learning"""
        episodes = []
        
        # Convert to NetworkX for easier subgraph extraction
        G = to_networkx(data, to_undirected=True)
        nodes = list(G.nodes())
        
        for i in range(num_episodes):
            # Random subgraph size
            subgraph_size = random.randint(*subgraph_size_range)
            subgraph_size = min(subgraph_size, len(nodes))
            
            # Random sampling nodes
            sampled_nodes = random.sample(nodes, subgraph_size)
            
            # Extract subgraph
            subG = G.subgraph(sampled_nodes).copy()
            
            # Ensure connected
            if not nx.is_connected(subG):
                # Take largest connected component
                largest_cc = max(nx.connected_components(subG), key=len)
                subG = subG.subgraph(largest_cc).copy()
            
            # Convert back to PyG
            episode_data = from_networkx(subG)
            
            # Map features từ original graph
            if data.x is not None:
                node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(subG.nodes())}
                episode_data.x = data.x[list(subG.nodes())]
            
            # Generate initial pseudo-labels bằng clustering
            episode_data.pseudo_labels = self._generate_initial_pseudo_labels(episode_data)
            
            episodes.append(episode_data)
        
        return episodes
    
    def _generate_initial_pseudo_labels(self, data: Data, n_clusters: Optional[int] = None) -> torch.Tensor:
        """Generate pseudo-labels ban đầu bằng spectral clustering"""
        
        # Estimate number of clusters
        if n_clusters is None:
            n_clusters = max(2, min(20, data.num_nodes // 50))  # Heuristic
        
        # Spectral clustering trên adjacency matrix
        edge_index = data.edge_index
        adj_matrix = torch.sparse_coo_tensor(
            edge_index, 
            torch.ones(edge_index.size(1)), 
            (data.num_nodes, data.num_nodes)
        ).to_dense().numpy()
        
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                      assign_labels='discretize', random_state=42)
        
        try:
            labels = clustering.fit_predict(adj_matrix)
        except:
            # Fallback to random labels
            labels = np.random.randint(0, n_clusters, data.num_nodes)
        
        return torch.LongTensor(labels)
    
    def _load_communities(self, communities_file: str) -> List[List[int]]:
        """Load ground truth communities từ file"""
        communities = []
        with open(communities_file, 'r') as f:
            for line in f:
                comm = [int(x) for x in line.strip().split()]
                if len(comm) > 0:
                    communities.append(comm)
        return communities


class EpisodeDataLoader:
    """DataLoader cho meta-learning episodes"""
    
    def __init__(self, processed_data_dir: str):
        self.processed_data_dir = processed_data_dir
        self.datasets = {}
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """Load tất cả processed datasets"""
        for file_name in os.listdir(self.processed_data_dir):
            if file_name.endswith('.pkl'):
                dataset_name = file_name[:-4]  # Remove .pkl
                with open(f"{self.processed_data_dir}/{file_name}", 'rb') as f:
                    self.datasets[dataset_name] = pickle.load(f)
    
    def get_meta_train_episodes(self, batch_size: int = 8) -> DataLoader:
        """Tạo DataLoader cho meta-training"""
        all_episodes = []
        
        # Combine episodes từ tất cả datasets
        for dataset_name, dataset_data in self.datasets.items():
            episodes = dataset_data['episodes']
            all_episodes.extend(episodes)
        
        # Shuffle episodes
        random.shuffle(all_episodes)
        
        return DataLoader(all_episodes, batch_size=batch_size, shuffle=True)
    
    def get_dataset_episodes(self, dataset_name: str) -> List[Data]:
        """Lấy episodes của một dataset cụ thể"""
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]['episodes']
        else:
            raise ValueError(f"Dataset {dataset_name} not found")
    
    def get_available_datasets(self) -> List[str]:
        """Lấy danh sách datasets có sẵn"""
        return list(self.datasets.keys())


if __name__ == "__main__":
    preprocessor = GraphPreprocessor(
        raw_data_dir="./data/raw",
        processed_data_dir="./data/processed"
    )
    preprocessor.preprocess_all()
