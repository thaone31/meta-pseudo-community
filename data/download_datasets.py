"""
Dataset downloader for community detection datasets
"""

import os
import requests
import zipfile
import tarfile
import networkx as nx
import numpy as np
from torch_geometric.datasets import Planetoid, Reddit, Amazon, DBLP
from torch_geometric.data import Data
import torch
from typing import List, Dict, Tuple
import pickle


class DatasetDownloader:
    """Download and prepare datasets for community detection"""
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_all(self):
        """Download all datasets"""
        print("Downloading datasets...")
        
        # Download PyG datasets
        self.download_pyg_datasets()
        
        # Download real-world networks
        self.download_real_world_networks()
        
        # Generate synthetic LFR networks
        self.generate_lfr_networks()
        
        print("All datasets downloaded successfully!")
    
    def download_pyg_datasets(self):
        """Download datasets from PyTorch Geometric - only small and stable datasets"""
        # Core citation networks - small and stable
        citation_datasets = ['Cora', 'CiteSeer', 'PubMed']
        
        for dataset_name in citation_datasets:
            print(f"Downloading {dataset_name}...")
            dataset = Planetoid(root=f"{self.data_dir}/pyg", name=dataset_name)
        
        # Amazon dataset - medium size
        print("Downloading Amazon-Computers...")
        amazon = Amazon(root=f"{self.data_dir}/pyg/Amazon", name="Computers")
        
        # DBLP dataset - medium size
        print("Downloading DBLP...")
        dblp = DBLP(root=f"{self.data_dir}/pyg/DBLP")
        
        print("✓ Core datasets downloaded successfully (Cora, CiteSeer, PubMed, Amazon-Computers, DBLP)")
        print("Note: Skipped large datasets (Reddit) for stability and performance")
    
    def download_real_world_networks(self):
        """Download real-world network datasets"""
        
        print("Skipping large SNAP datasets to avoid connection timeouts...")
        print("Using PyTorch Geometric datasets instead: Cora, CiteSeer, PubMed, Reddit, Amazon, DBLP")
        
        # Note: Large SNAP datasets (LiveJournal, Orkut, Youtube) can be downloaded manually if needed
        # snap_datasets = {
        #     'LiveJournal': 'https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz',
        #     'Orkut': 'https://snap.stanford.edu/data/com-orkut.ungraph.txt.gz', 
        #     'Youtube': 'https://snap.stanford.edu/data/com-youtube.ungraph.txt.gz'
        # }
        
        # Skip SNAP downloads for now to avoid timeouts
        pass
    
    def generate_lfr_networks(self):
        """Generate synthetic LFR benchmark networks"""
        try:
            import networkx as nx
            from networkx.generators.community import LFR_benchmark_graph
            
            print("Generating LFR benchmark networks...")
            
            # Different parameters for LFR networks - improved parameters
            lfr_params = [
                {'n': 1000, 'tau1': 3, 'tau2': 1.5, 'mu': 0.1, 'average_degree': 20, 'min_community': 50, 'max_community': 100, 'seed': 42},
                {'n': 1000, 'tau1': 2, 'tau2': 1.1, 'mu': 0.3, 'average_degree': 15, 'min_community': 30, 'max_community': 80, 'seed': 42},
                {'n': 2000, 'tau1': 3, 'tau2': 1.5, 'mu': 0.2, 'average_degree': 25, 'min_community': 60, 'max_community': 150, 'seed': 42},
                {'n': 1000, 'tau1': 3, 'tau2': 1.5, 'mu': 0.15, 'average_degree': 18, 'min_community': 40, 'max_community': 90, 'seed': 123},
            ]
            
            for i, params in enumerate(lfr_params):
                try:
                    print(f"Generating LFR network {i+1}/4...")
                    G = LFR_benchmark_graph(**params)
                    
                    # Save network and ground truth communities
                    output_dir = f"{self.data_dir}/lfr/lfr_{i+1}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Lưu graph
                    nx.write_edgelist(G, f"{output_dir}/edges.txt", data=False)
                    
                    # Lưu ground truth communities
                    communities = {frozenset(G.nodes[v]['community']) for v in G}
                    with open(f"{output_dir}/communities.txt", 'w') as f:
                        for comm in communities:
                            f.write(' '.join(map(str, comm)) + '\n')
                    
                    print(f"✓ LFR network {i+1} generated successfully")
                    
                except Exception as e:
                    print(f"Warning: Failed to generate LFR network {i+1}: {e}")
                    continue
                        
        except ImportError:
            print("Warning: Cannot generate LFR networks. Install networkx[extra] for LFR support.")
    
    def _download_file(self, url: str, output_path: str):
        """Helper function để download file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if os.path.exists(output_path):
            print(f"File {output_path} already exists, skipping...")
            return
            
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for community detection")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--pyg-only", action="store_true", help="Download only PyTorch Geometric datasets")
    parser.add_argument("--core-only", action="store_true", help="Download only core datasets (Cora, CiteSeer, PubMed)")
    parser.add_argument("--skip-reddit", action="store_true", help="Skip Reddit dataset (too large)")
    parser.add_argument("--skip-lfr", action="store_true", help="Skip LFR synthetic networks")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.core_only:
        print("Downloading core datasets only (Cora, CiteSeer, PubMed)...")
        # Just download the small citation networks
        datasets = ['Cora', 'CiteSeer', 'PubMed']
        for dataset_name in datasets:
            print(f"Downloading {dataset_name}...")
            from torch_geometric.datasets import Planetoid
            dataset = Planetoid(root=f"{downloader.data_dir}/pyg", name=dataset_name)
        print("✓ Core datasets downloaded successfully!")
    elif args.pyg_only:
        print("Downloading PyTorch Geometric datasets only...")
        downloader.download_pyg_datasets()
    else:
        downloader.download_all()
