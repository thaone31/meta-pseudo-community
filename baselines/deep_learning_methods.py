"""
Deep learning methods cho community detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx, negative_sampling
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import networkx as nx


class DeepWalk:
    """DeepWalk implementation using random walks"""
    
    def __init__(self, embedding_dim: int = 128, walk_length: int = 80, 
                 num_walks: int = 10, window_size: int = 5, min_count: int = 0, 
                 sg: int = 1, workers: int = 4):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.name = "DeepWalk"
    
    def detect_communities(self, data: Data, n_clusters: Optional[int] = None) -> Dict:
        """Detect communities using DeepWalk embeddings"""
        
        try:
            from node2vec import Node2Vec
            
            start_time = time.time()
            
            # Convert to NetworkX
            G = to_networkx(data, to_undirected=True)
            
            # Generate walks và embeddings
            node2vec = Node2Vec(G, dimensions=self.embedding_dim, walk_length=self.walk_length,
                              num_walks=self.num_walks, workers=self.workers, p=1, q=1)
            
            model = node2vec.fit(window=self.window_size, min_count=self.min_count, 
                               sg=self.sg, workers=self.workers)
            
            # Get embeddings
            embeddings = np.array([model.wv[str(node)] for node in sorted(G.nodes())])
            
            # Cluster embeddings
            if n_clusters is None:
                n_clusters = max(2, min(20, data.num_nodes // 50))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Compute modularity
            communities = []
            for comm_id in np.unique(labels):
                community = np.where(labels == comm_id)[0].tolist()
                communities.append(community)
            
            modularity = nx.community.modularity(G, communities)
            
            end_time = time.time()
            
            return {
                'labels': torch.LongTensor(labels),
                'embeddings': torch.FloatTensor(embeddings),
                'modularity': modularity,
                'num_communities': len(set(labels)),
                'runtime': end_time - start_time
            }
            
        except ImportError:
            raise ImportError("Please install node2vec: pip install node2vec")


class Node2VecMethod:
    """Node2Vec implementation"""
    
    def __init__(self, embedding_dim: int = 128, walk_length: int = 80,
                 num_walks: int = 10, p: float = 1.0, q: float = 1.0,
                 window_size: int = 5, min_count: int = 0, workers: int = 4):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.min_count = min_count
        self.workers = workers
        self.name = "Node2Vec"
    
    def detect_communities(self, data: Data, n_clusters: Optional[int] = None) -> Dict:
        """Detect communities using Node2Vec embeddings"""
        
        try:
            from node2vec import Node2Vec
            
            start_time = time.time()
            
            # Convert to NetworkX
            G = to_networkx(data, to_undirected=True)
            
            # Generate walks và embeddings
            node2vec = Node2Vec(G, dimensions=self.embedding_dim, walk_length=self.walk_length,
                              num_walks=self.num_walks, p=self.p, q=self.q, workers=self.workers)
            
            model = node2vec.fit(window=self.window_size, min_count=self.min_count, 
                               sg=1, workers=self.workers)
            
            # Get embeddings
            embeddings = np.array([model.wv[str(node)] for node in sorted(G.nodes())])
            
            # Cluster embeddings
            if n_clusters is None:
                n_clusters = max(2, min(20, data.num_nodes // 50))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Compute modularity
            communities = []
            for comm_id in np.unique(labels):
                community = np.where(labels == comm_id)[0].tolist()
                communities.append(community)
            
            modularity = nx.community.modularity(G, communities)
            
            end_time = time.time()
            
            return {
                'labels': torch.LongTensor(labels),
                'embeddings': torch.FloatTensor(embeddings),
                'modularity': modularity,
                'num_communities': len(set(labels)),
                'runtime': end_time - start_time
            }
            
        except ImportError:
            raise ImportError("Please install node2vec: pip install node2vec")


class GraphSAGE(nn.Module):
    """GraphSAGE implementation cho community detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class DGI(nn.Module):
    """Deep Graph Infomax implementation"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            nn.PReLU(),
            GCNConv(hidden_dim, hidden_dim)
        )
        
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.corruption = self._corruption
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        
        # Positive samples
        positive = self.encoder[0](x, edge_index)
        positive = self.encoder[1](positive)
        positive = self.encoder[2](positive, edge_index)
        
        # Negative samples (corrupted)
        corrupted_x = self.corruption(x)
        negative = self.encoder[0](corrupted_x, edge_index)
        negative = self.encoder[1](negative)
        negative = self.encoder[2](negative, edge_index)
        
        # Global representation
        summary = torch.sigmoid(positive.mean(dim=0))
        
        return positive, negative, summary
    
    def _corruption(self, x: torch.Tensor) -> torch.Tensor:
        """Corrupt node features"""
        return x[torch.randperm(x.size(0))]
    
    def loss(self, positive: torch.Tensor, negative: torch.Tensor, 
             summary: torch.Tensor) -> torch.Tensor:
        """Compute DGI loss"""
        
        # Positive scores
        pos_scores = self.discriminator(positive, summary.expand_as(positive)).squeeze()
        
        # Negative scores
        neg_scores = self.discriminator(negative, summary.expand_as(negative)).squeeze()
        
        # Binary cross entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        
        return pos_loss + neg_loss


class VGAE(nn.Module):
    """Variational Graph Auto-Encoder"""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        # Encoder
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)
        
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode node features"""
        
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        
        return mu, logstd
    
    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick"""
        
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Decode to reconstruct adjacency"""
        
        # Inner product decoder
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        
        mu, logstd = self.encode(x, edge_index)
        z = self.reparametrize(mu, logstd)
        adj_recon = self.decode(z, edge_index)
        
        return adj_recon, mu, logstd
    
    def loss(self, adj_recon: torch.Tensor, adj_true: torch.Tensor,
             mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """Compute VGAE loss"""
        
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(adj_recon.view(-1), adj_true.view(-1))
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))
        
        return recon_loss + kl_loss


class DMoN(nn.Module):
    """Deep Modularity Networks"""
    
    def __init__(self, input_dim: int, hidden_dim: int, n_clusters: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.n_clusters = n_clusters
        
        # GCN encoder
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, n_clusters))
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer với softmax
        x = self.convs[-1](x, edge_index)
        assignments = F.softmax(x, dim=-1)
        
        return assignments
    
    def modularity_loss(self, assignments: torch.Tensor, edge_index: torch.Tensor, 
                       num_nodes: int) -> torch.Tensor:
        """Compute modularity loss"""
        
        # Build adjacency matrix
        adj = torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.size(1)), 
            (num_nodes, num_nodes)
        ).to_dense()
        
        # Degree matrix
        degrees = adj.sum(dim=1)
        m = edge_index.size(1) / 2  # Number of edges
        
        # Expected edges under null model
        expected = torch.outer(degrees, degrees) / (2 * m)
        
        # Modularity matrix
        modularity_matrix = adj - expected
        
        # Soft modularity
        soft_modularity = torch.trace(assignments.t() @ modularity_matrix @ assignments)
        soft_modularity = soft_modularity / (2 * m)
        
        return -soft_modularity  # Negative because we want to maximize
    
    def collapse_regularization(self, assignments: torch.Tensor) -> torch.Tensor:
        """Regularization to prevent collapsed clusters"""
        
        # Cluster size regularization
        cluster_sizes = assignments.sum(dim=0)
        size_loss = torch.var(cluster_sizes)
        
        # Orthogonality regularization
        normalized_assignments = F.normalize(assignments, p=2, dim=0)
        correlation_matrix = torch.mm(normalized_assignments.t(), normalized_assignments)
        identity = torch.eye(self.n_clusters, device=assignments.device)
        orthogonal_loss = torch.norm(correlation_matrix - identity, p='fro')
        
        return size_loss + orthogonal_loss


class DeepLearningMethodsEvaluator:
    """Evaluator cho deep learning community detection methods"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def train_and_evaluate_dgi(self, data: Data, epochs: int = 200, 
                              lr: float = 0.001, hidden_dim: int = 512) -> Dict:
        """Train và evaluate DGI"""
        
        start_time = time.time()
        
        # Initialize model
        model = DGI(data.x.size(1), hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        data = data.to(self.device)
        
        # Training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            positive, negative, summary = model(data.x, data.edge_index)
            loss = model.loss(positive, negative, summary)
            loss.backward()
            optimizer.step()
        
        # Get embeddings
        model.eval()
        with torch.no_grad():
            embeddings, _, _ = model(data.x, data.edge_index)
            embeddings = embeddings.cpu().numpy()
        
        # Cluster embeddings
        n_clusters = max(2, min(20, data.num_nodes // 50))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Compute modularity
        G = to_networkx(data.cpu(), to_undirected=True)
        communities = []
        for comm_id in np.unique(labels):
            community = np.where(labels == comm_id)[0].tolist()
            communities.append(community)
        
        modularity = nx.community.modularity(G, communities)
        
        end_time = time.time()
        
        return {
            'labels': torch.LongTensor(labels),
            'embeddings': torch.FloatTensor(embeddings),
            'modularity': modularity,
            'num_communities': len(set(labels)),
            'runtime': end_time - start_time
        }
    
    def train_and_evaluate_vgae(self, data: Data, epochs: int = 200,
                               lr: float = 0.01, hidden_dim: int = 32,
                               latent_dim: int = 16) -> Dict:
        """Train và evaluate VGAE"""
        
        start_time = time.time()
        
        # Initialize model
        model = VGAE(data.x.size(1), hidden_dim, latent_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        data = data.to(self.device)
        
        # Build adjacency matrix for training
        adj_true = torch.sparse_coo_tensor(
            data.edge_index, torch.ones(data.edge_index.size(1)),
            (data.num_nodes, data.num_nodes)
        ).to_dense()
        
        # Training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            adj_recon, mu, logstd = model(data.x, data.edge_index)
            loss = model.loss(adj_recon, adj_true, mu, logstd)
            loss.backward()
            optimizer.step()
        
        # Get embeddings
        model.eval()
        with torch.no_grad():
            _, mu, _ = model(data.x, data.edge_index)
            embeddings = mu.cpu().numpy()
        
        # Cluster embeddings
        n_clusters = max(2, min(20, data.num_nodes // 50))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Compute modularity
        G = to_networkx(data.cpu(), to_undirected=True)
        communities = []
        for comm_id in np.unique(labels):
            community = np.where(labels == comm_id)[0].tolist()
            communities.append(community)
        
        modularity = nx.community.modularity(G, communities)
        
        end_time = time.time()
        
        return {
            'labels': torch.LongTensor(labels),
            'embeddings': torch.FloatTensor(embeddings),
            'modularity': modularity,
            'num_communities': len(set(labels)),
            'runtime': end_time - start_time
        }
    
    def train_and_evaluate_dmon(self, data: Data, epochs: int = 500,
                               lr: float = 0.001, hidden_dim: int = 64,
                               n_clusters: Optional[int] = None) -> Dict:
        """Train và evaluate DMoN"""
        
        start_time = time.time()
        
        if n_clusters is None:
            n_clusters = max(2, min(20, data.num_nodes // 50))
        
        # Initialize model
        model = DMoN(data.x.size(1), hidden_dim, n_clusters).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        data = data.to(self.device)
        
        # Training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            assignments = model(data.x, data.edge_index)
            
            # Modularity loss
            mod_loss = model.modularity_loss(assignments, data.edge_index, data.num_nodes)
            
            # Regularization
            reg_loss = model.collapse_regularization(assignments)
            
            total_loss = mod_loss + 0.001 * reg_loss
            total_loss.backward()
            optimizer.step()
        
        # Get final assignments
        model.eval()
        with torch.no_grad():
            assignments = model(data.x, data.edge_index)
            labels = torch.argmax(assignments, dim=1).cpu().numpy()
        
        # Compute modularity
        G = to_networkx(data.cpu(), to_undirected=True)
        communities = []
        for comm_id in np.unique(labels):
            community = np.where(labels == comm_id)[0].tolist()
            communities.append(community)
        
        modularity = nx.community.modularity(G, communities)
        
        end_time = time.time()
        
        return {
            'labels': torch.LongTensor(labels),
            'assignments': assignments.cpu(),
            'modularity': modularity,
            'num_communities': len(set(labels)),
            'runtime': end_time - start_time
        }


if __name__ == "__main__":
    # Test deep learning methods
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    num_nodes = 100
    input_dim = 64
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    data = Data(x=x, edge_index=edge_index)
    
    # Test evaluator
    evaluator = DeepLearningMethodsEvaluator(device)
    
    # Test DGI
    print("Testing DGI...")
    dgi_result = evaluator.train_and_evaluate_dgi(data, epochs=50)
    print(f"DGI: {dgi_result['num_communities']} communities, "
          f"modularity: {dgi_result['modularity']:.3f}")
    
    # Test VGAE
    print("Testing VGAE...")
    vgae_result = evaluator.train_and_evaluate_vgae(data, epochs=50)
    print(f"VGAE: {vgae_result['num_communities']} communities, "
          f"modularity: {vgae_result['modularity']:.3f}")
    
    # Test DMoN
    print("Testing DMoN...")
    dmon_result = evaluator.train_and_evaluate_dmon(data, epochs=100)
    print(f"DMoN: {dmon_result['num_communities']} communities, "
          f"modularity: {dmon_result['modularity']:.3f}")
    
    # Test Node2Vec (requires installation)
    try:
        print("Testing Node2Vec...")
        node2vec = Node2VecMethod()
        node2vec_result = node2vec.detect_communities(data)
        print(f"Node2Vec: {node2vec_result['num_communities']} communities, "
              f"modularity: {node2vec_result['modularity']:.3f}")
    except ImportError:
        print("Node2Vec not available (install node2vec package)")
