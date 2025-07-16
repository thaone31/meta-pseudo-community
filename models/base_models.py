"""
Base models cho community detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data
from typing import Optional, Tuple
import numpy as np


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Layers
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
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer without activation
        x = self.convs[-1](x, edge_index)
        
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network encoder"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer (single head)
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, dropout=dropout))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GINEncoder(nn.Module):
    """Graph Isomorphism Network encoder"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            nn_hidden = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_hidden))
        
        # Output layer
        nn_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.convs.append(GINConv(nn_out))
        
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])
    
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


class CommunityDetectionModel(nn.Module):
    """Base model cho community detection"""
    
    def __init__(self, encoder_type: str, input_dim: int, hidden_dim: int, 
                 embedding_dim: int, num_classes: int, **kwargs):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Choose encoder
        if encoder_type.lower() == 'gcn':
            self.encoder = GCNEncoder(input_dim, hidden_dim, embedding_dim, **kwargs)
        elif encoder_type.lower() == 'gat':
            self.encoder = GATEncoder(input_dim, hidden_dim, embedding_dim, **kwargs)
        elif encoder_type.lower() == 'gin':
            self.encoder = GINEncoder(input_dim, hidden_dim, embedding_dim, **kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # For contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                return_embeddings: bool = False) -> torch.Tensor:
        """Forward pass"""
        
        # Get node embeddings
        embeddings = self.encoder(x, edge_index)
        
        if return_embeddings:
            return embeddings
        
        # Classification
        logits = self.classifier(embeddings)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings"""
        return self.encoder(x, edge_index)
    
    def get_projected_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get projected embeddings for contrastive learning"""
        embeddings = self.encoder(x, edge_index)
        return self.projection_head(embeddings)


class DeepClusteringModel(nn.Module):
    """Deep clustering model với learnable cluster centers"""
    
    def __init__(self, encoder_type: str, input_dim: int, hidden_dim: int,
                 embedding_dim: int, num_clusters: int, **kwargs):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        
        # Encoder
        if encoder_type.lower() == 'gcn':
            self.encoder = GCNEncoder(input_dim, hidden_dim, embedding_dim, **kwargs)
        elif encoder_type.lower() == 'gat':
            self.encoder = GATEncoder(input_dim, hidden_dim, embedding_dim, **kwargs)
        elif encoder_type.lower() == 'gin':
            self.encoder = GINEncoder(input_dim, hidden_dim, embedding_dim, **kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, embedding_dim))
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        
        # Get embeddings
        embeddings = self.encoder(x, edge_index)
        
        # Compute similarities to cluster centers
        similarities = torch.mm(embeddings, self.cluster_centers.t())
        
        # Apply temperature scaling
        logits = similarities / self.temperature
        
        # Soft assignments
        soft_assignments = F.softmax(logits, dim=1)
        
        return embeddings, soft_assignments
    
    def get_hard_assignments(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get hard cluster assignments"""
        _, soft_assignments = self.forward(x, edge_index)
        return torch.argmax(soft_assignments, dim=1)


def create_model(model_type: str, config: dict) -> nn.Module:
    """Factory function để tạo models"""
    
    if model_type == 'community_detection':
        return CommunityDetectionModel(**config)
    elif model_type == 'deep_clustering':
        return DeepClusteringModel(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    num_nodes = 100
    input_dim = 64
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    # Test GCN model
    model = CommunityDetectionModel(
        encoder_type='gcn',
        input_dim=input_dim,
        hidden_dim=128,
        embedding_dim=64,
        num_classes=5
    )
    
    output = model(x, edge_index)
    print(f"Output shape: {output.shape}")
    
    embeddings = model.get_embeddings(x, edge_index)
    print(f"Embeddings shape: {embeddings.shape}")
