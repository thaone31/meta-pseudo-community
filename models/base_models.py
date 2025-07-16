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
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))  # Use LayerNorm instead of BatchNorm
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
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
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer (single head)
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, dropout=dropout))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
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
        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class CommunityDetectionModel(nn.Module):
    """Base model cho community detection với adaptive input dimension"""
    
    def __init__(self, encoder_type: str, input_dim: int, hidden_dim: int, 
                 embedding_dim: int, num_classes: int, adaptive_input: bool = True, **kwargs):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.adaptive_input = adaptive_input
        self.base_input_dim = input_dim
        
        # Feature projection layer for different input dimensions
        if adaptive_input:
            self.feature_projectors = nn.ModuleDict()
            # Start with a default projector
            self.feature_projectors['default'] = nn.Linear(input_dim, input_dim)
            self.adaptive_target_dim = input_dim
        
        # Choose encoder with appropriate parameters
        if encoder_type.lower() == 'gcn':
            # Filter out GAT-specific parameters
            gcn_kwargs = {k: v for k, v in kwargs.items() if k not in ['heads']}
            self.encoder = GCNEncoder(input_dim, hidden_dim, embedding_dim, **gcn_kwargs)
        elif encoder_type.lower() == 'gat':
            self.encoder = GATEncoder(input_dim, hidden_dim, embedding_dim, **kwargs)
        elif encoder_type.lower() == 'gin':
            # Filter out GAT-specific parameters  
            gin_kwargs = {k: v for k, v in kwargs.items() if k not in ['heads']}
            self.encoder = GINEncoder(input_dim, hidden_dim, embedding_dim, **gin_kwargs)
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
    
    def _get_or_create_projector(self, input_dim: int) -> nn.Module:
        """Get or create feature projector for given input dimension"""
        key = str(input_dim)
        
        if key not in self.feature_projectors:
            # Create new projector
            if input_dim == self.adaptive_target_dim:
                # Identity mapping
                projector = nn.Identity()
            else:
                # Linear projection
                projector = nn.Linear(input_dim, self.adaptive_target_dim)
                # Initialize with reasonable weights
                nn.init.xavier_uniform_(projector.weight)
                nn.init.zeros_(projector.bias)
            
            self.feature_projectors[key] = projector
            
        return self.feature_projectors[key]
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                return_embeddings: bool = False) -> torch.Tensor:
        """Forward pass với adaptive input handling"""
        
        # Apply feature projection if needed
        if self.adaptive_input and x.size(1) != self.adaptive_target_dim:
            projector = self._get_or_create_projector(x.size(1))
            x = projector(x)
        
        # Get node embeddings
        embeddings = self.encoder(x, edge_index)
        
        if return_embeddings:
            return embeddings
        
        # Classification
        logits = self.classifier(embeddings)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings với adaptive input handling"""
        
        # Apply feature projection if needed
        if self.adaptive_input and x.size(1) != self.adaptive_target_dim:
            projector = self._get_or_create_projector(x.size(1))
            x = projector(x)
            
        return self.encoder(x, edge_index)
    
    def get_projected_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get projected embeddings for contrastive learning"""
        embeddings = self.get_embeddings(x, edge_index)
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
    # Test models với adaptive input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing adaptive input model...")
    
    # Test với different input dimensions
    input_dims = [64, 128, 1433, 3703]  # Common sizes from different datasets
    num_nodes = 100
    
    # Test GCN model với adaptive input
    model = CommunityDetectionModel(
        encoder_type='gcn',
        input_dim=64,  # Base dimension
        hidden_dim=128,
        embedding_dim=64,
        num_classes=5,
        adaptive_input=True
    )
    
    for input_dim in input_dims:
        print(f"\nTesting input dimension: {input_dim}")
        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, 200))
        
        try:
            output = model(x, edge_index)
            embeddings = model.get_embeddings(x, edge_index)
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Embeddings shape: {embeddings.shape}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Feature projectors: {len(model.feature_projectors)}")
