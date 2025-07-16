"""
Meta-learning algorithms cho pseudo-label optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from collections import OrderedDict
import copy

from .base_models import CommunityDetectionModel
from .pseudo_labels import PseudoLabelGenerator, AdaptivePseudoLabelGenerator


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning cho pseudo-label optimization"""
    
    def __init__(self, model: nn.Module, lr_inner: float = 0.01, lr_outer: float = 0.001,
                 num_inner_steps: int = 5, first_order: bool = False):
        super().__init__()
        
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_outer)
        
    def forward(self, support_data: Data, query_data: Data, 
                pseudo_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """MAML forward pass"""
        
        # Inner loop: adapt model trên support set với pseudo-labels
        adapted_params = self._inner_loop(support_data, pseudo_labels)
        
        # Outer loop: evaluate trên query set
        query_loss, query_acc = self._evaluate_on_query(query_data, adapted_params)
        
        return {
            'query_loss': query_loss,
            'query_accuracy': query_acc,
            'adapted_params': adapted_params
        }
    
    def _inner_loop(self, support_data: Data, pseudo_labels: torch.Tensor) -> OrderedDict:
        """Inner loop adaptation"""
        
        # Create a copy of model parameters
        params = OrderedDict(self.model.named_parameters())
        
        for step in range(self.num_inner_steps):
            # Forward pass với current parameters
            logits = self._forward_with_params(support_data, params)
            
            # Compute loss với pseudo-labels
            loss = F.cross_entropy(logits, pseudo_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, params.values(), 
                create_graph=not self.first_order,
                retain_graph=True
            )
            
            # Update parameters
            updated_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - self.lr_inner * grad
            
            params = updated_params
        
        return params
    
    def _forward_with_params(self, data: Data, params: OrderedDict) -> torch.Tensor:
        """Forward pass với specific parameters"""
        
        # Temporarily replace model parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
            param.data = params[name]
        
        # Forward pass
        logits = self.model(data.x, data.edge_index)
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        return logits
    
    def _evaluate_on_query(self, query_data: Data, adapted_params: OrderedDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate adapted model trên query set"""
        
        # Forward với adapted parameters
        logits = self._forward_with_params(query_data, adapted_params)
        
        # Assume query data có ground truth labels
        if hasattr(query_data, 'y') and query_data.y is not None:
            loss = F.cross_entropy(logits, query_data.y)
            
            # Accuracy
            pred = torch.argmax(logits, dim=1)
            acc = (pred == query_data.y).float().mean()
        else:
            # Self-supervised loss (e.g., clustering quality)
            loss = self._compute_clustering_loss(logits, query_data)
            acc = torch.tensor(0.0)  # Placeholder
        
        return loss, acc
    
    def _compute_clustering_loss(self, logits: torch.Tensor, data: Data) -> torch.Tensor:
        """Compute clustering quality loss"""
        
        # Soft assignments
        soft_assignments = F.softmax(logits, dim=1)
        
        # Entropy regularization (encourage confident predictions)
        entropy = -torch.sum(soft_assignments * torch.log(soft_assignments + 1e-8), dim=1)
        entropy_loss = entropy.mean()
        
        # Modularity-based loss
        modularity_loss = self._compute_modularity_loss(soft_assignments, data.edge_index, data.num_nodes)
        
        # Combine losses
        total_loss = entropy_loss - 0.1 * modularity_loss  # Minimize entropy, maximize modularity
        
        return total_loss
    
    def _compute_modularity_loss(self, soft_assignments: torch.Tensor, 
                               edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute differentiable modularity loss"""
        
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
        soft_modularity = torch.trace(soft_assignments.t() @ modularity_matrix @ soft_assignments)
        soft_modularity = soft_modularity / (2 * m)
        
        return soft_modularity
    
    def meta_update(self, episodes: List[Tuple[Data, Data, torch.Tensor]]):
        """Meta-update step"""
        
        self.meta_optimizer.zero_grad()
        
        total_loss = 0
        batch_size = len(episodes)
        
        for support_data, query_data, pseudo_labels in episodes:
            result = self.forward(support_data, query_data, pseudo_labels)
            total_loss += result['query_loss']
        
        # Average loss
        meta_loss = total_loss / batch_size
        
        # Backward và update
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


class Reptile(nn.Module):
    """Reptile meta-learning algorithm"""
    
    def __init__(self, model: nn.Module, lr_inner: float = 0.01, lr_outer: float = 0.001,
                 num_inner_steps: int = 10):
        super().__init__()
        
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.num_inner_steps = num_inner_steps
        
        # Inner optimizer (sẽ được reset cho mỗi task)
        self.inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr_inner)
    
    def adapt_to_task(self, task_data: Data, pseudo_labels: torch.Tensor) -> OrderedDict:
        """Adapt model to specific task"""
        
        # Save original parameters
        original_params = OrderedDict()
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        # Inner loop adaptation
        for step in range(self.num_inner_steps):
            self.inner_optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(task_data.x, task_data.edge_index)
            
            # Loss với pseudo-labels
            loss = F.cross_entropy(logits, pseudo_labels)
            
            # Backward và update
            loss.backward()
            self.inner_optimizer.step()
        
        # Get adapted parameters
        adapted_params = OrderedDict()
        for name, param in self.model.named_parameters():
            adapted_params[name] = param.data.clone()
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        return adapted_params
    
    def meta_update(self, episodes: List[Tuple[Data, torch.Tensor]]):
        """Reptile meta-update"""
        
        all_adapted_params = []
        
        # Adapt to each task
        for task_data, pseudo_labels in episodes:
            adapted_params = self.adapt_to_task(task_data, pseudo_labels)
            all_adapted_params.append(adapted_params)
        
        # Meta-update: move towards average of adapted parameters
        for name, param in self.model.named_parameters():
            # Compute average adapted parameter
            avg_adapted = torch.stack([adapted[name] for adapted in all_adapted_params]).mean(dim=0)
            
            # Update toward average
            param.data = param.data + self.lr_outer * (avg_adapted - param.data)


class MetaPseudoLabelOptimizer:
    """Meta-optimizer cho pseudo-label generation strategy"""
    
    def __init__(self, base_model: nn.Module, meta_algorithm: str = 'maml',
                 pseudo_label_methods: List[str] = ['spectral', 'louvain', 'kmeans']):
        
        self.base_model = base_model
        
        # Meta-learning algorithm
        if meta_algorithm.lower() == 'maml':
            self.meta_learner = MAML(base_model)
        elif meta_algorithm.lower() == 'reptile':
            self.meta_learner = Reptile(base_model)
        else:
            raise ValueError(f"Unknown meta algorithm: {meta_algorithm}")
        
        # Pseudo-label generator
        self.pseudo_label_generator = AdaptivePseudoLabelGenerator(pseudo_label_methods)
        
        # Meta-parameters cho pseudo-label generation
        self.meta_params = nn.ParameterDict({
            'confidence_threshold': nn.Parameter(torch.tensor(0.8)),
            'method_weights': nn.Parameter(torch.ones(len(pseudo_label_methods))),
            'clustering_temperature': nn.Parameter(torch.tensor(1.0))
        })
        
        # Meta-optimizer cho meta-parameters
        self.meta_param_optimizer = torch.optim.Adam(self.meta_params.parameters(), lr=0.001)
    
    def meta_train_step(self, episodes: List[Data]) -> Dict[str, float]:
        """Meta-training step"""
        
        # Prepare episodes với pseudo-labels
        prepared_episodes = []
        
        for episode_data in episodes:
            # Split episode into support và query first
            support_data, query_data = self._split_episode(episode_data)
            
            # Generate pseudo-labels for support set only (for training)
            support_pseudo_labels = self._generate_meta_pseudo_labels(support_data)
            
            # Generate pseudo-labels for query set (for evaluation)
            query_pseudo_labels = self._generate_meta_pseudo_labels(query_data)
            
            prepared_episodes.append((support_data, query_data, support_pseudo_labels, query_pseudo_labels))
        
        # Meta-update base model
        if isinstance(self.meta_learner, MAML):
            # Prepare episodes in the format MAML expects: (support, query, support_labels)
            maml_episodes = [(support, query, support_labels) for support, query, support_labels, _ in prepared_episodes]
            meta_loss = self.meta_learner.meta_update(maml_episodes)
        else:  # Reptile
            # Prepare episodes in the format Reptile expects: (support, support_labels)
            task_episodes = [(support, support_labels) for support, query, support_labels, _ in prepared_episodes]
            self.meta_learner.meta_update(task_episodes)
            meta_loss = 0.0  # Reptile doesn't return loss
        
        # Meta-update pseudo-label parameters
        self._update_meta_parameters(prepared_episodes)
        
        return {
            'meta_loss': meta_loss,
            'confidence_threshold': self.meta_params['confidence_threshold'].item(),
            'method_weights': self.meta_params['method_weights'].detach().cpu().numpy()
        }
    
    def _generate_meta_pseudo_labels(self, data: Data) -> torch.Tensor:
        """Generate pseudo-labels với meta-learned parameters"""
        
        # Update pseudo-label generator weights
        method_weights = F.softmax(self.meta_params['method_weights'], dim=0)
        weight_dict = dict(zip(self.pseudo_label_generator.generators.keys(), method_weights.tolist()))
        self.pseudo_label_generator.update_method_weights(weight_dict)
        
        # Generate ensemble pseudo-labels
        result = self.pseudo_label_generator.generate_ensemble(data)
        
        # Apply meta-learned confidence threshold
        confidence_threshold = torch.sigmoid(self.meta_params['confidence_threshold'])
        high_conf_mask = result['confidence_scores'] >= confidence_threshold
        
        # For nodes với low confidence, use clustering on embeddings
        if hasattr(data, 'x') and data.x is not None:
            embeddings = self.base_model.get_embeddings(data.x, data.edge_index)
            
            # Temperature-scaled clustering
            temperature = F.softplus(self.meta_params['clustering_temperature'])
            # Simplified clustering logic here...
        
        return result['labels']
    
    def _split_episode(self, data: Data, support_ratio: float = 0.7) -> Tuple[Data, Data]:
        """Split episode into support và query sets"""
        
        num_nodes = data.num_nodes
        num_support = int(num_nodes * support_ratio)
        
        # Random split
        perm = torch.randperm(num_nodes)
        support_nodes = perm[:num_support]
        query_nodes = perm[num_support:]
        
        # Create subgraphs
        support_data = self._create_subgraph(data, support_nodes)
        query_data = self._create_subgraph(data, query_nodes)
        
        return support_data, query_data
    
    def _create_subgraph(self, data: Data, nodes: torch.Tensor) -> Data:
        """Create subgraph từ selected nodes"""
        
        # Node mapping
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(nodes)}
        
        # Filter edges
        edge_mask = torch.isin(data.edge_index[0], nodes) & torch.isin(data.edge_index[1], nodes)
        filtered_edges = data.edge_index[:, edge_mask]
        
        # Remap edge indices
        new_edges = torch.zeros_like(filtered_edges)
        for i in range(filtered_edges.size(1)):
            new_edges[0, i] = node_map[filtered_edges[0, i].item()]
            new_edges[1, i] = node_map[filtered_edges[1, i].item()]
        
        # Create subgraph data
        subgraph_data = Data(
            x=data.x[nodes] if data.x is not None else None,
            edge_index=new_edges,
            num_nodes=len(nodes)
        )
        
        # Copy other attributes nếu có
        for key, value in data:
            if key not in ['x', 'edge_index', 'num_nodes']:
                if torch.is_tensor(value) and value.size(0) == data.num_nodes:
                    setattr(subgraph_data, key, value[nodes])
        
        return subgraph_data
    
    def _update_meta_parameters(self, episodes: List[Tuple[Data, Data, torch.Tensor, torch.Tensor]]):
        """Update meta-parameters based on episode performance"""
        
        self.meta_param_optimizer.zero_grad()
        
        total_loss = 0
        
        for support_data, query_data, support_labels, query_labels in episodes:
            # Evaluate pseudo-label quality on query set
            quality_loss = self._compute_pseudo_label_quality_loss(query_data, query_labels)
            total_loss += quality_loss
        
        if total_loss > 0:
            meta_param_loss = total_loss / len(episodes)
            meta_param_loss.backward()
            self.meta_param_optimizer.step()
    
    def _compute_pseudo_label_quality_loss(self, data: Data, pseudo_labels: torch.Tensor) -> torch.Tensor:
        """Compute quality loss cho pseudo-labels"""
        
        # Modularity-based quality
        modularity = self._compute_modularity(data, pseudo_labels)
        
        # Silhouette-based quality nếu có embeddings
        silhouette_score = 0
        if data.x is not None:
            try:
                from sklearn.metrics import silhouette_score as sklearn_silhouette
                embeddings = self.base_model.get_embeddings(data.x, data.edge_index)
                embeddings_np = embeddings.detach().cpu().numpy()
                labels_np = pseudo_labels.cpu().numpy()
                
                if len(np.unique(labels_np)) > 1:
                    silhouette_score = sklearn_silhouette(embeddings_np, labels_np)
            except:
                silhouette_score = 0
        
        # Combine metrics (negative because we want to minimize loss)
        quality_loss = -(modularity + 0.5 * silhouette_score)
        
        return torch.tensor(quality_loss, requires_grad=True)
    
    def _compute_modularity(self, data: Data, labels: torch.Tensor) -> float:
        """Compute modularity score"""
        
        try:
            from torch_geometric.utils import to_networkx
            import networkx as nx
            
            # Convert to NetworkX
            G = to_networkx(data, to_undirected=True)
            
            # Create community structure
            communities = []
            unique_labels = labels.unique()
            
            for label in unique_labels:
                community = (labels == label).nonzero(as_tuple=True)[0].tolist()
                if len(community) > 0:
                    communities.append(community)
            
            # Compute modularity
            if len(communities) > 1:
                modularity = nx.community.modularity(G, communities)
            else:
                modularity = 0.0
                
        except:
            modularity = 0.0
        
        return modularity


if __name__ == "__main__":
    # Test meta-learning components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test model
    model = CommunityDetectionModel(
        encoder_type='gcn',
        input_dim=64,
        hidden_dim=128,
        embedding_dim=64,
        num_classes=5
    )
    
    # Test MAML
    maml = MAML(model)
    
    # Create test episode
    num_nodes = 50
    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    support_data = Data(x=x, edge_index=edge_index)
    query_data = Data(x=x, edge_index=edge_index)
    pseudo_labels = torch.randint(0, 5, (num_nodes,))
    
    # Test forward pass
    result = maml.forward(support_data, query_data, pseudo_labels)
    print(f"MAML query loss: {result['query_loss'].item()}")
    
    # Test MetaPseudoLabelOptimizer
    optimizer = MetaPseudoLabelOptimizer(model)
    episodes = [support_data]
    
    step_result = optimizer.meta_train_step(episodes)
    print(f"Meta-training step result: {step_result}")
