"""
Training script cho Meta-Learning Pseudo-Labels model
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_models import CommunityDetectionModel
from models.meta_learning import MetaPseudoLabelOptimizer
from data.preprocess import EpisodeDataLoader
from utils.evaluation import compute_all_metrics
from utils.visualization import plot_training_curves


class MetaTrainer:
    """Trainer cho Meta-Learning Pseudo-Labels"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self._setup_logging()
        self._setup_random_seeds()
        
        # Initialize data loader
        self.data_loader = EpisodeDataLoader(processed_data_dir="./data/processed")
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize meta-optimizer
        self.meta_optimizer = MetaPseudoLabelOptimizer(
            base_model=self.model,
            meta_algorithm=config['model']['meta_learner']['algorithm'],
            pseudo_label_methods=config['model']['pseudo_label']['methods']
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metric = -float('inf')
        self.patience_counter = 0
        
        # Logging
        self.train_losses = []
        self.val_metrics = []
        
    def _setup_device(self) -> torch.device:
        """Setup computing device"""
        device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        print(f"Using device: {device}")
        return device
    
    def _setup_logging(self):
        """Setup logging và save directories"""
        self.save_dir = self.config['logging']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # TensorBoard
        if self.config['logging'].get('tensorboard', False):
            self.writer = SummaryWriter(log_dir=f"{self.save_dir}/tensorboard")
        else:
            self.writer = None
        
        # Save config
        with open(f"{self.save_dir}/config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _setup_random_seeds(self):
        """Setup random seeds for reproducibility"""
        seed = self.config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        print(f"Random seed set to: {seed}")
    
    def _create_model(self) -> nn.Module:
        """Create base model"""
        model_config = self.config['model']['base_model']
        
        # Estimate input dimension từ data
        available_datasets = self.data_loader.get_available_datasets()
        if available_datasets:
            sample_episodes = self.data_loader.get_dataset_episodes(available_datasets[0])
            if sample_episodes and sample_episodes[0].x is not None:
                input_dim = sample_episodes[0].x.size(1)
            else:
                input_dim = 64  # Default
        else:
            input_dim = 64
        
        model = CommunityDetectionModel(
            encoder_type=model_config['encoder_type'],
            input_dim=input_dim,
            hidden_dim=model_config['hidden_dim'],
            embedding_dim=model_config['embedding_dim'],
            num_classes=model_config.get('num_classes', 10),  # Will be adjusted dynamically
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.1),
            heads=model_config.get('heads', 8)  # For GAT
        ).to(self.device)
        
        print(f"Created model: {model_config['encoder_type'].upper()}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def train_epoch(self, episodes: List) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        epoch_metrics = []
        
        # Process episodes in batches
        batch_size = self.config['training']['meta_batch_size']
        
        for i in range(0, len(episodes), batch_size):
            batch_episodes = episodes[i:i + batch_size]
            
            # Meta-training step
            step_metrics = self.meta_optimizer.meta_train_step(batch_episodes)
            epoch_metrics.append(step_metrics)
        
        # Average metrics
        avg_metrics = {}
        if epoch_metrics:
            for key in epoch_metrics[0].keys():
                if isinstance(epoch_metrics[0][key], (int, float)):
                    avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
                else:
                    avg_metrics[key] = epoch_metrics[0][key]  # Take first for non-numeric
        
        return avg_metrics
    
    def validate(self, episodes: List) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for episode in episodes:
                # Generate pseudo-labels
                pseudo_labels = self.meta_optimizer._generate_meta_pseudo_labels(episode)
                
                # Get embeddings
                embeddings = self.model.get_embeddings(episode.x, episode.edge_index)
                
                # Compute metrics
                metrics = compute_all_metrics(
                    pred_labels=pseudo_labels,
                    edge_index=episode.edge_index,
                    embeddings=embeddings,
                    num_nodes=episode.num_nodes
                )
                
                all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[f'val_{key}'] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def train(self):
        """Main training loop"""
        print("Starting meta-training...")
        
        # Get training episodes
        train_episodes = []
        val_episodes = []
        
        for dataset_name in self.config['data']['datasets']:
            if dataset_name in self.data_loader.get_available_datasets():
                episodes = self.data_loader.get_dataset_episodes(dataset_name)
                
                # Split train/val
                split_idx = int(len(episodes) * self.config['data']['train_val_split'])
                train_episodes.extend(episodes[:split_idx])
                val_episodes.extend(episodes[split_idx:])
        
        print(f"Training episodes: {len(train_episodes)}")
        print(f"Validation episodes: {len(val_episodes)}")
        
        # Training loop
        num_epochs = self.config['training']['num_epochs']
        eval_interval = self.config['training']['evaluation_interval']
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            self.current_epoch = epoch
            
            # Shuffle episodes
            random.shuffle(train_episodes)
            
            # Train epoch
            train_metrics = self.train_epoch(train_episodes)
            self.train_losses.append(train_metrics.get('meta_loss', 0.0))
            
            # Validation
            if epoch % eval_interval == 0:
                val_metrics = self.validate(val_episodes)
                self.val_metrics.append(val_metrics)
                
                # Logging
                self._log_metrics(epoch, train_metrics, val_metrics)
                
                # Save best model
                current_metric = val_metrics.get('val_modularity', 0.0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self._save_model('best_model.pth')
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            if epoch % self.config['logging']['save_model_every'] == 0:
                self._save_model(f'checkpoint_epoch_{epoch}.pth')
        
        # Final save
        self._save_model('final_model.pth')
        self._save_training_history()
        
        print("Training completed!")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics"""
        print(f"\nEpoch {epoch}:")
        print(f"Train Loss: {train_metrics.get('meta_loss', 0.0):.4f}")
        
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
        
        # TensorBoard logging
        if self.writer:
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Validation/{key}', value, epoch)
    
    def _save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.meta_learner.state_dict(),
            'meta_params': self.meta_optimizer.meta_params.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        torch.save(checkpoint, f"{self.save_dir}/{filename}")
        print(f"Model saved: {filename}")
    
    def _save_training_history(self):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        
        with open(f"{self.save_dir}/training_history.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        # Plot training curves
        if self.train_losses:
            plot_training_curves(
                train_losses=self.train_losses,
                save_path=f"{self.save_dir}/training_curves.png"
            )


def main():
    parser = argparse.ArgumentParser(description='Train Meta-Learning Pseudo-Labels Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = MetaTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_metric = checkpoint['best_metric']
        print(f"Resumed from epoch {trainer.current_epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
