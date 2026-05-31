"""
SSL Trainer for self-supervised line detection
Handles pretraining, fine-tuning, and evaluation
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from pathlib import Path
import json
from tqdm import tqdm

from training.losses import SSLTotalLoss

logger = logging.getLogger(__name__)


class SSLTrainer:
    """
    Trainer for self-supervised learning on line detection
    """
    
    def __init__(self, model, device, output_dir):
        """
        Args:
            model: dict with encoder, junction_head, mask_line_head, geometric_head
            device: torch device
            output_dir: directory to save checkpoints
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loss function
        self.criterion = SSLTotalLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
    
    def pretrain(self, train_loader, epochs=100, lr=1e-4, val_loader=None):
        """
        Pretraining phase: learn geometric and structural features
        """
        logger.info("Starting pretraining...")
        
        # Setup optimizer
        params = list(self.model['encoder'].parameters())
        for head_name in ['junction', 'mask_line', 'geometric']:
            params += list(self.model[head_name].parameters())
        
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, params)
            scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"pretrain_epoch_{epoch+1}.pt")
        
        logger.info("Pretraining completed!")
        self._save_checkpoint("pretrain_final.pt")
    
    def finetune(self, train_loader, epochs=50, lr=1e-5, val_loader=None):
        """
        Fine-tuning phase: adapt pretrained features to downstream task
        """
        logger.info("Starting fine-tuning...")
        
        # Freeze encoder, only train head
        for param in self.model['encoder'].parameters():
            param.requires_grad = False
        
        params = []
        for head_name in ['junction', 'mask_line', 'geometric']:
            params += list(self.model[head_name].parameters())
        
        optimizer = optim.Adam(params, lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, params)
            scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"finetune_epoch_{epoch+1}.pt")
        
        logger.info("Fine-tuning completed!")
        self._save_checkpoint("finetune_final.pt")
    
    def _train_epoch(self, train_loader, optimizer, params):
        """Single training epoch"""
        self.model['encoder'].train()
        for head_name in ['junction', 'mask_line', 'geometric']:
            self.model[head_name].train()
        
        total_loss = 0.0
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(self.device)
            targets = {key: val.to(self.device) for key, val in batch['targets'].items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            encoder_out = self.model['encoder'](images)
            features = encoder_out['spatial_features']
            
            line_map, direction_map = self.model['mask_line'](features)
            
            predictions = {
                'junction': self.model['junction'](features),
                'line_map': line_map,
                'direction_map': direction_map,
                'consistency': self.model['geometric'](features, features)
            }
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader):
        """Validation epoch"""
        self.model['encoder'].eval()
        for head_name in ['junction', 'mask_line', 'geometric']:
            self.model[head_name].eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                targets = {key: val.to(self.device) for key, val in batch['targets'].items()}
                
                encoder_out = self.model['encoder'](images)
                features = encoder_out['spatial_features']
                
                line_map, direction_map = self.model['mask_line'](features)
                
                predictions = {
                    'junction': self.model['junction'](features),
                    'line_map': line_map,
                    'direction_map': direction_map,
                    'consistency': self.model['geometric'](features, features)
                }
                
                loss, _ = self.criterion(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def evaluate(self, eval_loader, dataset_name):
        """
        Evaluate on benchmark datasets
        """
        self.model['encoder'].eval()
        for head_name in ['junction', 'mask_line', 'geometric']:
            self.model[head_name].eval()
        
        all_predictions = []
        all_targets = []
        
        logger.info(f"Evaluating on {dataset_name}...")
        
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                images = batch['image'].to(self.device)
                targets = {key: val.to(self.device) for key, val in batch['targets'].items()}
                
                encoder_out = self.model['encoder'](images)
                features = encoder_out['spatial_features']
                
                line_map, direction_map = self.model['mask_line'](features)
                
                predictions = {
                    'junction': self.model['junction'](features),
                    'line_map': line_map,
                    'direction_map': direction_map,
                    'consistency': self.model['geometric'](features, features)
                }
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets)
        self.history['metrics'][dataset_name] = metrics
        
        return metrics
    
    def _compute_metrics(self, predictions, targets):
        """Compute evaluation metrics"""
        # Placeholder for metric computation
        # Implement sAP, F-score, etc.
        metrics = {
            'sAP': 0.0,
            'f_score': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        return metrics
    
    def _save_checkpoint(self, name):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / name
        checkpoint = {
            'encoder': self.model['encoder'].state_dict(),
            'junction_head': self.model['junction'].state_dict(),
            'mask_line_head': self.model['mask_line'].state_dict(),
            'geometric_head': self.model['geometric'].state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
