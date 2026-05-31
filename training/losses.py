"""
Loss functions for self-supervised pretext tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JunctionPredictionLoss(nn.Module):
    """
    Loss for junction prediction task
    Uses focal loss to handle class imbalance
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predicted_heatmap, target_heatmap):
        """
        Args:
            predicted_heatmap: (B, 1, H, W) predicted junction heatmap
            target_heatmap: (B, 1, H, W) ground truth junction heatmap
        Returns:
            loss: scalar loss value
        """
        # Focal loss
        ce_loss = F.binary_cross_entropy(predicted_heatmap, target_heatmap, reduction='none')
        p_t = torch.where(target_heatmap == 1, predicted_heatmap, 1 - predicted_heatmap)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class MaskLineDetectionLoss(nn.Module):
    """
    Loss for mask line detection task
    Combines line detection loss and direction consistency loss
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, line_map, direction_map, target_line, target_direction):
        """
        Args:
            line_map: (B, 1, H, W) predicted line probability
            direction_map: (B, 8, H, W) predicted direction distribution
            target_line: (B, 1, H, W) target line map
            target_direction: (B, 8, H, W) target direction distribution
        Returns:
            loss: scalar loss value
        """
        # Line detection loss (BCE)
        line_loss = F.binary_cross_entropy(line_map, target_line)
        
        # Direction loss (KL divergence)
        direction_loss = F.kl_div(
            torch.log(direction_map + 1e-8),
            target_direction,
            reduction='batchmean'
        )
        
        total_loss = line_loss + 0.5 * direction_loss
        return total_loss


class GeometricConsistencyLoss(nn.Module):
    """
    Loss for enforcing geometric consistency
    Encourages consistent predictions across transformed views
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, consistency_score, target_consistency):
        """
        Args:
            consistency_score: (B, 1, H, W) predicted consistency
            target_consistency: (B, 1, H, W) target consistency
        Returns:
            loss: scalar loss value
        """
        loss = F.binary_cross_entropy(consistency_score, target_consistency)
        return loss


class SSLTotalLoss(nn.Module):
    """
    Combined loss for all pretext tasks
    """
    
    def __init__(self, weights=None):
        super().__init__()
        self.junction_loss = JunctionPredictionLoss()
        self.mask_line_loss = MaskLineDetectionLoss()
        self.consistency_loss = GeometricConsistencyLoss()
        
        # Loss weights for each task
        self.weights = weights or {
            'junction': 1.0,
            'mask_line': 1.0,
            'consistency': 0.5
        }
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with keys 'junction', 'mask_line', 'consistency'
            targets: dict with corresponding target values
        Returns:
            total_loss: weighted sum of all losses
            loss_dict: individual loss values
        """
        loss_dict = {}
        
        # Junction prediction loss
        junction_loss = self.junction_loss(
            predictions['junction'],
            targets['junction']
        )
        loss_dict['junction'] = junction_loss
        
        # Mask line detection loss
        mask_line_loss = self.mask_line_loss(
            predictions['line_map'],
            predictions['direction_map'],
            targets['line_map'],
            targets['direction_map']
        )
        loss_dict['mask_line'] = mask_line_loss
        
        # Geometric consistency loss
        consistency_loss = self.consistency_loss(
            predictions['consistency'],
            targets['consistency']
        )
        loss_dict['consistency'] = consistency_loss
        
        # Weighted total loss
        total_loss = (
            self.weights['junction'] * junction_loss +
            self.weights['mask_line'] * mask_line_loss +
            self.weights['consistency'] * consistency_loss
        )
        
        return total_loss, loss_dict
