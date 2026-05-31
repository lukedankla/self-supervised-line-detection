"""
Pretext Task Heads for Self-Supervised Learning
1. Junction Prediction - Predict junctions/keypoints
2. Mask Line Detection - Detect lines from masked images
3. Geometric Consistency - Maintain geometric consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JunctionPredictionHead(nn.Module):
    """
    Predicts junction points (line intersections and endpoints)
    Output: Heatmap of junction locations
    """
    
    def __init__(self, hidden_dim, num_classes=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Convolutional layers for heatmap generation
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim // 2)
        self.conv3 = nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) spatial features from encoder
        Returns:
            junction_heatmap: (B, 1, H, W) normalized heatmap
        """
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        # Apply sigmoid for junction probability
        junction_heatmap = torch.sigmoid(x)
        
        return junction_heatmap


class MaskLineDetectionHead(nn.Module):
    """
    Line Segment Detection from Masked Images
    Learns to detect complete lines from partially visible lines
    Output: Line probability map and line direction
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Line detection branch
        self.line_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.line_bn1 = nn.BatchNorm2d(hidden_dim)
        self.line_conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.line_bn2 = nn.BatchNorm2d(hidden_dim // 2)
        self.line_out = nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        
        # Direction estimation branch (8 directions)
        self.dir_conv1 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.dir_bn1 = nn.BatchNorm2d(hidden_dim // 2)
        self.dir_conv2 = nn.Conv2d(hidden_dim // 2, 8, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) spatial features from encoder
        Returns:
            line_map: (B, 1, H, W) line probability map
            direction_map: (B, 8, H, W) 8-direction encoding
        """
        # Line detection
        x_line = F.relu(self.line_bn1(self.line_conv1(features)))
        x_line = F.relu(self.line_bn2(self.line_conv2(x_line)))
        line_map = torch.sigmoid(self.line_out(x_line))
        
        # Direction estimation
        x_dir = F.relu(self.dir_bn1(self.dir_conv1(features)))
        direction_map = F.softmax(self.dir_conv2(x_dir), dim=1)
        
        return line_map, direction_map


class GeometricConsistencyHead(nn.Module):
    """
    Ensures geometric consistency in line detection
    Enforces constraints like collinearity and alignment
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Consistency check network
        self.consistency_conv1 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.consistency_bn1 = nn.BatchNorm2d(hidden_dim)
        self.consistency_conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.consistency_bn2 = nn.BatchNorm2d(hidden_dim // 2)
        self.consistency_out = nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
    
    def forward(self, features1, features2):
        """
        Args:
            features1: (B, C, H, W) features from first view
            features2: (B, C, H, W) features from second view (transformed)
        Returns:
            consistency_score: (B, 1, H, W) consistency probability map
        """
        # Concatenate features
        combined = torch.cat([features1, features2], dim=1)
        
        x = F.relu(self.consistency_bn1(self.consistency_conv1(combined)))
        x = F.relu(self.consistency_bn2(self.consistency_conv2(x)))
        consistency_score = torch.sigmoid(self.consistency_out(x))
        
        return consistency_score
