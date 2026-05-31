"""
LINEA Transformer-based Encoder for Line Detection
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.models as models


class LineaEncoder(nn.Module):
    """
    LINEA Encoder: Transformer-based architecture for line detection
    Implements the backbone for extracting geometric and structural features
    """
    
    def __init__(self, hidden_dim=256, num_layers=6, nhead=8, dim_feedforward=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # ResNet backbone for initial feature extraction
        self.backbone = models.resnet50(pretrained=True)
        self.backbone_out_dim = 2048
        
        # Reduce backbone output to hidden_dim
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(self.backbone_out_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            features: Tensor of shape (B, hidden_dim, H', W')
            flat_features: Tensor of shape (B, N, hidden_dim) for transformer
        """
        # Backbone feature extraction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Reduce dimensions
        features = self.reduce_conv(x)
        
        # Flatten for transformer
        B, C, H, W = features.shape
        flat_features = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(flat_features)
        transformer_out = self.ln(transformer_out)
        
        # Reshape back
        spatial_features = transformer_out.transpose(1, 2).view(B, C, H, W)
        
        return {
            'features': features,
            'spatial_features': spatial_features,
            'flat_features': transformer_out
        }


class LineaEncoderLite(nn.Module):
    """Lightweight version using ResNet18 backbone"""
    
    def __init__(self, hidden_dim=128, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.backbone = models.resnet18(pretrained=True)
        self.backbone_out_dim = 512
        
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(self.backbone_out_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        features = self.reduce_conv(x)
        
        B, C, H, W = features.shape
        flat_features = features.flatten(2).transpose(1, 2)
        
        transformer_out = self.transformer_encoder(flat_features)
        transformer_out = self.ln(transformer_out)
        
        spatial_features = transformer_out.transpose(1, 2).view(B, C, H, W)
        
        return {
            'features': features,
            'spatial_features': spatial_features,
            'flat_features': transformer_out
        }
