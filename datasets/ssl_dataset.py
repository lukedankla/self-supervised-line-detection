"""
Self-Supervised Dataset for Line Detection
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path


class SSLLineDataset(Dataset):
    """
    SSL Dataset for line detection with multiple modes:
    - pretrain: Self-supervised pretraining
    - finetune: Limited annotated data
    - eval: Evaluation on benchmarks
    """
    
    def __init__(self, data_dir, mode='pretrain', dataset='wireframe', split='train'):
        """
        Args:
            data_dir: Root directory of datasets
            mode: 'pretrain', 'finetune', or 'eval'
            dataset: 'wireframe', 'yorkurban', 'aerial', 'sketch'
            split: 'train', 'val', 'test'
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.dataset = dataset
        self.split = split
        
        # Data augmentation
        self.transform = self._get_transforms()
        
        # Load dataset paths
        self.image_paths = self._load_dataset()
    
    def _load_dataset(self):
        """Load image paths for the specified dataset"""
        dataset_dir = self.data_dir / self.dataset / self.split
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        image_paths = sorted(dataset_dir.glob('*.jpg')) + sorted(dataset_dir.glob('*.png'))
        return image_paths
    
    def _get_transforms(self):
        """Get augmentation transforms"""
        if self.mode == 'pretrain':
            return transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single sample with targets"""
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Generate targets based on mode
        if self.mode == 'pretrain':
            targets = self._generate_pretext_targets(image)
        else:
            targets = self._load_annotation(image_path)
        
        return {
            'image': image,
            'targets': targets,
            'path': str(image_path)
        }
    
    def _generate_pretext_targets(self, image):
        """Generate pretext task targets for self-supervised learning"""
        _, H, W = image.shape
        
        # Junction prediction target
        junction_target = torch.zeros(1, H, W)
        # Add random junction points for demonstration
        num_junctions = np.random.randint(10, 50)
        for _ in range(num_junctions):
            y, x = np.random.randint(0, H), np.random.randint(0, W)
            junction_target[0, max(0, y-2):min(H, y+3), max(0, x-2):min(W, x+3)] = 1
        
        # Line detection target
        line_target = torch.zeros(1, H, W)
        direction_target = torch.zeros(8, H, W)
        # Add random line segments
        num_lines = np.random.randint(5, 20)
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, W), np.random.randint(0, H)
            x2, y2 = np.random.randint(0, W), np.random.randint(0, H)
            # Draw line on target
            self._draw_line(line_target[0], x1, y1, x2, y2)
            # Assign direction
            dx, dy = x2 - x1, y2 - y1
            angle = np.arctan2(dy, dx) * 4 / np.pi  # Map to 8 directions
            dir_idx = int((angle + 4) % 8)
            direction_target[dir_idx, max(0, y1):min(H, y2+1), max(0, x1):min(W, x2+1)] = 1
        
        # Geometric consistency target
        consistency_target = torch.ones(1, H, W) * 0.8
        
        return {
            'junction': junction_target,
            'line_map': line_target,
            'direction_map': direction_target / (direction_target.sum(dim=0, keepdim=True) + 1e-8),
            'consistency': consistency_target
        }
    
    def _load_annotation(self, image_path):
        """Load ground truth annotation"""
        # Placeholder - implement based on your annotation format
        H, W = 512, 512
        
        return {
            'junction': torch.zeros(1, H, W),
            'line_map': torch.zeros(1, H, W),
            'direction_map': torch.ones(8, H, W) / 8,
            'consistency': torch.ones(1, H, W)
        }
    
    @staticmethod
    def _draw_line(target, x1, y1, x2, y2, thickness=2):
        """Draw line on target map"""
        # Simple line drawing
        steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
        for i in range(steps):
            t = i / steps if steps > 1 else 0
            x, y = int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)
            if 0 <= y < target.shape[0] and 0 <= x < target.shape[1]:
                target[max(0, y-thickness):min(target.shape[0], y+thickness+1),
                       max(0, x-thickness):min(target.shape[1], x+thickness+1)] = 1
