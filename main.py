"""
Self-Supervised Learning Framework for Line Detection
Main training and evaluation script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import logging

from models.linea_encoder import LineaEncoder
from models.pretext_tasks import JunctionPredictionHead, MaskLineDetectionHead, GeometricConsistencyHead
from datasets.ssl_dataset import SSLLineDataset
from training.trainer import SSLTrainer
from utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Self-Supervised Line Detection Framework')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file path')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--pretrain', action='store_true', help='Perform pretraining')
    parser.add_argument('--finetune', action='store_true', help='Perform finetuning')
    parser.add_argument('--evaluate', action='store_true', help='Perform evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    logger.info("Initializing model...")
    encoder = LineaEncoder()
    junction_head = JunctionPredictionHead(encoder.hidden_dim)
    mask_line_head = MaskLineDetectionHead(encoder.hidden_dim)
    geometric_head = GeometricConsistencyHead(encoder.hidden_dim)
    
    model = {
        'encoder': encoder,
        'junction': junction_head,
        'mask_line': mask_line_head,
        'geometric': geometric_head
    }
    
    # Move to device
    for key in model:
        model[key] = model[key].to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for key in model:
            if key in checkpoint:
                model[key].load_state_dict(checkpoint[key])
    
    # Pretraining phase
    if args.pretrain:
        logger.info("Starting pretraining phase...")
        pretrain_dataset = SSLLineDataset(
            data_dir=args.data_dir,
            mode='pretrain'
        )
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        trainer = SSLTrainer(model, device, args.output_dir)
        trainer.pretrain(pretrain_loader, epochs=args.epochs, lr=args.lr)
        logger.info("Pretraining completed!")
    
    # Fine-tuning phase
    if args.finetune:
        logger.info("Starting fine-tuning phase...")
        finetune_dataset = SSLLineDataset(
            data_dir=args.data_dir,
            mode='finetune'
        )
        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        trainer = SSLTrainer(model, device, args.output_dir)
        trainer.finetune(finetune_loader, epochs=args.epochs, lr=args.lr * 0.1)
        logger.info("Fine-tuning completed!")
    
    # Evaluation phase
    if args.evaluate:
        logger.info("Starting evaluation phase...")
        eval_datasets = {
            'wireframe': SSLLineDataset(data_dir=args.data_dir, mode='eval', dataset='wireframe'),
            'yorkurban': SSLLineDataset(data_dir=args.data_dir, mode='eval', dataset='yorkurban'),
            'aerial': SSLLineDataset(data_dir=args.data_dir, mode='eval', dataset='aerial'),
            'sketch': SSLLineDataset(data_dir=args.data_dir, mode='eval', dataset='sketch')
        }
        
        trainer = SSLTrainer(model, device, args.output_dir)
        for dataset_name, eval_dataset in eval_datasets.items():
            eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
            metrics = trainer.evaluate(eval_loader, dataset_name)
            logger.info(f"Evaluation on {dataset_name}: {metrics}")


if __name__ == '__main__':
    main()
