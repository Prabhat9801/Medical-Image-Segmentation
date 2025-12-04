"""
Training script for medical image segmentation models.
Supports UNet, UNet++, and TransUNet with different data fractions.
"""

import os
import sys
import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import models
from src.models.unet import UNet
from src.models.unetpp import UNetPlusPlus
from src.models.transunet import TransUNet

# Import dataset and utils
from src.datasets.isic_dataset import create_dataloaders
from src.utils import (
    CombinedLoss,
    dice_coefficient,
    iou_score,
    save_checkpoint,
    plot_training_history
)


def get_model(model_name: str, device: str):
    """
    Create model based on name.
    
    Args:
        model_name: One of ['unet', 'unetpp', 'transunet']
        device: Device to load model on
    
    Returns:
        model: PyTorch model
    """
    if model_name == 'unet':
        model = UNet(in_channels=3, out_channels=1, features=64)
    elif model_name == 'unetpp':
        model = UNetPlusPlus(in_channels=3, out_channels=1, features=32, deep_supervision=False)
    elif model_name == 'transunet':
        model = TransUNet(
            in_channels=3,
            out_channels=1,
            img_size=256,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Print model info
    info = model.get_model_info()
    print(f"\n{'='*60}")
    print(f"Model: {info['model_name']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"{'='*60}\n")
    
    return model


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
        
        # Update running metrics
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}'
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    epoch_iou = running_iou / len(loader)
    
    return epoch_loss, epoch_dice, epoch_iou


def validate(model, loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            
            # Update running metrics
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}'
            })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    epoch_iou = running_iou / len(loader)
    
    return epoch_loss, epoch_dice, epoch_iou


def train(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.model}_{int(args.data_fraction*100)}pct_{timestamp}"
    exp_dir = os.path.join(args.exp_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}")
    
    # Save arguments
    with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        data_fraction=args.data_fraction
    )
    
    # Create model
    print("\nCreating model...")
    model = get_model(args.model, device)
    
    # Loss function and optimizer
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training history
    history = {
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'lr': []
    }
    
    best_dice = 0.0
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_dice,
                os.path.join(exp_dir, 'best_model.pt')
            )
            print(f"  ✓ New best model saved! (Dice: {best_dice:.4f})")
        
        # Save latest model
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_dice,
                os.path.join(exp_dir, f'checkpoint_epoch_{epoch}.pt')
            )
        
        print(f"{'='*60}\n")
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs, val_loss, val_dice,
        os.path.join(exp_dir, 'final_model.pt')
    )
    
    # Save training history
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training history
    plot_training_history(
        history['train_loss'],
        history['val_loss'],
        history['train_dice'],
        history['val_dice'],
        save_path=os.path.join(exp_dir, 'training_history.png')
    )
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train medical image segmentation models')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'unetpp', 'transunet'],
                        help='Model architecture')
    
    # Data arguments
    parser.add_argument('--csv_file', type=str, default='data/processed/isic/splits.csv',
                        help='Path to CSV file with train/val/test splits')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                        help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--exp_dir', type=str, default='experiments',
                        help='Experiment directory')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
