"""
Utility functions for medical image segmentation.
Includes loss functions, metrics, and visualization helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# ==================== LOSS FUNCTIONS ====================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted masks (B, 1, H, W) - logits or probabilities
            target: Ground truth masks (B, 1, H, W) - binary 0/1
        Returns:
            Dice loss value
        """
        # Apply sigmoid if needed
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combination of Dice Loss and Binary Cross Entropy.
    """
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


# ==================== METRICS ====================

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient (F1 score for segmentation).
    
    Args:
        pred: Predicted masks (B, 1, H, W) - logits or probabilities
        target: Ground truth masks (B, 1, H, W) - binary 0/1
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient value (0 to 1, higher is better)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (Jaccard Index).
    
    Args:
        pred: Predicted masks (B, 1, H, W) - logits or probabilities
        target: Ground truth masks (B, 1, H, W) - binary 0/1
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score (0 to 1, higher is better)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate pixel-wise accuracy.
    
    Args:
        pred: Predicted masks (B, 1, H, W)
        target: Ground truth masks (B, 1, H, W)
    
    Returns:
        Pixel accuracy (0 to 1)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    correct = (pred == target).float().sum()
    total = target.numel()
    
    return (correct / total).item()


# ==================== VISUALIZATION ====================

def plot_predictions(
    images: np.ndarray,
    masks: np.ndarray,
    predictions: np.ndarray,
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> None:
    """
    Plot original images, ground truth masks, and predictions side by side.
    
    Args:
        images: Original images (N, H, W, 3)
        masks: Ground truth masks (N, H, W)
        predictions: Predicted masks (N, H, W)
        num_samples: Number of samples to plot
        save_path: Path to save the figure (optional)
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[i], cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.5,
    save_path: Optional[str] = None
) -> None:
    """
    Plot image with overlaid ground truth (green) and prediction (red).
    
    Args:
        image: Original image (H, W, 3)
        mask: Ground truth mask (H, W)
        prediction: Predicted mask (H, W)
        alpha: Transparency for overlay
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(image)
    mask_overlay = np.zeros((*mask.shape, 3))
    mask_overlay[mask > 0.5] = [0, 1, 0]  # Green
    axes[1].imshow(mask_overlay, alpha=alpha)
    axes[1].set_title('Ground Truth Overlay')
    axes[1].axis('off')
    
    # Prediction overlay
    axes[2].imshow(image)
    pred_overlay = np.zeros((*prediction.shape, 3))
    pred_overlay[prediction > 0.5] = [1, 0, 0]  # Red
    axes[2].imshow(pred_overlay, alpha=alpha)
    axes[2].set_title('Prediction Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_training_history(
    train_losses: list,
    val_losses: list,
    train_dices: list,
    val_dices: list,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss and Dice scores.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_dices: List of training Dice scores
        val_dices: List of validation Dice scores
        save_path: Path to save the figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dice plot
    ax2.plot(train_dices, label='Train Dice', linewidth=2)
    ax2.plot(val_dices, label='Val Dice', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Coefficient', fontsize=12)
    ax2.set_title('Training and Validation Dice Score', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# ==================== HELPER FUNCTIONS ====================

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    dice: float,
    path: str
) -> None:
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'dice': dice,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: str = 'cpu'
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int]:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint loaded from {path} (Epoch {epoch})")
    return model, optimizer, epoch
