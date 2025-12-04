"""
Evaluation script for trained segmentation models.
Generates predictions and calculates metrics on test set.
"""

import os
import argparse
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import models
from models.unet import UNet
from models.unetpp import UNetPlusPlus
from models.transunet import TransUNet

# Import dataset and utils
from datasets.isic_dataset import ISICDataset, get_val_transforms
from utils import (
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    load_checkpoint,
    plot_predictions,
    plot_overlay
)


def get_model(model_name: str, device: str):
    """Create model based on name."""
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
    return model


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from ImageNet normalization.
    
    Args:
        image: Normalized image (C, H, W)
    
    Returns:
        Denormalized image (H, W, C) in range [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    image = image * std + mean
    image = np.clip(image, 0, 1)
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    
    return image


def evaluate(args):
    """Main evaluation function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, os.path.basename(args.checkpoint).replace('.pt', ''))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create dataset
    print("\nLoading test dataset...")
    test_dataset = ISICDataset(
        csv_file=args.csv_file,
        split='test',
        transform=get_val_transforms(args.image_size)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("\nLoading model...")
    model = get_model(args.model, device)
    
    # Load checkpoint
    model, _, epoch = load_checkpoint(model, None, args.checkpoint, device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {epoch}")
    
    # Evaluation metrics
    all_dice = []
    all_iou = []
    all_accuracy = []
    
    # Storage for visualizations
    images_list = []
    masks_list = []
    predictions_list = []
    
    # Evaluate
    print("\nEvaluating...")
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate metrics
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = pixel_accuracy(outputs, masks)
            
            all_dice.append(dice)
            all_iou.append(iou)
            all_accuracy.append(acc)
            
            # Store for visualization (first N samples)
            if idx < args.num_vis:
                # Convert to numpy
                image_np = images[0].cpu().numpy()
                mask_np = masks[0, 0].cpu().numpy()
                pred_np = torch.sigmoid(outputs[0, 0]).cpu().numpy()
                pred_np = (pred_np > 0.5).astype(np.float32)
                
                # Denormalize image
                image_np = denormalize_image(image_np)
                
                images_list.append(image_np)
                masks_list.append(mask_np)
                predictions_list.append(pred_np)
    
    # Calculate statistics
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    mean_iou = np.mean(all_iou)
    std_iou = np.std(all_iou)
    mean_acc = np.mean(all_accuracy)
    std_acc = np.std(all_accuracy)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Dice Coefficient: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"IoU Score:        {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Pixel Accuracy:   {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'model': args.model,
        'epoch': epoch,
        'num_samples': len(test_dataset),
        'metrics': {
            'dice': {
                'mean': float(mean_dice),
                'std': float(std_dice),
                'all': [float(x) for x in all_dice]
            },
            'iou': {
                'mean': float(mean_iou),
                'std': float(std_iou),
                'all': [float(x) for x in all_iou]
            },
            'accuracy': {
                'mean': float(mean_acc),
                'std': float(std_acc),
                'all': [float(x) for x in all_accuracy]
            }
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {os.path.join(output_dir, 'results.json')}")
    
    # Create visualizations
    if len(images_list) > 0:
        print(f"\nCreating visualizations...")
        
        # Plot predictions
        plot_predictions(
            np.array(images_list),
            np.array(masks_list),
            np.array(predictions_list),
            num_samples=min(args.num_vis, len(images_list)),
            save_path=os.path.join(output_dir, 'predictions.png')
        )
        
        # Plot individual overlays
        for i in range(min(4, len(images_list))):
            plot_overlay(
                images_list[i],
                masks_list[i],
                predictions_list[i],
                save_path=os.path.join(output_dir, f'overlay_{i+1}.png')
            )
        
        print(f"Visualizations saved to: {output_dir}")
    
    # Create metrics distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(all_dice, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(mean_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dice:.4f}')
    axes[0].set_xlabel('Dice Coefficient')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Dice Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_iou, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(mean_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_iou:.4f}')
    axes[1].set_xlabel('IoU Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('IoU Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(all_accuracy, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.4f}')
    axes[2].set_xlabel('Pixel Accuracy')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Accuracy Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"Metrics distribution saved to: {os.path.join(output_dir, 'metrics_distribution.png')}")
    
    print(f"\n✓ Evaluation completed!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained segmentation models')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'unetpp', 'transunet'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--csv_file', type=str, default='data/processed/isic/splits.csv',
                        help='Path to CSV file with train/val/test splits')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    
    # Evaluation arguments
    parser.add_argument('--num_vis', type=int, default=8,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='reports/figures',
                        help='Output directory for results')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Evaluate
    evaluate(args)


if __name__ == '__main__':
    main()
