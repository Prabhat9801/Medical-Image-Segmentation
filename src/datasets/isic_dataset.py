"""
ISIC Dataset loader for skin lesion segmentation.
Reads preprocessed images and masks from CSV file.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ISICDataset(Dataset):
    """
    ISIC Skin Lesion Segmentation Dataset.
    
    Args:
        csv_file: Path to CSV file with columns [image_path, mask_path, split]
        split: One of ['train', 'val', 'test']
        transform: Albumentations transform pipeline (optional)
        data_fraction: Fraction of data to use (0.0 to 1.0)
    """
    
    def __init__(
        self,
        csv_file: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        data_fraction: float = 1.0
    ):
        self.csv_file = csv_file
        self.split = split
        self.transform = transform
        self.data_fraction = data_fraction
        
        # Load CSV
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        # Apply data fraction
        if data_fraction < 1.0:
            n_samples = int(len(self.df) * data_fraction)
            self.df = self.df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} samples for {split} split (fraction: {data_fraction})")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (3, H, W)
            mask: Tensor of shape (1, H, W)
        """
        row = self.df.iloc[idx]
        
        # Load image and mask
        image_path = row['image_path']
        mask_path = row['mask_path']
        
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Normalize mask to 0-1
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default: normalize and convert to tensor
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dim
        
        return image, mask


# ==================== TRANSFORMS ====================

def get_train_transforms(image_size: int = 256) -> A.Compose:
    """
    Training augmentation pipeline.
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(p=1.0, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
            A.MotionBlur(p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
            A.RGBShift(p=1.0),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 256) -> A.Compose:
    """
    Validation/Test transform pipeline (no augmentation).
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ==================== DATALOADER CREATION ====================

def create_dataloaders(
    csv_file: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 256,
    data_fraction: float = 1.0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        csv_file: Path to splits CSV file
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Target image size
        data_fraction: Fraction of training data to use
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = ISICDataset(
        csv_file=csv_file,
        split='train',
        transform=get_train_transforms(image_size),
        data_fraction=data_fraction
    )
    
    val_dataset = ISICDataset(
        csv_file=csv_file,
        split='val',
        transform=get_val_transforms(image_size),
        data_fraction=1.0  # Always use full validation set
    )
    
    test_dataset = ISICDataset(
        csv_file=csv_file,
        split='test',
        transform=get_val_transforms(image_size),
        data_fraction=1.0  # Always use full test set
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # For easier visualization
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset loading
    csv_file = '../data/processed/isic/splits.csv'
    
    if os.path.exists(csv_file):
        dataset = ISICDataset(csv_file, split='train', data_fraction=0.1)
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading one sample
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    else:
        print(f"CSV file not found: {csv_file}")
        print("Please run preprocessing notebook first!")
