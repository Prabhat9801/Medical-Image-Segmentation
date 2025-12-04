"""
Fast ISIC2018 Dataset Download Script using KaggleHub
Downloads the dataset and organizes it for preprocessing.
"""

import os
import shutil
from pathlib import Path

def download_dataset():
    """Download ISIC2018 dataset using kagglehub."""
    print("="*60)
    print("ISIC2018 Dataset Download")
    print("="*60)
    
    try:
        import kagglehub
        print("\n✅ KaggleHub found!")
    except ImportError:
        print("\n❌ KaggleHub not installed!")
        print("\nInstalling kagglehub...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'kagglehub'])
        import kagglehub
        print("✅ KaggleHub installed successfully!")
    
    # Download dataset
    print("\n📥 Downloading ISIC2018 dataset from Kaggle...")
    print("This may take 5-15 minutes depending on your internet speed...")
    
    path = kagglehub.dataset_download("tschandl/isic2018-challenge-task1-data-segmentation")
    
    print(f"\n✅ Dataset downloaded to: {path}")
    
    return path


def organize_dataset(download_path):
    """Organize downloaded dataset into project structure."""
    print("\n" + "="*60)
    print("Organizing Dataset")
    print("="*60)
    
    project_root = Path(__file__).parent
    raw_dir = project_root / "data" / "raw" / "isic"
    
    # Create directories
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    download_path = Path(download_path)
    
    # Find all folders in downloaded dataset
    print(f"\n📂 Scanning downloaded files...")
    
    # ISIC2018 structure:
    # - ISIC2018_Task1-2_Training_Input/ (images)
    # - ISIC2018_Task1_Training_GroundTruth/ (masks)
    # - ISIC2018_Task1-2_Validation_Input/ (validation images)
    # - ISIC2018_Task1-2_Test_Input/ (test images)
    
    copied_images = 0
    copied_masks = 0
    
    # Copy training images
    train_input = download_path / "ISIC2018_Task1-2_Training_Input"
    if train_input.exists():
        print(f"\n📁 Found training images: {train_input}")
        for img_file in train_input.glob("*.jpg"):
            dest = images_dir / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)
                copied_images += 1
        print(f"✅ Copied {copied_images} training images")
    
    # Copy validation images
    val_input = download_path / "ISIC2018_Task1-2_Validation_Input"
    if val_input.exists():
        print(f"\n📁 Found validation images: {val_input}")
        val_count = 0
        for img_file in val_input.glob("*.jpg"):
            dest = images_dir / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)
                copied_images += 1
                val_count += 1
        print(f"✅ Copied {val_count} validation images")
    
    # Copy test images (no masks available)
    test_input = download_path / "ISIC2018_Task1-2_Test_Input"
    if test_input.exists():
        print(f"\n📁 Found test images: {test_input}")
        test_count = 0
        for img_file in test_input.glob("*.jpg"):
            dest = images_dir / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)
                copied_images += 1
                test_count += 1
        print(f"✅ Copied {test_count} test images (no masks)")
    
    # Copy ground truth masks
    gt_dir = download_path / "ISIC2018_Task1_Training_GroundTruth"
    if gt_dir.exists():
        print(f"\n📁 Found ground truth masks: {gt_dir}")
        for mask_file in gt_dir.glob("*.png"):
            dest = masks_dir / mask_file.name
            if not dest.exists():
                shutil.copy2(mask_file, dest)
                copied_masks += 1
        print(f"✅ Copied {copied_masks} masks")
    
    # Summary
    print("\n" + "="*60)
    print("Dataset Organization Complete!")
    print("="*60)
    print(f"📊 Total images copied: {copied_images}")
    print(f"📊 Total masks copied: {copied_masks}")
    print(f"\n📂 Images location: {images_dir}")
    print(f"📂 Masks location: {masks_dir}")
    
    # Note about test set
    if test_count > 0:
        print(f"\n⚠️  Note: {test_count} test images have no masks (as per ISIC2018 challenge)")
        print("   These will be excluded during preprocessing.")
    
    return copied_images, copied_masks


def verify_dataset():
    """Verify dataset is properly organized."""
    print("\n" + "="*60)
    print("Verifying Dataset")
    print("="*60)
    
    project_root = Path(__file__).parent
    images_dir = project_root / "data" / "raw" / "isic" / "images"
    masks_dir = project_root / "data" / "raw" / "isic" / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        print("❌ Dataset directories not found!")
        return False
    
    images = list(images_dir.glob("*.jpg"))
    masks = list(masks_dir.glob("*.png"))
    
    print(f"\n✅ Found {len(images)} images")
    print(f"✅ Found {len(masks)} masks")
    
    # Check for matching pairs
    image_ids = {img.stem for img in images}
    mask_ids = {mask.stem.replace("_segmentation", "") for mask in masks}
    
    matched = image_ids & mask_ids
    print(f"✅ Matched pairs: {len(matched)}")
    
    if len(matched) > 0:
        print("\n✅ Dataset verification successful!")
        print(f"\n📝 Next step: Run preprocessing notebook")
        print(f"   jupyter notebook notebooks/01_isic_preprocessing.ipynb")
        return True
    else:
        print("\n❌ No matched image-mask pairs found!")
        return False


def main():
    """Main function."""
    print("\n" + "="*60)
    print("🏥 ISIC2018 Dataset Download & Setup")
    print("="*60)
    
    # Step 1: Download
    download_path = download_dataset()
    
    # Step 2: Organize
    organize_dataset(download_path)
    
    # Step 3: Verify
    verify_dataset()
    
    print("\n" + "="*60)
    print("✅ All Done!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("   1. Run: jupyter notebook notebooks/01_isic_preprocessing.ipynb")
    print("   2. Execute all cells to preprocess the data")
    print("   3. Continue with model training")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
