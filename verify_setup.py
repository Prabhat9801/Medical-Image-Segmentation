"""
Project Setup Verification Script
Run this to verify all components are correctly installed and configured.
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists."""
    if os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} NOT FOUND")
        return False

def check_imports():
    """Check if required packages can be imported."""
    print("\n" + "="*60)
    print("Checking Python Packages...")
    print("="*60)
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            all_ok = False
    
    return all_ok

def check_optional_imports():
    """Check optional packages (for Colab)."""
    print("\n" + "="*60)
    print("Checking Optional Packages (for Colab)...")
    print("="*60)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'TIMM',
        'albumentations': 'Albumentations',
    }
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"⚠️  {name} NOT installed (install in Colab)")

def main():
    print("\n" + "="*60)
    print("Medical Image Segmentation - Project Verification")
    print("="*60)
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Check directory structure
    print("\n" + "="*60)
    print("Checking Directory Structure...")
    print("="*60)
    
    dirs_ok = True
    dirs_ok &= check_directory(project_root / "data", "Data directory")
    dirs_ok &= check_directory(project_root / "data/raw", "Raw data directory")
    dirs_ok &= check_directory(project_root / "data/processed", "Processed data directory")
    dirs_ok &= check_directory(project_root / "src", "Source code directory")
    dirs_ok &= check_directory(project_root / "src/models", "Models directory")
    dirs_ok &= check_directory(project_root / "src/datasets", "Datasets directory")
    dirs_ok &= check_directory(project_root / "notebooks", "Notebooks directory")
    dirs_ok &= check_directory(project_root / "reports", "Reports directory")
    
    # Check source files
    print("\n" + "="*60)
    print("Checking Source Files...")
    print("="*60)
    
    files_ok = True
    files_ok &= check_file(project_root / "src/utils.py", "Utils module")
    files_ok &= check_file(project_root / "src/train.py", "Training script")
    files_ok &= check_file(project_root / "src/eval.py", "Evaluation script")
    files_ok &= check_file(project_root / "src/models/unet.py", "UNet model")
    files_ok &= check_file(project_root / "src/models/unetpp.py", "UNet++ model")
    files_ok &= check_file(project_root / "src/models/transunet.py", "TransUNet model")
    files_ok &= check_file(project_root / "src/datasets/isic_dataset.py", "ISIC dataset")
    
    # Check notebooks
    print("\n" + "="*60)
    print("Checking Notebooks...")
    print("="*60)
    
    notebooks_ok = True
    notebooks_ok &= check_file(project_root / "notebooks/01_isic_preprocessing.ipynb", "Preprocessing notebook")
    notebooks_ok &= check_file(project_root / "notebooks/02_model_testing.ipynb", "Model testing notebook")
    notebooks_ok &= check_file(project_root / "notebooks/03_colab_training.ipynb", "Colab training notebook")
    
    # Check documentation
    print("\n" + "="*60)
    print("Checking Documentation...")
    print("="*60)
    
    docs_ok = True
    docs_ok &= check_file(project_root / "README.md", "README")
    docs_ok &= check_file(project_root / "QUICKSTART.md", "Quick Start Guide")
    docs_ok &= check_file(project_root / "PROJECT_SUMMARY.md", "Project Summary")
    docs_ok &= check_file(project_root / "requirements.txt", "Requirements file")
    docs_ok &= check_file(project_root / "reports/report.md", "Report template")
    
    # Check Python packages
    packages_ok = check_imports()
    check_optional_imports()
    
    # Check if data exists
    print("\n" + "="*60)
    print("Checking Data...")
    print("="*60)
    
    raw_images = project_root / "data/raw/isic/images"
    raw_masks = project_root / "data/raw/isic/masks"
    
    if raw_images.exists() and list(raw_images.glob("*")):
        print(f"✅ Raw images found: {len(list(raw_images.glob('*')))} files")
    else:
        print("⚠️  Raw images not found - Download ISIC dataset")
    
    if raw_masks.exists() and list(raw_masks.glob("*")):
        print(f"✅ Raw masks found: {len(list(raw_masks.glob('*')))} files")
    else:
        print("⚠️  Raw masks not found - Download ISIC dataset")
    
    processed_csv = project_root / "data/processed/isic/splits.csv"
    if processed_csv.exists():
        print(f"✅ Processed data splits found")
    else:
        print("⚠️  Processed data not found - Run preprocessing notebook")
    
    # Final summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    all_ok = dirs_ok and files_ok and notebooks_ok and docs_ok and packages_ok
    
    if all_ok:
        print("✅ All core components verified successfully!")
        print("\n📋 Next Steps:")
        print("   1. Download ISIC dataset if not done")
        print("   2. Run preprocessing notebook")
        print("   3. Test models locally (optional)")
        print("   4. Upload processed data to Google Drive")
        print("   5. Train models in Colab")
        print("\n📚 See QUICKSTART.md for detailed instructions")
    else:
        print("⚠️  Some components are missing. Please check the errors above.")
        print("\n💡 Tip: Make sure you're running this from the project root directory")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
