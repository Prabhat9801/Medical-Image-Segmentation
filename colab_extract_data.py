"""
Complete data extraction script for Colab - FIXED VERSION
Run this in Colab to extract and fix the ISIC dataset
"""

import zipfile
import os
import shutil
import pandas as pd
import glob

# Paths
zip_path = "/content/drive/MyDrive/isic_processed_256.zip"
final_path = "/content/Medical-Image-Segmentation/data/processed/isic"

print("📦 Extracting data with Windows path fix...")

# Remove existing if present
if os.path.exists(final_path):
    shutil.rmtree(final_path)

# Create the directory structure
os.makedirs(final_path, exist_ok=True)
os.makedirs(os.path.join(final_path, "images"), exist_ok=True)
os.makedirs(os.path.join(final_path, "masks"), exist_ok=True)

# Extract and fix Windows paths
with zipfile.ZipFile(zip_path, 'r') as z:
    for file_info in z.filelist:
        orig_name = file_info.filename
        fixed_name = orig_name.replace('\\', '/')
        
        if fixed_name.startswith('isic/'):
            fixed_name = fixed_name[5:]
        
        if 'images/' in fixed_name or (fixed_name.endswith('.png') and 'mask' not in fixed_name.lower()):
            filename = os.path.basename(fixed_name)
            target_path = os.path.join(final_path, "images", filename)
        elif 'masks/' in fixed_name or '_mask.png' in fixed_name:
            filename = os.path.basename(fixed_name)
            target_path = os.path.join(final_path, "masks", filename)
        elif fixed_name.endswith('splits.csv'):
            target_path = os.path.join(final_path, "splits.csv")
        else:
            continue
        
        with z.open(file_info) as source, open(target_path, 'wb') as target:
            shutil.copyfileobj(source, target)

print("✅ Extraction complete!")

# Fix splits.csv paths - CRITICAL FIX
print("\n🔧 Fixing splits.csv paths...")
splits_path = os.path.join(final_path, "splits.csv")

if os.path.exists(splits_path):
    df = pd.read_csv(splits_path)
    
    print(f"Before fix - sample path: {df.iloc[0]['image_path']}")
    
    # CRITICAL: Replace backslashes BEFORE extracting basename
    df['image_path'] = df['image_path'].apply(
        lambda x: os.path.join(final_path, "images", os.path.basename(x.replace('\\', '/')))
    )
    df['mask_path'] = df['mask_path'].apply(
        lambda x: os.path.join(final_path, "masks", os.path.basename(x.replace('\\', '/')))
    )
    
    print(f"After fix - sample path: {df.iloc[0]['image_path']}")
    
    # Save the fixed CSV
    df.to_csv(splits_path, index=False)
    print("✅ splits.csv paths fixed!")
    
    # Verify files exist
    sample_img = df.iloc[0]['image_path']
    sample_mask = df.iloc[0]['mask_path']
    
    if os.path.exists(sample_img) and os.path.exists(sample_mask):
        print("✅ Sample files verified - paths are correct!")
    else:
        print(f"❌ WARNING: Sample files not found!")
        print(f"   Image: {sample_img} - exists: {os.path.exists(sample_img)}")
        print(f"   Mask: {sample_mask} - exists: {os.path.exists(sample_mask)}")
else:
    print("❌ splits.csv not found!")

# Final verification
image_count = len(glob.glob(os.path.join(final_path, "images", "*.png")))
mask_count = len(glob.glob(os.path.join(final_path, "masks", "*.png")))

print(f"\n{'='*60}")
print("📊 VERIFICATION")
print(f"{'='*60}")
print(f"✅ Images extracted: {image_count}")
print(f"✅ Masks extracted: {mask_count}")
print(f"✅ splits.csv: {'Found' if os.path.exists(splits_path) else 'NOT FOUND'}")
print(f"\n🎉 Data extraction successful!")
