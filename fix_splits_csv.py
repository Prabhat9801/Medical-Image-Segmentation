# Fix splits.csv paths - IMPROVED VERSION
import pandas as pd
import os

splits_path = "/content/Medical-Image-Segmentation/data/processed/isic/splits.csv"
final_path = "/content/Medical-Image-Segmentation/data/processed/isic"

print("🔧 Fixing splits.csv paths...")

if os.path.exists(splits_path):
    df = pd.read_csv(splits_path)
    
    print(f"Original path example: {df.iloc[0]['image_path']}")
    
    # Fix paths - replace ALL backslashes first, then extract filename
    df['image_path'] = df['image_path'].apply(
        lambda x: os.path.join(final_path, "images", os.path.basename(x.replace('\\', '/')))
    )
    df['mask_path'] = df['mask_path'].apply(
        lambda x: os.path.join(final_path, "masks", os.path.basename(x.replace('\\', '/')))
    )
    
    print(f"Fixed path example: {df.iloc[0]['image_path']}")
    
    # Verify files exist
    sample_img = df.iloc[0]['image_path']
    sample_mask = df.iloc[0]['mask_path']
    print(f"\n✅ Sample image exists: {os.path.exists(sample_img)}")
    print(f"✅ Sample mask exists: {os.path.exists(sample_mask)}")
    
    # Save
    df.to_csv(splits_path, index=False)
    print(f"\n✅ splits.csv fixed with {len(df)} rows!")
else:
    print("❌ splits.csv not found!")
