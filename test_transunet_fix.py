"""
Quick test script for TransUNet after fix
"""

import sys
sys.path.append('src')

import torch
from models.transunet import TransUNet

print("="*60)
print("Testing Fixed TransUNet")
print("="*60)

# Create model
model = TransUNet(
    in_channels=3,
    out_channels=1,
    img_size=256,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# Model info
info = model.get_model_info()
print(f"\nModel: {info['model_name']}")
print(f"Total parameters: {info['total_params']:,}")
print(f"Trainable parameters: {info['trainable_params']:,}")

# Test forward pass
x = torch.randn(2, 3, 256, 256)
print(f"\nInput shape: {x.shape}")

try:
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("\n" + "="*60)
    print("✅ TransUNet test PASSED!")
    print("="*60)
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("="*60)
    raise
