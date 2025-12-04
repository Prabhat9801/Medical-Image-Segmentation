# ============================================================
# UPDATED TransUNet Test Cell - Use This Instead
# ============================================================

# Force reload to get the fixed version
import sys
import importlib

# Clear cached modules
modules_to_clear = ['models.transunet', 'models.unet', 'models.unetpp', 'models']
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]

# Re-import
sys.path.insert(0, '../src')
from models.transunet import TransUNet
import torch

print("Testing TransUNet (FIXED VERSION)...\n")

# Create model
transunet = TransUNet(
    in_channels=3,
    out_channels=1,
    img_size=256,
    patch_size=16,
    embed_dim=768,
    depth=6,  # Reduced for testing
    num_heads=12
)

# Model info
info = transunet.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Total parameters: {info['total_params']:,}")
print(f"Trainable parameters: {info['trainable_params']:,}")

# Test forward pass
x = torch.randn(2, 3, 256, 256)
print(f"\nInput shape: {x.shape}")

try:
    with torch.no_grad():
        output = transunet(x)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("\n✅ TransUNet test PASSED!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n⚠️  If you still see the error, please:")
    print("   1. Click 'Kernel' → 'Restart Kernel'")
    print("   2. Re-run all cells from the beginning")
    raise
