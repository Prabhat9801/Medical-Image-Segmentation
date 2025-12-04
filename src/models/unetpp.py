"""
UNet++ implementation for medical image segmentation.
Based on: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018)

UNet++ features:
- Nested skip pathways
- Dense skip connections
- Deep supervision (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolution block: Conv -> BN -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    UNet++ architecture with nested skip pathways.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (1 for binary segmentation)
        features: Base number of features (default: 32)
        deep_supervision: Use deep supervision for training
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: int = 32,
        deep_supervision: bool = False
    ):
        super(UNetPlusPlus, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        
        # Feature channels at each level
        nb_filter = [features, features*2, features*4, features*8, features*16]
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Encoder (column 0)
        self.conv0_0 = self._make_layer(in_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(nb_filter[0], nb_filter[1])
        self.conv2_0 = self._make_layer(nb_filter[1], nb_filter[2])
        self.conv3_0 = self._make_layer(nb_filter[2], nb_filter[3])
        self.conv4_0 = self._make_layer(nb_filter[3], nb_filter[4])
        
        # Nested skip pathways
        # Column 1
        self.conv0_1 = self._make_layer(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._make_layer(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = self._make_layer(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = self._make_layer(nb_filter[3] + nb_filter[4], nb_filter[3])
        
        # Column 2
        self.conv0_2 = self._make_layer(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = self._make_layer(nb_filter[2]*2 + nb_filter[3], nb_filter[2])
        
        # Column 3
        self.conv0_3 = self._make_layer(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(nb_filter[1]*3 + nb_filter[2], nb_filter[1])
        
        # Column 4
        self.conv0_4 = self._make_layer(nb_filter[0]*4 + nb_filter[1], nb_filter[0])
        
        # Output layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
    
    def _make_layer(self, in_channels: int, out_channels: int):
        """Create a double convolution layer."""
        return nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output segmentation map (B, 1, H, W) - logits
            If deep_supervision=True, returns list of outputs
        """
        # Column 0 (Encoder)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        
        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        
        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        
        # Column 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'UNet++',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'deep_supervision': self.deep_supervision
        }


def test_unetpp():
    """Test UNet++ forward pass."""
    print("Testing UNet++...")
    
    # Test without deep supervision
    print("\n1. Testing without deep supervision:")
    model = UNetPlusPlus(in_channels=3, out_channels=1, features=32, deep_supervision=False)
    
    info = model.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    x = torch.randn(2, 3, 256, 256)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test with deep supervision
    print("\n2. Testing with deep supervision:")
    model_ds = UNetPlusPlus(in_channels=3, out_channels=1, features=32, deep_supervision=True)
    
    with torch.no_grad():
        outputs = model_ds(x)
    
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i+1} shape: {out.shape}")
    
    # Test with different input sizes
    print("\n3. Testing different input sizes:")
    for size in [128, 256, 512]:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            output = model(x)
        print(f"  Input: {x.shape} -> Output: {output.shape}")
    
    print("\n✓ UNet++ test passed!")


if __name__ == '__main__':
    test_unetpp()
