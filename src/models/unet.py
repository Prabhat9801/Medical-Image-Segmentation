"""
UNet implementation for medical image segmentation.
Based on: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv -> BN -> ReLU) x 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: Upsample -> Concat -> DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Upsampled feature map from decoder
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch (if any)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution: 1x1 conv to get final segmentation map
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture for binary segmentation.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (1 for binary segmentation)
        features: Base number of features (default: 64)
        bilinear: Use bilinear upsampling instead of transposed conv
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: int = 64,
        bilinear: bool = True
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)
        
        # Decoder
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)
        
        # Output
        self.outc = OutConv(features, out_channels)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output segmentation map (B, 1, H, W) - logits
        """
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        
        return logits
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'UNet',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'bilinear': self.bilinear
        }


def test_unet():
    """Test UNet forward pass."""
    print("Testing UNet...")
    
    # Create model
    model = UNet(in_channels=3, out_channels=1, features=64)
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel: {info['model_name']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test with different input sizes
    print("\nTesting different input sizes:")
    for size in [128, 256, 512]:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            output = model(x)
        print(f"  Input: {x.shape} -> Output: {output.shape}")
    
    print("\n✓ UNet test passed!")


if __name__ == '__main__':
    test_unet()
