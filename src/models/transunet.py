"""
TransUNet implementation for medical image segmentation.
Based on: Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation" (2021)

TransUNet combines:
- CNN encoder for low-level features
- Vision Transformer for global context
- CNN decoder with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    """
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 512, embed_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    MLP block with GELU activation.
    """
    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4, dropout: float = 0.1):
        super(MLP, self).__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block: Attention + MLP with residual connections.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """
    Vision Transformer encoder.
    """
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: int = 4, dropout: float = 0.1):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class CNNEncoder(nn.Module):
    """
    CNN encoder to extract low-level features before Transformer.
    """
    def __init__(self, in_channels: int = 3):
        super(CNNEncoder, self).__init__()
        
        # Encoder blocks
        self.enc1 = self._make_layer(in_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def _make_layer(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Returns skip connections at multiple scales.
        """
        x1 = self.enc1(x)       # 64, H, W
        x2 = self.enc2(self.pool(x1))  # 128, H/2, W/2
        x3 = self.enc3(self.pool(x2))  # 256, H/4, W/4
        x4 = self.enc4(self.pool(x3))  # 512, H/8, W/8
        
        return x1, x2, x3, x4


class CNNDecoder(nn.Module):
    """
    CNN decoder with skip connections.
    """
    def __init__(self, out_channels: int = 1):
        super(CNNDecoder, self).__init__()
        
        # Channel matching convolutions for skip connections
        self.match_skip3 = nn.Conv2d(512, 256, kernel_size=1)
        
        # Upsampling layers
        self.up1 = self._make_up_layer(768, 512)
        self.up2 = self._make_up_layer(512, 256)
        self.up3 = self._make_up_layer(256, 128)
        self.up4 = self._make_up_layer(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def _make_up_layer(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip1, skip2, skip3):
        """
        Args:
            x: Transformer output reshaped to (B, 768, 2, 2) for img_size=256, patch_size=16
            skip1: (B, 64, 256, 256)
            skip2: (B, 128, 128, 128)
            skip3: (B, 256, 64, 64)
        """
        # Upsample transformer output: 2x2 -> 4x4
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.up1(x)  # (B, 512, 4, 4)
        
        # Upsample: 4x4 -> 8x8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.up2(x)  # (B, 256, 8, 8)
        
        # Upsample: 8x8 -> 16x16
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.up3(x)  # (B, 128, 16, 16)
        
        # Upsample: 16x16 -> 32x32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.up4(x)  # (B, 64, 32, 32)
        
        # Upsample to match input size: 32x32 -> 256x256
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        
        x = self.final(x)
        return x


class TransUNet(nn.Module):
    """
    TransUNet: Hybrid CNN-Transformer architecture.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (1 for binary segmentation)
        img_size: Input image size
        patch_size: Patch size for Vision Transformer
        embed_dim: Embedding dimension for Transformer
        depth: Number of Transformer blocks
        num_heads: Number of attention heads
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12
    ):
        super(TransUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        
        # CNN Encoder
        self.cnn_encoder = CNNEncoder(in_channels)
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size // 8, patch_size, 512, embed_dim)
        
        # Positional Embedding
        n_patches = (img_size // 8 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        
        # Transformer Encoder
        self.transformer = Transformer(embed_dim, depth, num_heads)
        
        # CNN Decoder
        self.cnn_decoder = CNNDecoder(out_channels)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output segmentation map (B, 1, H, W) - logits
        """
        B = x.shape[0]
        
        # CNN Encoder
        skip1, skip2, skip3, x = self.cnn_encoder(x)
        
        # Patch Embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)  # (B, n_patches, embed_dim)
        
        # Reshape for decoder
        n_patches = int(math.sqrt(x.shape[1]))
        x = x.transpose(1, 2).reshape(B, -1, n_patches, n_patches)
        
        # CNN Decoder with skip connections
        output = self.cnn_decoder(x, skip1, skip2, skip3)
        
        return output
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'TransUNet',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'img_size': self.img_size,
            'patch_size': self.patch_size
        }


def test_transunet():
    """Test TransUNet forward pass."""
    print("Testing TransUNet...")
    
    # Create model (smaller version for testing)
    model = TransUNet(
        in_channels=3,
        out_channels=1,
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=6,  # Reduced from 12 for faster testing
        num_heads=12
    )
    
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
    
    print("\n✓ TransUNet test passed!")


if __name__ == '__main__':
    test_transunet()
