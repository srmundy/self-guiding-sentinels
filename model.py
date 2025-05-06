import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        """
        x: (B, C, H, W) - batch of images
        returns: (B, n_patches, embed_dim) - patch embeddings
        """
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class PositionEmbedding(nn.Module):
    """
    Adds positional embeddings to patch embeddings.
    """
    def __init__(self, n_patches, embed_dim=768):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        x: (B, n_patches, embed_dim) - patch embeddings
        returns: (B, n_patches+1, embed_dim) - patch embeddings with position and cls token
        """
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches+1, embed_dim)
        x = x + self.pos_embed  # Add positional embedding
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (B, N, C) where B is batch size, N is sequence length, C is embed_dim
        """
        B, N, C = x.shape
        
        # Get query, key, value projections
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, n_heads, N, head_dim)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, n_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention weights to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class MLP(nn.Module):
    """
    MLP module with 2 layers.
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
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
    Transformer encoder block.
    """
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer model as backbone.
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        embed_dim=768,
        depth=12, 
        n_heads=12, 
        mlp_ratio=4.0, 
        dropout=0.1
    ):
        super().__init__()
        
        # Patch and position embeddings
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.pos_embed = PositionEmbedding(n_patches, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Forward pass.
        x: (B, C, H, W) - batch of images
        returns: (B, embed_dim) - features
        """
        # Create patch embeddings
        x = self.patch_embed(x)
        
        # Add position embeddings and class token
        x = self.pos_embed(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Use the class token as the output
        return x[:, 0]


class DiffusionComponent(nn.Module):
    """
    Lightweight diffusion component for modeling uncertainty.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768, dropout=0.1):
        super().__init__()
        
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        x: (B, input_dim) - features from the Vision Transformer
        returns: (B, output_dim) - enhanced features
        """
        trajectory_features = self.trajectory_predictor(x)
        uncertainty_features = self.uncertainty_estimator(x)
        
        # Concatenate the features
        combined_features = torch.cat([trajectory_features, uncertainty_features], dim=1)
        
        # Fuse the features
        enhanced_features = self.feature_fusion(combined_features)
        
        return enhanced_features


class SelfGuidingSentinels(nn.Module):
    """
    Complete Self-Guiding Sentinels model for tampering detection.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        vit_depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        diffusion_hidden_dim=512,
        n_classes=4,
        dropout=0.1
    ):
        super().__init__()
        
        # Vision Transformer backbone
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=vit_depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Diffusion component
        self.diffusion = DiffusionComponent(
            input_dim=embed_dim,
            hidden_dim=diffusion_hidden_dim,
            output_dim=embed_dim,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        """
        x: (B, C, H, W) - batch of images
        returns: (B, n_classes) - class logits
        """
        # Extract features using Vision Transformer
        vit_features = self.vit(x)
        
        # Enhance features using Diffusion component
        enhanced_features = self.diffusion(vit_features)
        
        # Classify using the enhanced features
        logits = self.classifier(enhanced_features)
        
        return logits
    
    def predict(self, x):
        """
        Predict the class probabilities.
        x: (B, C, H, W) - batch of images
        returns: (B, n_classes) - class probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# Example usage
def main():
    # Create a model instance
    model = SelfGuidingSentinels(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        vit_depth=12,
        n_heads=12,
        diffusion_hidden_dim=512,
        n_classes=4,
        dropout=0.1
    )
    
    # Create a random batch of images
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    logits = model(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output classes: {logits.argmax(dim=1)}")
    
    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")


if __name__ == "__main__":
    main()
