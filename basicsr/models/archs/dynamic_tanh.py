"""
Dynamic Tanh (DyT) Module - Replacement for LayerNorm
Paper: Transformers without Normalization (DyT)
Adapted for HOGformer architecture
"""

import torch
import torch.nn as nn
from einops import rearrange


class DynamicTanh(nn.Module):
    """
    Dynamic Tanh layer as a drop-in replacement for LayerNorm.
    
    Args:
        normalized_shape (int): Number of features (channels)
        alpha_init_value (float): Initial value for alpha parameter (default: 0.5)
                                  Paper recommends 0.5 for vision tasks
    
    Forward:
        DyT(x) = γ * tanh(αx) + β
        where α is learnable scalar, γ and β are learnable vectors
    
    Note: This implementation supports channels_last format used in HOGformer
          Input/Output shape: [B, H*W, C] (converted from [B, C, H, W])
    """
    
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super(DynamicTanh, self).__init__()
        
        # Store parameters
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # γ
        self.bias = nn.Parameter(torch.zeros(normalized_shape))    # β
    
    def forward(self, x):
        """
        Forward pass of DynamicTanh.
        
        Args:
            x: Input tensor of shape [B, H*W, C] (channels_last format)
        
        Returns:
            Output tensor of same shape after DyT transformation
        """
        # Apply tanh with learnable scaling
        x = torch.tanh(self.alpha * x)
        
        # Apply affine transformation (element-wise for channels_last)
        x = x * self.weight + self.bias
        
        return x
    
    def extra_repr(self):
        """String representation for model summary"""
        return (f"normalized_shape={self.normalized_shape}, "
                f"alpha_init_value={self.alpha_init_value}")


class DynamicTanh2d(nn.Module):
    """
    DynamicTanh for channels_first format (if needed for Conv layers).
    Currently not used in HOGformer but provided for completeness.
    
    Input shape: [B, C, H, W]
    """
    
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super(DynamicTanh2d, self).__init__()
        
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        """
        x = torch.tanh(self.alpha * x)
        # Reshape for broadcasting: [C] -> [C, 1, 1]
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x
    
    def extra_repr(self):
        return (f"normalized_shape={self.normalized_shape}, "
                f"alpha_init_value={self.alpha_init_value}")


def convert_ln_to_dyt(module, alpha_init_value=0.5):
    """
    Recursively convert all LayerNorm layers to DynamicTanh.
    
    Args:
        module: PyTorch module to convert
        alpha_init_value: Initial alpha value for DyT layers
    
    Returns:
        Converted module with DyT replacing LayerNorm
    
    Usage:
        model = HOGformer(...)
        model = convert_ln_to_dyt(model, alpha_init_value=0.5)
    """
    module_output = module
    
    # Check if current module is LayerNorm-like
    if isinstance(module, nn.LayerNorm):
        # Replace with DynamicTanh
        module_output = DynamicTanh(
            module.normalized_shape[0],  # Extract channel dimension
            alpha_init_value=alpha_init_value
        )
        print(f"[DyT] Converted LayerNorm({module.normalized_shape}) -> DynamicTanh")
    
    # Recursively convert children modules
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child, alpha_init_value))
    
    del module
    return module_output


# Utility function for testing
def test_dyt_equivalence():
    """
    Test that DyT has similar behavior to LayerNorm (not identical, but stable).
    """
    batch_size, seq_len, dim = 2, 16, 64
    x = torch.randn(batch_size, seq_len, dim)
    
    # LayerNorm
    ln = nn.LayerNorm(dim)
    out_ln = ln(x)
    
    # DynamicTanh
    dyt = DynamicTanh(dim, alpha_init_value=0.5)
    out_dyt = dyt(x)
    
    print("Input shape:", x.shape)
    print("LayerNorm output range:", out_ln.min().item(), "to", out_ln.max().item())
    print("DynamicTanh output range:", out_dyt.min().item(), "to", out_dyt.max().item())
    print("Both outputs have same shape:", out_ln.shape == out_dyt.shape)
    
    # Check gradients flow
    loss_ln = out_ln.sum()
    loss_dyt = out_dyt.sum()
    loss_ln.backward()
    loss_dyt.backward()
    
    print("Gradients computed successfully for both!")


if __name__ == "__main__":
    test_dyt_equivalence()