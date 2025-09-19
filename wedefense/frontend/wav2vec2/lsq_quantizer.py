"""Learnable Step Size Quantization (LSQ) implementation for WeDefense.

This module implements LSQ quantization as described in:
"LSQ: Learned Step Size Quantization" (https://arxiv.org/abs/1902.08153)

The implementation provides:
- LSQQuantizerFunction: Custom autograd function for LSQ quantization
- LSQQuantizer: PyTorch module wrapper for easy integration
- Support for both weight and activation quantization
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSQQuantizerFunction(torch.autograd.Function):
    """Custom autograd function for LSQ quantization.
    
    This function implements the forward and backward passes for LSQ quantization
    with proper gradient computation for both the input tensor and the step size.
    
    Args:
        x: Input tensor to be quantized
        step_size: Learnable step size parameter
        num_bits: Number of bits for quantization (default: 8)
        symmetric: Whether to use symmetric quantization (default: True)
    """
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        step_size: torch.Tensor,
        num_bits: int = 8,
        symmetric: bool = True
    ) -> torch.Tensor:
        """Forward pass for LSQ quantization.
        
        Args:
            ctx: Context for saving tensors for backward pass
            x: Input tensor to quantize
            step_size: Learnable step size parameter
            num_bits: Number of bits for quantization
            symmetric: Whether to use symmetric quantization
            
        Returns:
            Quantized tensor
        """
        # Save tensors for backward pass
        ctx.save_for_backward(x, step_size)
        ctx.num_bits = num_bits
        ctx.symmetric = symmetric
        
        # Calculate quantization parameters
        if symmetric:
            # Symmetric quantization: [-2^(b-1), 2^(b-1)-1]
            q_min = -(2 ** (num_bits - 1))
            q_max = 2 ** (num_bits - 1) - 1
        else:
            # Asymmetric quantization: [0, 2^b-1]
            q_min = 0
            q_max = 2 ** num_bits - 1
        
        # Quantization process
        # 1. Scale by step size
        x_scaled = x / step_size
        
        # 2. Clamp to quantization range
        x_clamped = torch.clamp(x_scaled, q_min, q_max)
        
        # 3. Round to nearest integer
        x_rounded = torch.round(x_clamped)
        
        # 4. Scale back by step size
        x_quantized = x_rounded * step_size
        
        return x_quantized
    
    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """Backward pass for LSQ quantization.
        
        Args:
            ctx: Context containing saved tensors
            grad_output: Gradient from the next layer
            
        Returns:
            Tuple of gradients: (grad_x, grad_step_size, None, None)
        """
        # Retrieve saved tensors
        x, step_size = ctx.saved_tensors
        num_bits = ctx.num_bits
        symmetric = ctx.symmetric
        
        # Calculate quantization parameters
        if symmetric:
            q_min = -(2 ** (num_bits - 1))
            q_max = 2 ** (num_bits - 1) - 1
        else:
            q_min = 0
            q_max = 2 ** num_bits - 1
        
        # Gradient for input x: Straight-Through Estimator (STE)
        # The gradient is 1 where x is in the valid range, 0 otherwise
        x_scaled = x / step_size
        grad_x = grad_output.clone()
        
        # Gradient for step size: LSQ specific formula
        # grad_s = grad_output * (x_rounded - x_scaled)
        x_scaled_clamped = torch.clamp(x_scaled, q_min, q_max)
        x_rounded = torch.round(x_scaled_clamped)
        
        # LSQ gradient formula
        grad_step_size = grad_output * (x_rounded - x_scaled)
        
        # Sum over all dimensions except the step size dimension
        # (assuming step_size is a scalar or has the same shape as x)
        if step_size.numel() == 1:
            grad_step_size = grad_step_size.sum()
        else:
            # If step_size has multiple elements, sum over appropriate dimensions
            # Keep the same shape as step_size
            if step_size.shape != grad_step_size.shape:
                # Sum over dimensions that don't match step_size
                sum_dims = []
                for i in range(grad_step_size.ndim):
                    if i >= len(step_size.shape) or grad_step_size.shape[i] != step_size.shape[i]:
                        sum_dims.append(i)
                if sum_dims:
                    grad_step_size = grad_step_size.sum(dim=tuple(sum_dims))
            
            # Ensure the shape matches step_size
            if grad_step_size.shape != step_size.shape:
                grad_step_size = grad_step_size.view(step_size.shape)
        
        return grad_x, grad_step_size, None, None


class LSQQuantizer(nn.Module):
    """Learnable Step Size Quantizer module.
    
    This module wraps the LSQQuantizerFunction and provides a convenient
    interface for quantization with learnable step sizes.
    
    Args:
        num_bits: Number of bits for quantization (default: 8)
        symmetric: Whether to use symmetric quantization (default: True)
        init_step_size: Initial value for step size (default: None, auto-calculated)
        per_channel: Whether to use per-channel quantization (default: False)
        channel_axis: Axis for per-channel quantization (default: 0)
    """
    
    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        init_step_size: Optional[float] = None,
        per_channel: bool = False,
        channel_axis: int = 0
    ):
        super().__init__()
        
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        
        # Initialize step size
        if init_step_size is None:
            # Default initialization: 2 * mean(|x|) / sqrt(2^b)
            # This is a common heuristic for LSQ initialization
            init_step_size = 1.0 / (2 ** (num_bits - 1))
        
        if per_channel:
            # Per-channel quantization: step_size will be initialized later
            # when we know the shape of the input tensor
            self.step_size = None
            self._init_step_size = init_step_size
        else:
            # Scalar quantization
            self.step_size = nn.Parameter(torch.tensor(init_step_size))
    
    def _initialize_per_channel_step_size(self, x: torch.Tensor) -> None:
        """Initialize per-channel step size based on input tensor shape."""
        if not self.per_channel or self.step_size is not None:
            return
        
        # Calculate the shape for step_size parameter
        step_size_shape = [1] * x.ndim
        step_size_shape[self.channel_axis] = x.shape[self.channel_axis]
        
        # Initialize step size for each channel
        init_value = self._init_step_size
        self.step_size = nn.Parameter(
            torch.full(step_size_shape, init_value, device=x.device, dtype=x.dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LSQ quantization.
        
        Args:
            x: Input tensor to quantize
            
        Returns:
            Quantized tensor
        """
        if self.per_channel and self.step_size is None:
            self._initialize_per_channel_step_size(x)
        
        return LSQQuantizerFunction.apply(
            x, self.step_size, self.num_bits, self.symmetric
        )
    
    def get_quantization_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the quantization error (MSE between original and quantized).
        
        Args:
            x: Input tensor
            
        Returns:
            Mean squared error between original and quantized tensor
        """
        x_quantized = self.forward(x)
        return F.mse_loss(x, x_quantized)
    
    def get_effective_bits(self, x: torch.Tensor) -> float:
        """Calculate the effective number of bits used.
        
        This is useful for monitoring the actual quantization precision.
        
        Args:
            x: Input tensor
            
        Returns:
            Effective number of bits
        """
        if self.symmetric:
            q_min = -(2 ** (self.num_bits - 1))
            q_max = 2 ** (self.num_bits - 1) - 1
        else:
            q_min = 0
            q_max = 2 ** self.num_bits - 1
        
        x_scaled = x / self.step_size
        x_clamped = torch.clamp(x_scaled, q_min, q_max)
        
        # Calculate the actual range used
        actual_min = x_clamped.min().item()
        actual_max = x_clamped.max().item()
        actual_range = actual_max - actual_min
        
        # Calculate effective bits
        if actual_range > 0:
            effective_bits = math.log2(actual_range + 1)
        else:
            effective_bits = 0.0
        
        return min(effective_bits, self.num_bits)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"num_bits={self.num_bits}, symmetric={self.symmetric}, per_channel={self.per_channel}"


def create_lsq_quantizer(
    num_bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_axis: int = 0
) -> LSQQuantizer:
    """Factory function to create LSQ quantizer with common configurations.
    
    Args:
        num_bits: Number of bits for quantization
        symmetric: Whether to use symmetric quantization
        per_channel: Whether to use per-channel quantization
        channel_axis: Axis for per-channel quantization
        
    Returns:
        Configured LSQQuantizer instance
    """
    return LSQQuantizer(
        num_bits=num_bits,
        symmetric=symmetric,
        per_channel=per_channel,
        channel_axis=channel_axis
    )


# Common quantization configurations
LSQ_8BIT_SYMMETRIC = lambda: create_lsq_quantizer(num_bits=8, symmetric=True)
LSQ_8BIT_ASYMMETRIC = lambda: create_lsq_quantizer(num_bits=8, symmetric=False)
LSQ_4BIT_SYMMETRIC = lambda: create_lsq_quantizer(num_bits=4, symmetric=True)
LSQ_4BIT_ASYMMETRIC = lambda: create_lsq_quantizer(num_bits=4, symmetric=False)
