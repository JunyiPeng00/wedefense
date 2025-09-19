#!/usr/bin/env python3
"""Standalone test script for LSQ quantization implementation."""

import sys
import os
import math
import unittest
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wedefense'))

from wedefense.frontend.wav2vec2.lsq_quantizer import LSQQuantizer, LSQQuantizerFunction, create_lsq_quantizer
from wedefense.frontend.wav2vec2.quantized_layers import QuantizedLinear, QuantizedConv1d
from wedefense.frontend.wav2vec2.quantization_utils import (
    QuantizationConfig,
    quantize_model,
    get_quantization_stats,
    compare_model_sizes,
    get_quantization_config
)
from wedefense.frontend.wav2vec2.quantization_training import QuantizationAwareTrainer


class TestLSQQuantizerFunction(unittest.TestCase):
    """Test cases for LSQQuantizerFunction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.input_size = 100
    
    def test_forward_pass_8bit_symmetric(self):
        """Test forward pass with 8-bit symmetric quantization."""
        x = torch.randn(self.batch_size, self.input_size, device=self.device)
        step_size = torch.tensor(0.1, device=self.device, requires_grad=True)
        
        quantized = LSQQuantizerFunction.apply(x, step_size, 8, True)
        
        # Check output shape
        self.assertEqual(quantized.shape, x.shape)
        
        # Check quantization range for 8-bit symmetric
        q_min = -(2 ** 7)
        q_max = 2 ** 7 - 1
        scaled = quantized / step_size
        self.assertTrue(torch.all(scaled >= q_min))
        self.assertTrue(torch.all(scaled <= q_max))
    
    def test_forward_pass_4bit_asymmetric(self):
        """Test forward pass with 4-bit asymmetric quantization."""
        x = torch.randn(self.batch_size, self.input_size, device=self.device)
        step_size = torch.tensor(0.05, device=self.device, requires_grad=True)
        
        quantized = LSQQuantizerFunction.apply(x, step_size, 4, False)
        
        # Check quantization range for 4-bit asymmetric
        q_min = 0
        q_max = 2 ** 4 - 1
        scaled = quantized / step_size
        self.assertTrue(torch.all(scaled >= q_min))
        self.assertTrue(torch.all(scaled <= q_max))
    
    def test_gradient_flow(self):
        """Test gradient flow through quantization."""
        x = torch.randn(self.batch_size, self.input_size, device=self.device, requires_grad=True)
        step_size = torch.tensor(0.1, device=self.device, requires_grad=True)
        
        quantized = LSQQuantizerFunction.apply(x, step_size, 8, True)
        loss = quantized.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(step_size.grad)
        
        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(step_size.grad.shape, step_size.shape)


class TestLSQQuantizer(unittest.TestCase):
    """Test cases for LSQQuantizer module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.input_size = 100
    
    def test_quantizer_initialization(self):
        """Test quantizer initialization."""
        quantizer = LSQQuantizer(num_bits=8, symmetric=True)
        
        self.assertEqual(quantizer.num_bits, 8)
        self.assertTrue(quantizer.symmetric)
        self.assertFalse(quantizer.per_channel)
        self.assertIsNotNone(quantizer.step_size)
    
    def test_forward_pass(self):
        """Test quantizer forward pass."""
        quantizer = LSQQuantizer(num_bits=8, symmetric=True).to(self.device)
        x = torch.randn(self.batch_size, self.input_size, device=self.device)
        
        quantized = quantizer(x)
        
        self.assertEqual(quantized.shape, x.shape)
        self.assertFalse(torch.allclose(x, quantized))  # Should be different due to quantization
    
    def test_per_channel_quantization(self):
        """Test per-channel quantization."""
        quantizer = LSQQuantizer(
            num_bits=8, 
            symmetric=True, 
            per_channel=True, 
            channel_axis=0
        ).to(self.device)
        
        x = torch.randn(10, 20, device=self.device)  # 10 channels, 20 features
        
        quantized = quantizer(x)
        
        self.assertEqual(quantized.shape, x.shape)
        self.assertIsNotNone(quantizer.step_size)
        self.assertEqual(quantizer.step_size.shape[0], 10)  # One step size per channel
    
    def test_quantization_error(self):
        """Test quantization error calculation."""
        quantizer = LSQQuantizer(num_bits=8, symmetric=True).to(self.device)
        x = torch.randn(self.batch_size, self.input_size, device=self.device)
        
        error = quantizer.get_quantization_error(x)
        
        self.assertIsInstance(error, torch.Tensor)
        self.assertGreaterEqual(error.item(), 0)  # Error should be non-negative
    
    def test_effective_bits(self):
        """Test effective bits calculation."""
        quantizer = LSQQuantizer(num_bits=8, symmetric=True).to(self.device)
        x = torch.randn(self.batch_size, self.input_size, device=self.device)
        
        effective_bits = quantizer.get_effective_bits(x)
        
        self.assertIsInstance(effective_bits, float)
        self.assertGreaterEqual(effective_bits, 0)
        self.assertLessEqual(effective_bits, 8)


class TestQuantizedLayers(unittest.TestCase):
    """Test cases for quantized layer implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.input_size = 100
        self.output_size = 50
    
    def test_quantized_linear(self):
        """Test QuantizedLinear layer."""
        layer = QuantizedLinear(
            self.input_size, 
            self.output_size, 
            quantize_weights=True,
            quantize_activations=False
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.input_size, device=self.device)
        output = layer(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        self.assertIsNotNone(layer.weight_quantizer)
        self.assertIsNone(layer.activation_quantizer)
    
    def test_quantized_conv1d(self):
        """Test QuantizedConv1d layer."""
        layer = QuantizedConv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            quantize_weights=True,
            quantize_activations=False
        ).to(self.device)
        
        x = torch.randn(self.batch_size, 16, 100, device=self.device)
        output = layer(x)
        
        self.assertEqual(output.shape, (self.batch_size, 32, 98))  # 100 - 3 + 1
        self.assertIsNotNone(layer.weight_quantizer)
        self.assertIsNone(layer.activation_quantizer)
    
    def test_quantization_stats(self):
        """Test quantization statistics collection."""
        layer = QuantizedLinear(
            self.input_size, 
            self.output_size, 
            quantize_weights=True
        ).to(self.device)
        
        stats = layer.get_quantization_stats()
        
        self.assertIn('weight_quantization_error', stats)
        self.assertIn('weight_effective_bits', stats)
        self.assertIsInstance(stats['weight_quantization_error'], float)
        self.assertIsInstance(stats['weight_effective_bits'], float)


class TestQuantizationUtils(unittest.TestCase):
    """Test cases for quantization utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 50)
                self.linear2 = nn.Linear(50, 10)
                self.conv1d = nn.Conv1d(16, 32, 3)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.linear2(x)
                return x
        
        self.model = TestModel().to(self.device)
        self.config = QuantizationConfig(
            weight_bits=8,
            activation_bits=8,
            quantize_weights=True,
            quantize_activations=False
        )
    
    def test_model_quantization(self):
        """Test model quantization."""
        quantized_model = quantize_model(self.model, self.config)
        
        # Check that linear layers are converted
        self.assertIsInstance(quantized_model.linear1, QuantizedLinear)
        self.assertIsInstance(quantized_model.linear2, QuantizedLinear)
        
        # Check that conv1d layer is converted
        self.assertIsInstance(quantized_model.conv1d, QuantizedConv1d)
    
    def test_quantization_stats(self):
        """Test quantization statistics collection."""
        quantized_model = quantize_model(self.model, self.config)
        stats = get_quantization_stats(quantized_model)
        
        self.assertIn('total_parameters', stats)
        self.assertIn('quantized_parameters', stats)
        self.assertIn('quantization_errors', stats)
        self.assertIn('effective_bits', stats)
        self.assertIn('quantized_modules', stats)
    
    def test_model_size_comparison(self):
        """Test model size comparison."""
        quantized_model = quantize_model(self.model, self.config)
        comparison = compare_model_sizes(self.model, quantized_model)
        
        self.assertIn('original_size_mb', comparison)
        self.assertIn('quantized_size_mb', comparison)
        self.assertIn('compression_ratio', comparison)
        self.assertIn('size_reduction_percent', comparison)
        
        # Quantized model should be smaller
        self.assertGreater(comparison['compression_ratio'], 1.0)
        self.assertGreater(comparison['size_reduction_percent'], 0.0)
    
    def test_predefined_configs(self):
        """Test predefined quantization configurations."""
        configs = ['8bit_symmetric', '8bit_asymmetric', '4bit_symmetric', 'mixed_precision', 'weights_only']
        
        for config_name in configs:
            config = get_quantization_config(config_name)
            self.assertIsInstance(config, QuantizationConfig)
            self.assertIn('weight_bits', config.to_dict())


def run_performance_benchmark():
    """Run performance benchmark for quantization."""
    print("Running LSQ Quantization Performance Benchmark...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a larger model for benchmarking
    class BenchmarkModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(512, 512) for _ in range(5)
            ])
            self.output = nn.Linear(512, 10)
        
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.output(x)
    
    model = BenchmarkModel().to(device)
    config = get_quantization_config('8bit_symmetric')
    
    # Benchmark original model
    x = torch.randn(1000, 512, device=device)
    
    # Warm up
    for _ in range(10):
        _ = model(x)
    
    # Time original model
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    if device.type == 'cuda':
        start_time.record()
        for _ in range(100):
            _ = model(x)
        end_time.record()
        torch.cuda.synchronize()
        original_time = start_time.elapsed_time(end_time) / 100
    else:
        import time
        start = time.time()
        for _ in range(100):
            _ = model(x)
        end = time.time()
        original_time = (end - start) * 10  # Convert to ms
    
    # Benchmark quantized model
    quantized_model = quantize_model(model, config)
    
    # Warm up
    for _ in range(10):
        _ = quantized_model(x)
    
    # Time quantized model
    if device.type == 'cuda':
        start_time.record()
        for _ in range(100):
            _ = quantized_model(x)
        end_time.record()
        torch.cuda.synchronize()
        quantized_time = start_time.elapsed_time(end_time) / 100
    else:
        start = time.time()
        for _ in range(100):
            _ = quantized_model(x)
        end = time.time()
        quantized_time = (end - start) * 10  # Convert to ms
    
    # Calculate compression ratio
    original_size = sum(p.numel() for p in model.parameters()) * 4  # 4 bytes per float32
    quantized_size = sum(p.numel() for p in quantized_model.parameters()) * 1  # 1 byte per int8
    
    print(f"Original model time: {original_time:.2f} ms")
    print(f"Quantized model time: {quantized_time:.2f} ms")
    print(f"Speedup: {original_time / quantized_time:.2f}x")
    print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")


def test_basic_functionality():
    """Test basic LSQ functionality without unittest."""
    print("Testing basic LSQ functionality...")
    
    # Test LSQQuantizer
    quantizer = LSQQuantizer(num_bits=8, symmetric=True)
    x = torch.randn(32, 100)
    quantized = quantizer(x)
    
    print(f"✓ LSQQuantizer forward pass: {x.shape} -> {quantized.shape}")
    
    # Test QuantizedLinear
    layer = QuantizedLinear(100, 50, quantize_weights=True)
    output = layer(x)
    
    print(f"✓ QuantizedLinear forward pass: {x.shape} -> {output.shape}")
    
    # Test model quantization
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(100, 50)
            self.linear2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    config = get_quantization_config('8bit_symmetric')
    quantized_model = quantize_model(model, config)
    
    print(f"✓ Model quantization: {type(model.linear1)} -> {type(quantized_model.linear1)}")
    
    # Test size comparison
    size_comp = compare_model_sizes(model, quantized_model)
    print(f"✓ Model compression: {size_comp['compression_ratio']:.2f}x reduction")
    
    print("All basic tests passed!")


if __name__ == "__main__":
    print("LSQ Quantization Test Suite")
    print("=" * 50)
    
    # Run basic functionality test first
    test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("Running Unit Tests...")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    print("\n" + "=" * 50)
    run_performance_benchmark()
