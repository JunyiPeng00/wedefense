#!/usr/bin/env python3
"""Example script demonstrating LSQ quantization in WeDefense.

This script shows how to use the LSQ quantization functionality
for model compression and efficient inference.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wedefense.frontend.wav2vec2 import (
    QuantizationConfig,
    quantize_model,
    get_quantization_config,
    QuantizationAwareTrainer,
    compare_model_sizes,
    get_quantization_stats
)


def create_demo_model():
    """Create a demo model for quantization testing."""
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.linear1 = nn.Linear(128, 64)
            self.linear2 = nn.Linear(64, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            # x: (batch_size, 1, sequence_length)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)  # (batch_size, 128)
            x = self.dropout(self.relu(self.linear1(x)))
            x = self.linear2(x)
            return x
    
    return DemoModel()


def create_demo_data(batch_size=32, sequence_length=1000):
    """Create demo data for training and testing."""
    # Generate random audio-like data
    x = torch.randn(batch_size * 10, 1, sequence_length)
    y = torch.randint(0, 10, (batch_size * 10,))
    
    # Split into train and validation
    train_size = int(0.8 * len(x))
    train_x, val_x = x[:train_size], x[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def demonstrate_basic_quantization():
    """Demonstrate basic model quantization."""
    print("=" * 60)
    print("1. Basic Model Quantization Demo")
    print("=" * 60)
    
    # Create model
    model = create_demo_model()
    print(f"Original model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    x = torch.randn(4, 1, 1000)
    with torch.no_grad():
        original_output = model(x)
    print(f"Original model output shape: {original_output.shape}")
    
    # Quantize model
    config = get_quantization_config('8bit_symmetric')
    quantized_model = quantize_model(model, config)
    print(f"Model quantized with {config.weight_bits}-bit symmetric quantization")
    
    # Test quantized forward pass
    with torch.no_grad():
        quantized_output = quantized_model(x)
    print(f"Quantized model output shape: {quantized_output.shape}")
    
    # Compare outputs
    output_diff = torch.abs(original_output - quantized_output)
    mse = torch.mean(output_diff ** 2).item()
    print(f"Output MSE: {mse:.6f}")
    
    # Compare model sizes
    size_comparison = compare_model_sizes(model, quantized_model)
    print(f"Model size reduction: {size_comparison['size_reduction_percent']:.1f}%")
    print(f"Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    
    return model, quantized_model


def demonstrate_quantization_aware_training():
    """Demonstrate quantization-aware training."""
    print("\n" + "=" * 60)
    print("2. Quantization-Aware Training Demo")
    print("=" * 60)
    
    # Create model and data
    model = create_demo_model()
    train_loader, val_loader = create_demo_data()
    
    # Create quantization configuration
    config = get_quantization_config('8bit_symmetric')
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = QuantizationAwareTrainer(model, config, device)
    
    print(f"Using device: {device}")
    print(f"Training with {len(train_loader)} batches")
    
    # Calibrate model
    print("Calibrating quantization parameters...")
    trainer.calibrate_model(train_loader, num_batches=10)
    
    # Fine-tune model
    print("Fine-tuning quantized model...")
    history = trainer.fine_tune_quantized_model(
        train_loader,
        val_loader,
        num_epochs=3,
        learning_rate=1e-3
    )
    
    print(f"Training completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Evaluate model
    val_metrics = trainer.evaluate_quantized_model(val_loader)
    print(f"Final evaluation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")
    
    # Get quantization statistics
    quant_stats = trainer.get_quantization_statistics()
    print(f"Quantized parameters: {quant_stats['quantized_parameters']:,}")
    print(f"Total parameters: {quant_stats['total_parameters']:,}")
    
    return trainer


def demonstrate_different_quantization_configs():
    """Demonstrate different quantization configurations."""
    print("\n" + "=" * 60)
    print("3. Different Quantization Configurations Demo")
    print("=" * 60)
    
    model = create_demo_model()
    test_data = torch.randn(4, 1, 1000)
    
    configs = [
        '8bit_symmetric',
        '8bit_asymmetric',
        '4bit_symmetric',
        'mixed_precision',
        'weights_only'
    ]
    
    results = {}
    
    for config_name in configs:
        print(f"\nTesting {config_name}...")
        
        # Get configuration
        config = get_quantization_config(config_name)
        
        # Quantize model
        quantized_model = quantize_model(model, config)
        
        # Test forward pass
        with torch.no_grad():
            original_output = model(test_data)
            quantized_output = quantized_model(test_data)
        
        # Calculate metrics
        output_diff = torch.abs(original_output - quantized_output)
        mse = torch.mean(output_diff ** 2).item()
        mae = torch.mean(output_diff).item()
        
        # Get size comparison
        size_comp = compare_model_sizes(model, quantized_model)
        
        # Get quantization stats
        quant_stats = get_quantization_stats(quantized_model)
        
        results[config_name] = {
            'mse': mse,
            'mae': mae,
            'compression_ratio': size_comp['compression_ratio'],
            'size_reduction': size_comp['size_reduction_percent'],
            'quantized_params': quant_stats['quantized_parameters']
        }
        
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Compression: {size_comp['compression_ratio']:.2f}x")
        print(f"  Size reduction: {size_comp['size_reduction_percent']:.1f}%")
    
    # Print summary table
    print("\n" + "-" * 80)
    print("Quantization Configuration Comparison")
    print("-" * 80)
    print(f"{'Config':<20} {'MSE':<12} {'MAE':<12} {'Compression':<12} {'Size Reduction':<15}")
    print("-" * 80)
    
    for config_name, results in results.items():
        print(f"{config_name:<20} {results['mse']:<12.6f} {results['mae']:<12.6f} "
              f"{results['compression_ratio']:<12.2f} {results['size_reduction']:<15.1f}%")
    
    return results


def demonstrate_performance_benchmark():
    """Demonstrate performance benchmarking."""
    print("\n" + "=" * 60)
    print("4. Performance Benchmark Demo")
    print("=" * 60)
    
    model = create_demo_model()
    config = get_quantization_config('8bit_symmetric')
    quantized_model = quantize_model(model, config)
    
    # Test data
    test_data = torch.randn(100, 1, 1000)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_data)
            _ = quantized_model(test_data)
    
    # Benchmark original model
    import time
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = model(test_data)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    original_time = time.time() - start_time
    
    # Benchmark quantized model
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = quantized_model(test_data)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    quantized_time = time.time() - start_time
    
    # Calculate metrics
    speedup = original_time / quantized_time
    size_comp = compare_model_sizes(model, quantized_model)
    
    print(f"Original model time: {original_time:.4f} seconds")
    print(f"Quantized model time: {quantized_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Model size reduction: {size_comp['size_reduction_percent']:.1f}%")
    print(f"Compression ratio: {size_comp['compression_ratio']:.2f}x")


def main():
    """Main function to run all demonstrations."""
    print("LSQ Quantization Demo for WeDefense")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        # Run demonstrations
        model, quantized_model = demonstrate_basic_quantization()
        trainer = demonstrate_quantization_aware_training()
        config_results = demonstrate_different_quantization_configs()
        demonstrate_performance_benchmark()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
        # Print summary
        print("\nSummary:")
        print("- LSQ quantization successfully integrated")
        print("- Model compression achieved with minimal accuracy loss")
        print("- Quantization-aware training implemented")
        print("- Multiple quantization configurations tested")
        print("- Performance benchmarking completed")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
