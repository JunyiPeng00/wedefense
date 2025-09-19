#!/usr/bin/env python3
"""Simple WavLM Base Quantization Example for WeDefense.

This script provides a simple, practical example of quantizing WavLM Base models
for anti-spoofing tasks in WeDefense.
"""

import os
import sys
import torch
import torch.nn as nn
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wedefense.frontend.wav2vec2 import (
    wavlm_base,
    quantize_model,
    get_quantization_config,
    compare_model_sizes,
    get_quantization_stats
)


def create_and_quantize_wavlm_base():
    """Create and quantize a WavLM Base model."""
    print("WavLM Base Quantization Example")
    print("=" * 50)
    
    # 1. Create WavLM Base model
    print("\n1. Creating WavLM Base model...")
    model = wavlm_base()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Original model parameters: {total_params:,}")
    
    # 2. Test original model
    print("\n2. Testing original model...")
    model.eval()
    
    # Create test audio (batch_size=2, sequence_length=16000 for 1 second at 16kHz)
    test_audio = torch.randn(2, 16000)
    test_lengths = torch.tensor([16000, 16000])
    
    with torch.no_grad():
        start_time = time.time()
        outputs, output_lengths = model(test_audio, test_lengths)
        original_time = time.time() - start_time
    
    print(f"   Input shape: {test_audio.shape}")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Processing time: {original_time*1000:.2f} ms")
    
    # 3. Quantize model
    print("\n3. Quantizing model...")
    
    # Get 8-bit symmetric quantization config
    config = get_quantization_config('8bit_symmetric')
    print(f"   Using {config.weight_bits}-bit symmetric quantization")
    
    # Quantize the model
    quantized_model = quantize_model(model, config)
    
    # 4. Test quantized model
    print("\n4. Testing quantized model...")
    quantized_model.eval()
    
    with torch.no_grad():
        start_time = time.time()
        quant_outputs, quant_output_lengths = quantized_model(test_audio, test_lengths)
        quantized_time = time.time() - start_time
    
    print(f"   Output shape: {quant_outputs.shape}")
    print(f"   Processing time: {quantized_time*1000:.2f} ms")
    
    # 5. Compare results
    print("\n5. Comparing results...")
    
    # Calculate output differences
    output_diff = torch.abs(outputs - quant_outputs)
    mse = torch.mean(output_diff ** 2).item()
    mae = torch.mean(output_diff).item()
    max_diff = torch.max(output_diff).item()
    
    print(f"   Output MSE: {mse:.6f}")
    print(f"   Output MAE: {mae:.6f}")
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Speedup: {original_time/quantized_time:.2f}x")
    
    # 6. Model size comparison
    print("\n6. Model size comparison...")
    
    size_comparison = compare_model_sizes(model, quantized_model)
    print(f"   Original size: {size_comparison['original_size_mb']:.2f} MB")
    print(f"   Quantized size: {size_comparison['quantized_size_mb']:.2f} MB")
    print(f"   Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    print(f"   Size reduction: {size_comparison['size_reduction_percent']:.1f}%")
    
    # 7. Quantization statistics
    print("\n7. Quantization statistics...")
    
    quant_stats = get_quantization_stats(quantized_model)
    print(f"   Quantized modules: {len(quant_stats['quantized_modules'])}")
    print(f"   Quantized parameters: {quant_stats['quantized_parameters']:,}")
    print(f"   Total parameters: {quant_stats['total_parameters']:,}")
    
    return model, quantized_model, {
        'mse': mse,
        'mae': mae,
        'max_diff': max_diff,
        'speedup': original_time/quantized_time,
        'compression_ratio': size_comparison['compression_ratio'],
        'size_reduction': size_comparison['size_reduction_percent']
    }


def test_different_quantization_levels():
    """Test different quantization levels."""
    print("\n" + "=" * 50)
    print("Testing Different Quantization Levels")
    print("=" * 50)
    
    # Create base model
    model = wavlm_base()
    model.eval()
    
    # Test data
    test_audio = torch.randn(1, 16000)
    test_lengths = torch.tensor([16000])
    
    # Test different configurations
    configs = [
        ('8bit_symmetric', '8-bit Symmetric'),
        ('8bit_asymmetric', '8-bit Asymmetric'),
        ('4bit_symmetric', '4-bit Symmetric'),
        ('weights_only', 'Weights Only (8-bit)')
    ]
    
    results = {}
    
    print(f"\n{'Configuration':<20} {'MSE':<12} {'MAE':<12} {'Compression':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for config_name, display_name in configs:
        # Get configuration
        config = get_quantization_config(config_name)
        
        # Quantize model
        quantized_model = quantize_model(model, config)
        quantized_model.eval()
        
        # Test performance
        with torch.no_grad():
            start_time = time.time()
            orig_outputs, _ = model(test_audio, test_lengths)
            orig_time = time.time() - start_time
            
            start_time = time.time()
            quant_outputs, _ = quantized_model(test_audio, test_lengths)
            quant_time = time.time() - start_time
        
        # Calculate metrics
        output_diff = torch.abs(orig_outputs - quant_outputs)
        mse = torch.mean(output_diff ** 2).item()
        mae = torch.mean(output_diff).item()
        
        # Get size comparison
        size_comp = compare_model_sizes(model, quantized_model)
        
        results[config_name] = {
            'mse': mse,
            'mae': mae,
            'compression_ratio': size_comp['compression_ratio'],
            'speedup': orig_time/quant_time
        }
        
        print(f"{display_name:<20} {mse:<12.6f} {mae:<12.6f} "
              f"{size_comp['compression_ratio']:<12.2f} {orig_time/quant_time:<10.2f}")
    
    return results


def demonstrate_quantization_for_inference():
    """Demonstrate quantization for inference optimization."""
    print("\n" + "=" * 50)
    print("Quantization for Inference Optimization")
    print("=" * 50)
    
    # Create model
    model = wavlm_base()
    model.eval()
    
    # Create test data (simulate real audio processing)
    batch_sizes = [1, 4, 8, 16]
    sequence_length = 16000  # 1 second at 16kHz
    
    print(f"\nTesting inference performance with different batch sizes:")
    print(f"{'Batch Size':<12} {'Original (ms)':<15} {'Quantized (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    
    # Quantize model
    config = get_quantization_config('8bit_symmetric')
    quantized_model = quantize_model(model, config)
    quantized_model.eval()
    
    for batch_size in batch_sizes:
        # Create test data
        test_audio = torch.randn(batch_size, sequence_length)
        test_lengths = torch.tensor([sequence_length] * batch_size)
        
        # Test original model
        with torch.no_grad():
            start_time = time.time()
            _ = model(test_audio, test_lengths)
            orig_time = (time.time() - start_time) * 1000
        
        # Test quantized model
        with torch.no_grad():
            start_time = time.time()
            _ = quantized_model(test_audio, test_lengths)
            quant_time = (time.time() - start_time) * 1000
        
        speedup = orig_time / quant_time
        print(f"{batch_size:<12} {orig_time:<15.2f} {quant_time:<15.2f} {speedup:<10.2f}")


def save_quantized_model_example():
    """Example of saving and loading quantized model."""
    print("\n" + "=" * 50)
    print("Saving and Loading Quantized Model")
    print("=" * 50)
    
    # Create and quantize model
    model = wavlm_base()
    config = get_quantization_config('8bit_symmetric')
    quantized_model = quantize_model(model, config)
    
    # Save model
    model_path = "wavlm_base_quantized.pth"
    torch.save({
        'state_dict': quantized_model.state_dict(),
        'config': config.to_dict(),
        'model_type': 'wavlm_base_quantized'
    }, model_path)
    
    print(f"Quantized model saved to: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    loaded_model = wavlm_base()
    loaded_model.load_state_dict(checkpoint['state_dict'])
    
    print("Model loaded successfully")
    
    # Test loaded model
    test_audio = torch.randn(1, 16000)
    test_lengths = torch.tensor([16000])
    
    with torch.no_grad():
        outputs, _ = loaded_model(test_audio, test_lengths)
    
    print(f"Loaded model output shape: {outputs.shape}")
    
    # Clean up
    os.remove(model_path)
    print("Temporary model file removed")


def main():
    """Main function."""
    print("WavLM Base Quantization for WeDefense")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        # 1. Basic quantization example
        print("\n1. Basic Quantization Example")
        print("-" * 40)
        model, quantized_model, metrics = create_and_quantize_wavlm_base()
        
        # 2. Test different quantization levels
        print("\n2. Different Quantization Levels")
        print("-" * 40)
        config_results = test_different_quantization_levels()
        
        # 3. Inference optimization
        print("\n3. Inference Optimization")
        print("-" * 40)
        demonstrate_quantization_for_inference()
        
        # 4. Save/load example
        print("\n4. Save/Load Example")
        print("-" * 40)
        save_quantized_model_example()
        
        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"✓ WavLM Base model successfully quantized")
        print(f"✓ Compression ratio: {metrics['compression_ratio']:.2f}x")
        print(f"✓ Size reduction: {metrics['size_reduction']:.1f}%")
        print(f"✓ Output MSE: {metrics['mse']:.6f}")
        print(f"✓ Speedup: {metrics['speedup']:.2f}x")
        print(f"✓ Multiple quantization configurations tested")
        print(f"✓ Inference optimization demonstrated")
        print(f"✓ Save/load functionality verified")
        
        print("\nWavLM Base quantization example completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
