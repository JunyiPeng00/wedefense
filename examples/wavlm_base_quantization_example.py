#!/usr/bin/env python3
"""WavLM Base Quantization Example for WeDefense.

This script demonstrates how to quantize a WavLM Base model using LSQ quantization
for efficient inference in anti-spoofing tasks.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wedefense.frontend.wav2vec2 import (
    wavlm_base,
    QuantizationConfig,
    quantize_model,
    get_quantization_config,
    QuantizationAwareTrainer,
    compare_model_sizes,
    get_quantization_stats
)


def create_wavlm_base_model():
    """Create a WavLM Base model for quantization."""
    print("Creating WavLM Base model...")
    
    # Create WavLM Base model
    model = wavlm_base(
        encoder_projection_dropout=0.1,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.1,
        encoder_layer_drop=0.1,
        aux_num_out=None
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"WavLM Base model created with {total_params:,} parameters")
    
    return model


def create_audio_data(batch_size=8, sequence_length=16000, num_samples=100):
    """Create synthetic audio data for testing."""
    print(f"Creating synthetic audio data: {num_samples} samples, {sequence_length} samples each")
    
    # Generate synthetic audio waveforms (16kHz, 1 second)
    audio_data = []
    labels = []
    
    for i in range(num_samples):
        # Generate random audio waveform
        audio = torch.randn(sequence_length) * 0.1  # Small amplitude
        # Add some structure (sine wave + noise)
        t = torch.linspace(0, 1, sequence_length)
        audio += 0.05 * torch.sin(2 * np.pi * 440 * t)  # A4 note
        audio += 0.02 * torch.randn(sequence_length)  # Noise
        
        audio_data.append(audio)
        labels.append(i % 2)  # Binary labels for demo
    
    # Create dataset
    audio_tensor = torch.stack(audio_data)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Split into train/val
    train_size = int(0.8 * len(audio_tensor))
    train_audio = audio_tensor[:train_size]
    train_labels = labels_tensor[:train_size]
    val_audio = audio_tensor[train_size:]
    val_labels = labels_tensor[train_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_audio, train_labels)
    val_dataset = TensorDataset(val_audio, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_audio)}")
    print(f"Validation samples: {len(val_audio)}")
    
    return train_loader, val_loader


def test_wavlm_forward_pass(model, device):
    """Test WavLM model forward pass."""
    print("Testing WavLM forward pass...")
    
    model = model.to(device)
    model.eval()
    
    # Test with different sequence lengths
    test_lengths = [16000, 8000, 32000]  # 1s, 0.5s, 2s at 16kHz
    
    for length in test_lengths:
        # Create test audio
        test_audio = torch.randn(2, length, device=device)
        test_lengths_tensor = torch.tensor([length, length], device=device)
        
        with torch.no_grad():
            # Forward pass
            start_time = time.time()
            outputs, output_lengths = model(test_audio, test_lengths_tensor)
            end_time = time.time()
        
        print(f"  Input: {test_audio.shape} -> Output: {outputs.shape}")
        print(f"  Processing time: {(end_time - start_time)*1000:.2f} ms")
        print(f"  Output length: {output_lengths}")


def quantize_wavlm_model(model, config_name='8bit_symmetric'):
    """Quantize WavLM model with specified configuration."""
    print(f"\nQuantizing WavLM model with {config_name} configuration...")
    
    # Get quantization configuration
    config = get_quantization_config(config_name)
    print(f"Quantization config: {config.to_dict()}")
    
    # Quantize the model
    quantized_model = quantize_model(model, config)
    
    # Get quantization statistics
    stats = get_quantization_stats(quantized_model)
    print(f"Quantized modules: {len(stats['quantized_modules'])}")
    print(f"Quantized parameters: {stats['quantized_parameters']:,}")
    print(f"Total parameters: {stats['total_parameters']:,}")
    
    return quantized_model, config


def compare_model_performance(original_model, quantized_model, test_audio, device):
    """Compare performance between original and quantized models."""
    print("\nComparing model performance...")
    
    original_model = original_model.to(device)
    quantized_model = quantized_model.to(device)
    
    # Set to evaluation mode
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        # Test original model
        start_time = time.time()
        orig_outputs, orig_lengths = original_model(test_audio)
        orig_time = time.time() - start_time
        
        # Test quantized model
        start_time = time.time()
        quant_outputs, quant_lengths = quantized_model(test_audio)
        quant_time = time.time() - start_time
    
    # Calculate differences
    output_diff = torch.abs(orig_outputs - quant_outputs)
    mse = torch.mean(output_diff ** 2).item()
    mae = torch.mean(output_diff).item()
    max_diff = torch.max(output_diff).item()
    
    # Calculate relative differences
    orig_norm = torch.mean(orig_outputs ** 2).item()
    relative_mse = mse / orig_norm if orig_norm > 0 else 0
    
    print(f"Original model time: {orig_time*1000:.2f} ms")
    print(f"Quantized model time: {quant_time*1000:.2f} ms")
    print(f"Speedup: {orig_time/quant_time:.2f}x")
    print(f"Output MSE: {mse:.6f}")
    print(f"Relative MSE: {relative_mse:.6f}")
    print(f"Output MAE: {mae:.6f}")
    print(f"Max difference: {max_diff:.6f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'max_diff': max_diff,
        'relative_mse': relative_mse,
        'speedup': orig_time/quant_time
    }


def demonstrate_quantization_aware_training(model, train_loader, val_loader, device):
    """Demonstrate quantization-aware training for WavLM."""
    print("\nDemonstrating Quantization-Aware Training...")
    
    # Create quantization configuration
    config = get_quantization_config('8bit_symmetric')
    
    # Create trainer
    trainer = QuantizationAwareTrainer(model, config, device)
    
    print("Calibrating quantization parameters...")
    trainer.calibrate_model(train_loader, num_batches=5)
    
    print("Fine-tuning quantized model...")
    history = trainer.fine_tune_quantized_model(
        train_loader,
        val_loader,
        num_epochs=3,
        learning_rate=1e-4
    )
    
    print(f"Training completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if 'val_acc' in history:
        print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    return trainer.quantized_model, history


def test_different_quantization_configs(model, test_audio, device):
    """Test different quantization configurations."""
    print("\nTesting different quantization configurations...")
    
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
        
        # Compare performance
        perf = compare_model_performance(model, quantized_model, test_audio, device)
        
        # Get size comparison
        size_comp = compare_model_sizes(model, quantized_model)
        
        # Get quantization stats
        quant_stats = get_quantization_stats(quantized_model)
        
        results[config_name] = {
            'performance': perf,
            'size_comparison': size_comp,
            'quantization_stats': quant_stats
        }
        
        print(f"  Compression ratio: {size_comp['compression_ratio']:.2f}x")
        print(f"  Size reduction: {size_comp['size_reduction_percent']:.1f}%")
        print(f"  Output MSE: {perf['mse']:.6f}")
        print(f"  Speedup: {perf['speedup']:.2f}x")
    
    return results


def create_quantization_report(results):
    """Create a detailed quantization report."""
    print("\n" + "="*80)
    print("WavLM Base Quantization Report")
    print("="*80)
    
    print(f"{'Configuration':<20} {'Compression':<12} {'Size Reduction':<15} {'MSE':<12} {'Speedup':<10}")
    print("-"*80)
    
    for config_name, result in results.items():
        size_comp = result['size_comparison']
        perf = result['performance']
        
        print(f"{config_name:<20} {size_comp['compression_ratio']:<12.2f} "
              f"{size_comp['size_reduction_percent']:<15.1f}% {perf['mse']:<12.6f} "
              f"{perf['speedup']:<10.2f}x")
    
    # Find best configuration
    best_config = min(results.keys(), 
                     key=lambda x: results[x]['performance']['mse'])
    
    print(f"\nBest configuration (lowest MSE): {best_config}")
    print(f"  MSE: {results[best_config]['performance']['mse']:.6f}")
    print(f"  Compression: {results[best_config]['size_comparison']['compression_ratio']:.2f}x")
    print(f"  Speedup: {results[best_config]['performance']['speedup']:.2f}x")


def main():
    """Main function to run WavLM Base quantization example."""
    print("WavLM Base Quantization Example for WeDefense")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 1. Create WavLM Base model
        print("\n" + "="*60)
        print("1. Creating WavLM Base Model")
        print("="*60)
        
        model = create_wavlm_base_model()
        
        # 2. Test original model
        print("\n" + "="*60)
        print("2. Testing Original Model")
        print("="*60)
        
        test_audio = torch.randn(2, 16000, device=device)  # 2 samples, 1 second each
        test_wavlm_forward_pass(model, device)
        
        # 3. Create test data
        print("\n" + "="*60)
        print("3. Creating Test Data")
        print("="*60)
        
        train_loader, val_loader = create_audio_data(batch_size=4, sequence_length=16000, num_samples=50)
        
        # 4. Basic quantization
        print("\n" + "="*60)
        print("4. Basic Quantization")
        print("="*60)
        
        quantized_model, config = quantize_wavlm_model(model, '8bit_symmetric')
        
        # 5. Compare performance
        print("\n" + "="*60)
        print("5. Performance Comparison")
        print("="*60)
        
        compare_model_performance(model, quantized_model, test_audio, device)
        
        # 6. Quantization-aware training
        print("\n" + "="*60)
        print("6. Quantization-Aware Training")
        print("="*60)
        
        qat_model, training_history = demonstrate_quantization_aware_training(
            model, train_loader, val_loader, device
        )
        
        # 7. Test different configurations
        print("\n" + "="*60)
        print("7. Different Quantization Configurations")
        print("="*60)
        
        results = test_different_quantization_configs(model, test_audio, device)
        
        # 8. Generate report
        print("\n" + "="*60)
        print("8. Quantization Report")
        print("="*60)
        
        create_quantization_report(results)
        
        print("\n" + "="*60)
        print("WavLM Base Quantization Example Completed Successfully!")
        print("="*60)
        
        # Summary
        print("\nSummary:")
        print("- WavLM Base model successfully quantized")
        print("- Multiple quantization configurations tested")
        print("- Quantization-aware training demonstrated")
        print("- Performance and size comparisons completed")
        print("- Detailed quantization report generated")
        
    except Exception as e:
        print(f"\nError during WavLM quantization example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
