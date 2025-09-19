#!/usr/bin/env python3
"""Test script for quantization and HP framework integration.

This script demonstrates how to use the new apply_quantization functions
with the HP framework for structured pruning.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
from wedefense.frontend.wav2vec2 import (
    wavlm_base, 
    apply_quantization, 
    apply_quantization_with_hp_integration,
    get_quantization_config
)

def test_basic_quantization():
    """Test basic quantization without HP framework."""
    print("=" * 60)
    print("Test 1: Basic Quantization")
    print("=" * 60)
    
    # Create model
    model = wavlm_base()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Apply quantization
    quantized_model = apply_quantization(model, preserve_hp_gating=True)
    print(f"Quantized model parameters: {sum(p.numel() for p in quantized_model.parameters()):,}")
    
    # Test inference
    batch_size = 2
    seq_len = 16000
    audio_input = torch.randn(batch_size, seq_len)
    audio_lengths = torch.tensor([seq_len] * batch_size)
    
    with torch.no_grad():
        start_time = time.time()
        outputs, lengths = quantized_model(audio_input, audio_lengths)
        inference_time = time.time() - start_time
    
    print(f"Output shape: {outputs.shape}")
    print(f"Inference time: {inference_time:.3f}s")
    print("✓ Basic quantization test passed!")
    return quantized_model

def test_hp_integration():
    """Test quantization with HP framework integration."""
    print("\n" + "=" * 60)
    print("Test 2: Quantization with HP Framework Integration")
    print("=" * 60)
    
    # Create model
    model = wavlm_base()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Apply quantization with HP integration
    quantized_model = apply_quantization_with_hp_integration(
        model,
        enable_pruning=True,
        pruning_config={
            'target_sparsity': 0.3,
            'sparsity_warmup_epochs': 5,
            'sparsity_schedule': 'cosine',
            'min_sparsity': 0.0
        }
    )
    print(f"Quantized + HP model parameters: {sum(p.numel() for p in quantized_model.parameters()):,}")
    
    # Test inference
    batch_size = 2
    seq_len = 16000
    audio_input = torch.randn(batch_size, seq_len)
    audio_lengths = torch.tensor([seq_len] * batch_size)
    
    with torch.no_grad():
        start_time = time.time()
        outputs, lengths = quantized_model(audio_input, audio_lengths)
        inference_time = time.time() - start_time
    
    print(f"Output shape: {outputs.shape}")
    print(f"Inference time: {inference_time:.3f}s")
    print("✓ HP integration test passed!")
    return quantized_model

def test_different_quantization_configs():
    """Test different quantization configurations."""
    print("\n" + "=" * 60)
    print("Test 3: Different Quantization Configurations")
    print("=" * 60)
    
    model = wavlm_base()
    configs = ['8bit_symmetric', '8bit_asymmetric', '4bit_symmetric']
    
    results = {}
    for config_name in configs:
        print(f"\nTesting {config_name}...")
        
        # Get quantization config
        quant_config = get_quantization_config(config_name)
        
        # Apply quantization
        quantized_model = apply_quantization(model, config=quant_config)
        
        # Test inference
        batch_size = 1
        seq_len = 16000
        audio_input = torch.randn(batch_size, seq_len)
        audio_lengths = torch.tensor([seq_len])
        
        with torch.no_grad():
            start_time = time.time()
            outputs, lengths = quantized_model(audio_input, audio_lengths)
            inference_time = time.time() - start_time
        
        results[config_name] = {
            'output_shape': outputs.shape,
            'inference_time': inference_time,
            'parameters': sum(p.numel() for p in quantized_model.parameters())
        }
        
        print(f"  Output shape: {outputs.shape}")
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  Parameters: {results[config_name]['parameters']:,}")
    
    print("\nConfiguration comparison:")
    print(f"{'Config':<20} {'Parameters':<12} {'Time (ms)':<10}")
    print("-" * 45)
    for config_name, result in results.items():
        print(f"{config_name:<20} {result['parameters']:<12,} {result['inference_time']*1000:<10.1f}")
    
    print("✓ Different configurations test passed!")

def test_gating_preservation():
    """Test that HP gating is preserved during quantization."""
    print("\n" + "=" * 60)
    print("Test 4: HP Gating Preservation")
    print("=" * 60)
    
    model = wavlm_base()
    
    # Count gating modules before quantization
    gating_before = 0
    for name, module in model.named_modules():
        if hasattr(module, 'hard_concrete') and module.hard_concrete is not None:
            gating_before += 1
        if hasattr(module, 'hard_concrete_for_heads') and module.hard_concrete_for_heads is not None:
            gating_before += 1
        if hasattr(module, 'hard_concrete_for_layer') and module.hard_concrete_for_layer is not None:
            gating_before += 1
        if hasattr(module, 'hard_concrete_for_intermediate') and module.hard_concrete_for_intermediate is not None:
            gating_before += 1
    
    print(f"Gating modules before quantization: {gating_before}")
    
    # Apply quantization with HP integration
    quantized_model = apply_quantization_with_hp_integration(
        model,
        enable_pruning=True
    )
    
    # Count gating modules after quantization
    gating_after = 0
    for name, module in quantized_model.named_modules():
        if hasattr(module, 'hard_concrete') and module.hard_concrete is not None:
            gating_after += 1
        if hasattr(module, 'hard_concrete_for_heads') and module.hard_concrete_for_heads is not None:
            gating_after += 1
        if hasattr(module, 'hard_concrete_for_layer') and module.hard_concrete_for_layer is not None:
            gating_after += 1
        if hasattr(module, 'hard_concrete_for_intermediate') and module.hard_concrete_for_intermediate is not None:
            gating_after += 1
    
    print(f"Gating modules after quantization: {gating_after}")
    
    if gating_after >= gating_before:
        print("✓ HP gating preservation test passed!")
    else:
        print("✗ HP gating preservation test failed!")
        return False
    
    return True

def test_training_compatibility():
    """Test that quantized models are compatible with training."""
    print("\n" + "=" * 60)
    print("Test 5: Training Compatibility")
    print("=" * 60)
    
    model = wavlm_base()
    quantized_model = apply_quantization_with_hp_integration(model)
    
    # Set to training mode
    quantized_model.train()
    
    # Create dummy data
    batch_size = 2
    seq_len = 16000
    audio_input = torch.randn(batch_size, seq_len, requires_grad=True)
    audio_lengths = torch.tensor([seq_len] * batch_size)
    
    # Forward pass
    outputs, lengths = quantized_model(audio_input, audio_lengths)
    
    # Create dummy loss
    loss = outputs.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_count = 0
    for name, param in quantized_model.named_parameters():
        if param.grad is not None:
            grad_count += 1
    
    print(f"Parameters with gradients: {grad_count}")
    print(f"Total parameters: {sum(1 for _ in quantized_model.parameters())}")
    
    if grad_count > 0:
        print("✓ Training compatibility test passed!")
        return True
    else:
        print("✗ Training compatibility test failed!")
        return False

def main():
    """Run all tests."""
    print("Quantization and HP Framework Integration Tests")
    print("=" * 60)
    
    try:
        # Run tests
        test_basic_quantization()
        test_hp_integration()
        test_different_quantization_configs()
        
        gating_ok = test_gating_preservation()
        training_ok = test_training_compatibility()
        
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        if gating_ok and training_ok:
            print("✓ All tests passed! Quantization and HP framework integration is working correctly.")
        else:
            print("✗ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
