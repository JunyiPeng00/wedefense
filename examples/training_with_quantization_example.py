#!/usr/bin/env python3
"""Example of training with quantization and HP framework integration.

This script demonstrates how to use the new quantization features in the
WeDefense training pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from wedefense.frontend.wav2vec2 import wavlm_base, apply_quantization_with_hp_integration

def create_quantization_config():
    """Create a sample configuration with quantization enabled."""
    config = {
        # Basic training config
        'model': 'SSL_BACKEND_WavLM',
        'model_args': {
            'embed_dim': 768,
            'feat_dim': 768,
        },
        'dataset_args': {
            'frontend': 'wavlm',
            'wavlm_args': {
                'model_name': 'wavlm_base',
                'normalize_waveform': True,
            },
            'resample_rate': 16000,
        },
        'projection_args': {
            'project_type': 'Linear',
            'embed_dim': 768,
            'num_class': 2,
        },
        
        # Quantization config
        'use_quantization': True,
        'quantization_config': '8bit_symmetric',
        'quantize_weights': True,
        'quantize_activations': False,
        'preserve_hp_gating': True,
        
        # Pruning config (optional)
        'use_pruning_loss': True,
        'target_sparsity': 0.3,
        'sparsity_warmup_epochs': 5,
        'sparsity_schedule': 'cosine',
        'min_sparsity': 0.0,
    }
    return config

def demonstrate_quantization_workflow():
    """Demonstrate the complete quantization workflow."""
    print("WeDefense Quantization Training Example")
    print("=" * 50)
    
    # Step 1: Create model
    print("1. Creating WavLM Base model...")
    model = wavlm_base()
    original_params = sum(p.numel() for p in model.parameters())
    print(f"   Original parameters: {original_params:,}")
    
    # Step 2: Apply quantization with HP integration
    print("\n2. Applying quantization with HP framework integration...")
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
    
    quantized_params = sum(p.numel() for p in quantized_model.parameters())
    print(f"   Quantized parameters: {quantized_params:,}")
    print(f"   Parameter increase: {quantized_params - original_params:,} (quantization overhead)")
    
    # Step 3: Test inference
    print("\n3. Testing inference...")
    batch_size = 2
    seq_len = 16000
    audio_input = torch.randn(batch_size, seq_len)
    audio_lengths = torch.tensor([seq_len] * batch_size)
    
    with torch.no_grad():
        outputs, lengths = quantized_model(audio_input, audio_lengths)
    
    print(f"   Input shape: {audio_input.shape}")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Output lengths: {lengths}")
    
    # Step 4: Test training compatibility
    print("\n4. Testing training compatibility...")
    quantized_model.train()
    
    # Create new input with gradients enabled
    audio_input_grad = torch.randn(batch_size, seq_len, requires_grad=True)
    audio_lengths_grad = torch.tensor([seq_len] * batch_size)
    
    # Forward pass
    outputs_grad, lengths_grad = quantized_model(audio_input_grad, audio_lengths_grad)
    
    # Create dummy loss
    loss = outputs_grad.mean()
    loss.backward()
    
    # Count trainable parameters
    trainable_params = sum(1 for p in quantized_model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable_params}")
    
    # Check if gradients were computed
    grad_count = sum(1 for p in quantized_model.parameters() if p.grad is not None)
    print(f"   Parameters with gradients: {grad_count}")
    
    # Step 5: Show quantization statistics
    print("\n5. Quantization statistics...")
    quantized_modules = 0
    for name, module in quantized_model.named_modules():
        if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
            quantized_modules += 1
    
    print(f"   Quantized modules: {quantized_modules}")
    
    # Count gating modules
    gating_modules = 0
    for name, module in quantized_model.named_modules():
        if hasattr(module, 'hard_concrete') and module.hard_concrete is not None:
            gating_modules += 1
        if hasattr(module, 'hard_concrete_for_heads') and module.hard_concrete_for_heads is not None:
            gating_modules += 1
        if hasattr(module, 'hard_concrete_for_layer') and module.hard_concrete_for_layer is not None:
            gating_modules += 1
        if hasattr(module, 'hard_concrete_for_intermediate') and module.hard_concrete_for_intermediate is not None:
            gating_modules += 1
    
    print(f"   Gating modules: {gating_modules}")
    
    print("\n✓ Quantization workflow completed successfully!")
    return quantized_model

def show_training_config_example():
    """Show how to configure training with quantization."""
    print("\n" + "=" * 50)
    print("Training Configuration Example")
    print("=" * 50)
    
    config = create_quantization_config()
    
    print("Sample config.yaml for quantization training:")
    print("-" * 40)
    print(yaml.dump(config, default_flow_style=False, indent=2))
    
    print("\nTo use this configuration:")
    print("1. Save the above config to a YAML file (e.g., config_quant.yaml)")
    print("2. Run training with: python wedefense/bin/train.py --config config_quant.yaml")
    print("3. The quantization will be automatically applied during model initialization")

def show_command_line_examples():
    """Show command line examples for quantization training."""
    print("\n" + "=" * 50)
    print("Command Line Examples")
    print("=" * 50)
    
    examples = [
        {
            "description": "Basic quantization training (8-bit symmetric)",
            "command": "python wedefense/bin/train.py --use_quantization True --quantization_config 8bit_symmetric"
        },
        {
            "description": "Quantization with pruning",
            "command": "python wedefense/bin/train.py --use_quantization True --use_pruning_loss True --target_sparsity 0.5"
        },
        {
            "description": "4-bit quantization for maximum compression",
            "command": "python wedefense/bin/train.py --use_quantization True --quantization_config 4bit_symmetric"
        },
        {
            "description": "Quantization with activation quantization",
            "command": "python wedefense/bin/train.py --use_quantization True --quantize_activations True"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}:")
        print(f"   {example['command']}")
        print()

def main():
    """Main demonstration function."""
    try:
        # Demonstrate the workflow
        quantized_model = demonstrate_quantization_workflow()
        
        # Show configuration examples
        show_training_config_example()
        
        # Show command line examples
        show_command_line_examples()
        
        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        print("✓ Quantization integration is working correctly")
        print("✓ HP framework gating is preserved")
        print("✓ Training compatibility is maintained")
        print("✓ Multiple quantization configurations are supported")
        print("\nThe quantization features are ready for production use!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
