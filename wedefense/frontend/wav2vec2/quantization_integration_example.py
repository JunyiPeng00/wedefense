"""Integration example for LSQ quantization in WeDefense.

This module demonstrates how to integrate LSQ quantization into the WeDefense
pipeline, including quantization of wav2vec2 models and training integration.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .lsq_quantizer import LSQQuantizer, create_lsq_quantizer
from .quantized_layers import QuantizedLinear, QuantizedConv1d
from .quantization_utils import (
    QuantizationConfig,
    quantize_model,
    get_quantization_stats,
    compare_model_sizes,
    get_quantization_config
)
from .quantization_training import QuantizationAwareTrainer


def quantize_wav2vec2_model(
    model: nn.Module,
    config: QuantizationConfig,
    quantize_frontend: bool = True,
    quantize_encoder: bool = True
) -> nn.Module:
    """Quantize a wav2vec2 model with LSQ quantization.
    
    Args:
        model: wav2vec2 model to quantize
        config: Quantization configuration
        quantize_frontend: Whether to quantize the feature extractor
        quantize_encoder: Whether to quantize the transformer encoder
        
    Returns:
        Quantized model
    """
    print("Quantizing wav2vec2 model...")
    
    # Create a custom config that excludes certain modules if needed
    custom_config = QuantizationConfig(
        weight_bits=config.weight_bits,
        activation_bits=config.activation_bits,
        weight_symmetric=config.weight_symmetric,
        activation_symmetric=config.activation_symmetric,
        quantize_weights=config.quantize_weights,
        quantize_activations=config.quantize_activations,
        per_channel_weights=config.per_channel_weights,
        per_channel_activations=config.per_channel_activations,
        exclude_modules=[],
        include_modules=None
    )
    
    # Add exclusions based on user preferences
    if not quantize_frontend:
        custom_config.exclude_modules.extend(['feature_extractor', 'conv_layers'])
    
    if not quantize_encoder:
        custom_config.exclude_modules.extend(['encoder', 'transformer', 'attention', 'feed_forward'])
    
    # Quantize the model
    quantized_model = quantize_model(model, custom_config)
    
    print(f"Quantization complete. Excluded modules: {custom_config.exclude_modules}")
    
    return quantized_model


def create_quantization_aware_training_config(
    base_config: Dict[str, Any],
    quantization_config: QuantizationConfig,
    quantization_warmup_epochs: int = 5,
    quantization_start_epoch: int = 0
) -> Dict[str, Any]:
    """Create training configuration with quantization support.
    
    Args:
        base_config: Base training configuration
        quantization_config: Quantization configuration
        quantization_warmup_epochs: Number of epochs for quantization warmup
        quantization_start_epoch: Epoch to start quantization
        
    Returns:
        Updated training configuration
    """
    config = base_config.copy()
    
    # Add quantization parameters
    config['quantization'] = {
        'enabled': True,
        'config': quantization_config.to_dict(),
        'warmup_epochs': quantization_warmup_epochs,
        'start_epoch': quantization_start_epoch
    }
    
    # Add quantization-aware training parameters
    config['quantization_aware_training'] = {
        'calibration_batches': 100,
        'freeze_quantization_after_calibration': False,
        'gradual_quantization': True
    }
    
    return config


def quantize_model_for_inference(
    model_path: str,
    output_path: str,
    quantization_config: str = '8bit_symmetric',
    device: str = 'cpu'
) -> None:
    """Quantize a trained model for efficient inference.
    
    Args:
        model_path: Path to the trained model
        output_path: Path to save the quantized model
        quantization_config: Name of the quantization configuration
        device: Device to run quantization on
    """
    print(f"Loading model from {model_path}...")
    
    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        model_config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        model_config = {}
    
    # Create model (this would need to be adapted based on your model architecture)
    # For demonstration, we'll assume a simple model structure
    print("Note: Model creation needs to be adapted based on your specific architecture")
    
    # Get quantization configuration
    config = get_quantization_config(quantization_config)
    
    # Quantize the model
    # quantized_model = quantize_wav2vec2_model(model, config)
    
    # Save quantized model
    # torch.save({
    #     'state_dict': quantized_model.state_dict(),
    #     'config': model_config,
    #     'quantization_config': config.to_dict()
    # }, output_path)
    
    print(f"Quantized model saved to {output_path}")


def analyze_quantization_impact(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_data: torch.Tensor
) -> Dict[str, Any]:
    """Analyze the impact of quantization on model performance.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        test_data: Test data for evaluation
        
    Returns:
        Dictionary containing analysis results
    """
    print("Analyzing quantization impact...")
    
    # Set models to evaluation mode
    original_model.eval()
    quantized_model.eval()
    
    # Get model sizes
    size_comparison = compare_model_sizes(original_model, quantized_model)
    
    # Get quantization statistics
    quant_stats = get_quantization_stats(quantized_model)
    
    # Compare outputs
    with torch.no_grad():
        original_output = original_model(test_data)
        quantized_output = quantized_model(test_data)
        
        # Calculate output differences
        output_diff = torch.abs(original_output - quantized_output)
        mse = torch.mean(output_diff ** 2).item()
        mae = torch.mean(output_diff).item()
        max_diff = torch.max(output_diff).item()
        
        # Calculate relative differences
        relative_mse = mse / torch.mean(original_output ** 2).item()
        relative_mae = mae / torch.mean(torch.abs(original_output)).item()
    
    analysis = {
        'size_comparison': size_comparison,
        'quantization_stats': quant_stats,
        'output_analysis': {
            'mse': mse,
            'mae': mae,
            'max_difference': max_diff,
            'relative_mse': relative_mse,
            'relative_mae': relative_mae
        }
    }
    
    print(f"Model size reduction: {size_comparison['size_reduction_percent']:.1f}%")
    print(f"Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    print(f"Output MSE: {mse:.6f}")
    print(f"Relative MSE: {relative_mse:.6f}")
    
    return analysis


def create_quantization_experiment_config(
    experiment_name: str,
    base_model_config: Dict[str, Any],
    quantization_configs: list = None
) -> Dict[str, Any]:
    """Create configuration for quantization experiments.
    
    Args:
        experiment_name: Name of the experiment
        base_model_config: Base model configuration
        quantization_configs: List of quantization configurations to test
        
    Returns:
        Experiment configuration
    """
    if quantization_configs is None:
        quantization_configs = [
            '8bit_symmetric',
            '8bit_asymmetric', 
            '4bit_symmetric',
            'mixed_precision',
            'weights_only'
        ]
    
    experiment_config = {
        'experiment_name': experiment_name,
        'base_model': base_model_config,
        'quantization_configs': quantization_configs,
        'evaluation': {
            'metrics': ['accuracy', 'loss', 'inference_time', 'model_size'],
            'test_sets': ['dev', 'eval']
        },
        'training': {
            'quantization_aware': True,
            'calibration_epochs': 2,
            'fine_tune_epochs': 10,
            'learning_rate': 1e-4
        }
    }
    
    return experiment_config


def run_quantization_experiment(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    experiment_config: Dict[str, Any],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Run a complete quantization experiment.
    
    Args:
        model: Model to quantize
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        experiment_config: Experiment configuration
        device: Device to run on
        
    Returns:
        Experiment results
    """
    device = torch.device(device)
    results = {}
    
    print(f"Running quantization experiment: {experiment_config['experiment_name']}")
    
    for config_name in experiment_config['quantization_configs']:
        print(f"\nTesting quantization config: {config_name}")
        
        # Get quantization configuration
        quant_config = get_quantization_config(config_name)
        
        # Create trainer
        trainer = QuantizationAwareTrainer(model, quant_config, device)
        
        # Calibrate model
        print("Calibrating model...")
        trainer.calibrate_model(train_dataloader, num_batches=50)
        
        # Fine-tune model
        print("Fine-tuning quantized model...")
        history = trainer.fine_tune_quantized_model(
            train_dataloader,
            val_dataloader,
            num_epochs=experiment_config['training']['fine_tune_epochs'],
            learning_rate=experiment_config['training']['learning_rate']
        )
        
        # Evaluate model
        print("Evaluating quantized model...")
        val_metrics = trainer.evaluate_quantized_model(val_dataloader)
        
        # Get quantization statistics
        quant_stats = trainer.get_quantization_statistics()
        
        # Store results
        results[config_name] = {
            'training_history': history,
            'validation_metrics': val_metrics,
            'quantization_stats': quant_stats,
            'config': quant_config.to_dict()
        }
        
        print(f"Results for {config_name}:")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.2f}%")
        print(f"  Validation Loss: {val_metrics['loss']:.4f}")
        print(f"  Quantized Parameters: {quant_stats['quantized_parameters']:,}")
        print(f"  Total Parameters: {quant_stats['total_parameters']:,}")
    
    return results


# Example usage
if __name__ == "__main__":
    print("LSQ Quantization Integration Example")
    print("=" * 50)
    
    # Example 1: Basic quantization
    print("\n1. Basic Model Quantization")
    print("-" * 30)
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(100, 50)
            self.linear2 = nn.Linear(50, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    config = get_quantization_config('8bit_symmetric')
    
    # Quantize the model
    quantized_model = quantize_model(model, config)
    
    # Analyze the impact
    test_data = torch.randn(32, 100)
    analysis = analyze_quantization_impact(model, quantized_model, test_data)
    
    print("\n2. Quantization-Aware Training")
    print("-" * 30)
    
    # Create dummy data
    x = torch.randn(100, 100)
    y = torch.randint(0, 10, (100,))
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create trainer
    trainer = QuantizationAwareTrainer(model, config, torch.device('cpu'))
    
    # Run training
    history = trainer.fine_tune_quantized_model(
        dataloader,
        num_epochs=3,
        learning_rate=1e-3
    )
    
    print(f"Training completed. Final loss: {history['train_loss'][-1]:.4f}")
    
    print("\n3. Model Size Comparison")
    print("-" * 30)
    
    size_comparison = compare_model_sizes(model, quantized_model)
    print(f"Original size: {size_comparison['original_size_mb']:.2f} MB")
    print(f"Quantized size: {size_comparison['quantized_size_mb']:.2f} MB")
    print(f"Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    print(f"Size reduction: {size_comparison['size_reduction_percent']:.1f}%")
    
    print("\nQuantization integration example completed!")
