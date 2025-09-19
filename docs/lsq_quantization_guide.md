# LSQ Quantization Guide for WeDefense

This guide explains how to use the Learnable Step Size Quantization (LSQ) functionality integrated into WeDefense for model compression and efficient inference.

## Overview

LSQ (Learnable Step Size Quantization) is a state-of-the-art quantization method that learns optimal quantization parameters during training. It provides:

- **High compression ratios** (up to 4x model size reduction)
- **Minimal accuracy loss** through learnable step sizes
- **Flexible quantization** (4-bit, 8-bit, mixed precision)
- **Easy integration** with existing WeDefense models

## Quick Start

### Basic Quantization

```python
from wedefense.frontend.wav2vec2 import (
    quantize_model, 
    get_quantization_config,
    compare_model_sizes
)

# Load your model
model = load_your_model()

# Get quantization configuration
config = get_quantization_config('8bit_symmetric')

# Quantize the model
quantized_model = quantize_model(model, config)

# Compare model sizes
size_comparison = compare_model_sizes(model, quantized_model)
print(f"Compression ratio: {size_comparison['compression_ratio']:.2f}x")
```

### Quantization-Aware Training

```python
from wedefense.frontend.wav2vec2 import QuantizationAwareTrainer

# Create trainer
trainer = QuantizationAwareTrainer(model, config, device)

# Calibrate quantization parameters
trainer.calibrate_model(train_dataloader, num_batches=100)

# Fine-tune the quantized model
history = trainer.fine_tune_quantized_model(
    train_dataloader,
    val_dataloader,
    num_epochs=10,
    learning_rate=1e-4
)
```

## Configuration Options

### Predefined Configurations

| Configuration | Weight Bits | Activation Bits | Symmetric | Use Case |
|---------------|-------------|-----------------|-----------|----------|
| `8bit_symmetric` | 8 | 8 | Yes | Balanced performance/size |
| `8bit_asymmetric` | 8 | 8 | No | Better for skewed distributions |
| `4bit_symmetric` | 4 | 4 | Yes | Maximum compression |
| `mixed_precision` | 8 | 16 | Yes | High accuracy, moderate compression |
| `weights_only` | 8 | 32 | Yes | Quantize weights only |

### Custom Configuration

```python
from wedefense.frontend.wav2vec2 import QuantizationConfig

config = QuantizationConfig(
    weight_bits=8,
    activation_bits=8,
    weight_symmetric=True,
    activation_symmetric=True,
    quantize_weights=True,
    quantize_activations=False,
    per_channel_weights=True,
    exclude_modules=['layer_norm', 'bias']
)
```

## Advanced Usage

### Gradual Quantization

```python
from wedefense.frontend.wav2vec2 import create_gradual_quantization_schedule

# Create gradual quantization schedule
schedule = create_gradual_quantization_schedule(
    total_epochs=20,
    quantization_start_epoch=5,
    quantization_warmup_epochs=10
)

# Apply quantization based on schedule
for epoch in range(20):
    if schedule[epoch]:
        model = quantize_model_with_gradual_schedule(model, config, schedule, epoch)
    # Continue training...
```

### Per-Channel Quantization

```python
from wedefense.frontend.wav2vec2 import create_lsq_quantizer

# Create per-channel quantizer for weights
weight_quantizer = create_lsq_quantizer(
    num_bits=8,
    symmetric=True,
    per_channel=True,
    channel_axis=0  # Quantize per output channel
)

# Use in custom layer
class CustomQuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_quantizer = weight_quantizer
    
    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.linear.weight)
        return F.linear(x, quantized_weight, self.linear.bias)
```

### Quantization Statistics

```python
from wedefense.frontend.wav2vec2 import get_quantization_stats

# Get detailed quantization statistics
stats = get_quantization_stats(quantized_model)

print(f"Total parameters: {stats['total_parameters']:,}")
print(f"Quantized parameters: {stats['quantized_parameters']:,}")
print(f"Quantization errors: {stats['quantization_errors']}")
print(f"Effective bits: {stats['effective_bits']}")
```

## Integration with WeDefense Pipeline

### Training Integration

```python
# In your training script
from wedefense.frontend.wav2vec2 import QuantizationAwareTrainer

def train_with_quantization(model, train_loader, val_loader, config):
    # Create quantization trainer
    trainer = QuantizationAwareTrainer(model, config, device)
    
    # Calibrate on a subset of data
    trainer.calibrate_model(train_loader, num_batches=50)
    
    # Fine-tune with quantization
    history = trainer.fine_tune_quantized_model(
        train_loader,
        val_loader,
        num_epochs=config['epochs'],
        learning_rate=config['learning_rate']
    )
    
    return trainer.quantized_model, history
```

### Model Conversion

```python
# Convert existing model to quantized version
def convert_model_for_inference(model_path, output_path, quant_config='8bit_symmetric'):
    # Load model
    checkpoint = torch.load(model_path)
    model = create_model_from_checkpoint(checkpoint)
    
    # Quantize
    config = get_quantization_config(quant_config)
    quantized_model = quantize_model(model, config)
    
    # Save quantized model
    torch.save({
        'state_dict': quantized_model.state_dict(),
        'config': checkpoint.get('config', {}),
        'quantization_config': config.to_dict()
    }, output_path)
```

## Performance Optimization

### Memory Optimization

```python
# Freeze quantization parameters after calibration
from wedefense.frontend.wav2vec2 import freeze_quantization_parameters

trainer.calibrate_model(train_dataloader)
freeze_quantization_parameters(trainer.quantized_model)

# Continue training with frozen quantization
```

### Inference Optimization

```python
# Set model to evaluation mode for deterministic quantization
quantized_model.eval()

# Use torch.jit.script for further optimization
scripted_model = torch.jit.script(quantized_model)
```

## Troubleshooting

### Common Issues

1. **Large quantization error**: Increase calibration batches or use gradual quantization
2. **Training instability**: Reduce learning rate or use symmetric quantization
3. **Memory issues**: Use per-channel quantization or reduce batch size

### Debugging

```python
# Check quantization statistics
stats = get_quantization_stats(quantized_model)
for module_name, error in stats['quantization_errors'].items():
    if error > 0.1:  # High quantization error
        print(f"High quantization error in {module_name}: {error}")

# Check effective bits
for module_name, bits in stats['effective_bits'].items():
    if bits < 4:  # Very low effective bits
        print(f"Low effective bits in {module_name}: {bits}")
```

## Examples

See `examples/lsq_quantization_example.py` for complete working examples including:

- Basic model quantization
- Quantization-aware training
- Different quantization configurations
- Performance benchmarking
- Integration with WeDefense pipeline

## References

- [LSQ: Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)
- [WeDefense Paper](https://arxiv.org/abs/2508.16232)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)

## API Reference

### Core Classes

- `LSQQuantizer`: Main quantization module
- `QuantizedLinear`: Quantized linear layer
- `QuantizedConv1d`: Quantized 1D convolution layer
- `QuantizationConfig`: Configuration management
- `QuantizationAwareTrainer`: Training integration

### Utility Functions

- `quantize_model()`: Quantize entire model
- `get_quantization_stats()`: Get quantization statistics
- `compare_model_sizes()`: Compare original vs quantized model sizes
- `create_gradual_quantization_schedule()`: Create gradual quantization schedule
