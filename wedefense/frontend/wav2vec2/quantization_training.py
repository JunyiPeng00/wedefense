"""Quantization-aware training utilities for WeDefense.

This module provides utilities for integrating quantization into the training process,
including quantization-aware training (QAT) and fine-tuning of quantized models.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .quantization_utils import (
    QuantizationConfig,
    create_quantization_aware_training_hooks,
    freeze_quantization_parameters,
    get_quantization_config,
    quantize_model,
    unfreeze_quantization_parameters
)


class QuantizationAwareTrainer:
    """Trainer for quantization-aware training (QAT).
    
    This class provides utilities for training models with quantization,
    including gradual quantization, calibration, and fine-tuning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        device: torch.device = torch.device('cpu')
    ):
        """Initialize the QAT trainer.
        
        Args:
            model: Model to train
            config: Quantization configuration
            device: Device to run training on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.quantization_hooks = []
        
        # Quantize the model
        self.quantized_model = quantize_model(model, config, inplace=False)
        self.quantized_model = self.quantized_model.to(device)
    
    def setup_quantization_hooks(self) -> None:
        """Setup quantization hooks for training."""
        self.quantization_hooks = create_quantization_aware_training_hooks(
            self.quantized_model, self.config
        )
    
    def remove_quantization_hooks(self) -> None:
        """Remove quantization hooks."""
        for hook in self.quantization_hooks:
            hook.remove()
        self.quantization_hooks = []
    
    def calibrate_model(
        self,
        dataloader: DataLoader,
        num_batches: int = 100,
        freeze_quantization: bool = True
    ) -> None:
        """Calibrate quantization parameters using calibration data.
        
        Args:
            dataloader: DataLoader for calibration data
            num_batches: Number of batches to use for calibration
            freeze_quantization: Whether to freeze quantization parameters after calibration
        """
        self.quantized_model.eval()
        
        if freeze_quantization:
            freeze_quantization_parameters(self.quantized_model)
        else:
            unfreeze_quantization_parameters(self.quantized_model)
        
        # Run calibration
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if len(batch) > 1 else None
                elif torch.is_tensor(batch):
                    inputs = batch.to(self.device)
                    targets = None
                else:
                    continue
                
                # Forward pass to update quantization parameters
                _ = self.quantized_model(inputs)
    
    def fine_tune_quantized_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, List[float]]:
        """Fine-tune the quantized model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            scheduler: Learning rate scheduler (optional)
            criterion: Loss function (optional)
            
        Returns:
            Dictionary containing training history
        """
        # Setup quantization hooks
        self.setup_quantization_hooks()
        
        # Setup optimizer
        optimizer = optim.AdamW(
            self.quantized_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            self.quantized_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_dataloader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.quantized_model(inputs)
                
                if targets is not None:
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Calculate accuracy
                    if outputs.dim() > 1:
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += targets.size(0)
                        train_correct += (predicted == targets).sum().item()
            
            # Validation phase
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            if val_dataloader is not None:
                self.quantized_model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        # Move batch to device
                        if isinstance(batch, (list, tuple)):
                            inputs = batch[0].to(self.device)
                            targets = batch[1].to(self.device)
                        else:
                            inputs = batch.to(self.device)
                            targets = None
                        
                        outputs = self.quantized_model(inputs)
                        
                        if targets is not None:
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                            
                            # Calculate accuracy
                            if outputs.dim() > 1:
                                _, predicted = torch.max(outputs.data, 1)
                                val_total += targets.size(0)
                                val_correct += (predicted == targets).sum().item()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Record history
            avg_train_loss = train_loss / len(train_dataloader)
            train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            if val_dataloader is not None:
                avg_val_loss = val_loss / len(val_dataloader)
                val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
                
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Remove quantization hooks
        self.remove_quantization_hooks()
        
        return history
    
    def evaluate_quantized_model(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """Evaluate the quantized model.
        
        Args:
            dataloader: Data loader for evaluation
            criterion: Loss function (optional)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.quantized_model.eval()
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                outputs = self.quantized_model(inputs)
                
                if targets is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    if outputs.dim() > 1:
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def get_quantization_statistics(self) -> Dict[str, Any]:
        """Get quantization statistics for the model.
        
        Returns:
            Dictionary containing quantization statistics
        """
        from .quantization_utils import get_quantization_stats
        
        return get_quantization_stats(self.quantized_model)


def create_gradual_quantization_schedule(
    total_epochs: int,
    quantization_start_epoch: int = 0,
    quantization_warmup_epochs: int = 5
) -> List[bool]:
    """Create a schedule for gradual quantization.
    
    Args:
        total_epochs: Total number of training epochs
        quantization_start_epoch: Epoch to start quantization
        quantization_warmup_epochs: Number of epochs for quantization warmup
        
    Returns:
        List of boolean values indicating whether to quantize in each epoch
    """
    schedule = []
    
    for epoch in range(total_epochs):
        if epoch < quantization_start_epoch:
            # No quantization
            schedule.append(False)
        elif epoch < quantization_start_epoch + quantization_warmup_epochs:
            # Gradual quantization warmup
            warmup_progress = (epoch - quantization_start_epoch) / quantization_warmup_epochs
            # Use probability-based quantization during warmup
            schedule.append(torch.rand(1).item() < warmup_progress)
        else:
            # Full quantization
            schedule.append(True)
    
    return schedule


def quantize_model_with_gradual_schedule(
    model: nn.Module,
    config: QuantizationConfig,
    schedule: List[bool],
    current_epoch: int
) -> nn.Module:
    """Apply quantization based on gradual schedule.
    
    Args:
        model: Model to quantize
        config: Quantization configuration
        schedule: Gradual quantization schedule
        current_epoch: Current training epoch
        
    Returns:
        Model with quantization applied if scheduled
    """
    if current_epoch < len(schedule) and schedule[current_epoch]:
        return quantize_model(model, config, inplace=True)
    else:
        return model


# Example usage and testing functions
def test_quantization_integration():
    """Test the quantization integration with a simple model."""
    import torch.nn as nn
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(100, 50)
            self.linear2 = nn.Linear(50, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    # Create model and config
    model = TestModel()
    config = get_quantization_config('8bit_symmetric')
    
    # Test quantization
    quantized_model = quantize_model(model, config)
    
    # Test forward pass
    x = torch.randn(32, 100)
    output = quantized_model(x)
    
    print(f"Original model output shape: {output.shape}")
    print(f"Quantized model has {len([m for m in quantized_model.modules() if hasattr(m, 'weight_quantizer')])} quantized layers")
    
    return quantized_model


if __name__ == "__main__":
    # Run test
    test_quantization_integration()
