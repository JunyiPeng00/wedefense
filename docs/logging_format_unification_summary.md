# Logging Format Unification Summary

## Overview

This document summarizes the unification of logging output formats between `get_hf_ssl_pruning.py` and `convert_hf_ssl_models.py` to ensure consistent output behavior across both modules.

## Changes Made

### Before Unification

**`get_hf_ssl_pruning.py`** used Python logging module:
```python
import logging

logger = setup_logger(__name__, logging.INFO)
logger.info(f"Starting model download and conversion for {self.upstream_name}.")
logger.info('Loaded pretrained ckpt to upstream: missing=%s, unexpected=%s', ...)
```

**`convert_hf_ssl_models.py`** used print statements:
```python
if is_rank_zero():
    print(f"Loading model: {model_id}")
if is_rank_zero():
    print(f"[{model_name}] Missing keys: {len(missing)}")
```

### After Unification

Both files now use **consistent print statements** with `is_rank_zero()` checks:

**`get_hf_ssl_pruning.py`** (updated):
```python
if is_rank_zero():
    print(f"Starting model download and conversion for {self.upstream_name}.")
if is_rank_zero():
    print(f'[{self.upstream_name}] Loaded pretrained ckpt to upstream: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')
```

**`convert_hf_ssl_models.py`** (unchanged):
```python
if is_rank_zero():
    print(f"Loading model: {model_id}")
if is_rank_zero():
    print(f"[{model_name}] Missing keys: {len(missing)}")
```

## Specific Changes in `get_hf_ssl_pruning.py`

### 1. Removed Logging Module Dependencies
```python
# REMOVED
import logging

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    # ... logger setup code ...

logger = setup_logger(__name__, logging.INFO)
```

### 2. Updated Output Statements

**Model Loading Messages:**
```python
# BEFORE
logger.info(f"Starting model download and conversion for {self.upstream_name}.")

# AFTER
if is_rank_zero():
    print(f"Starting model download and conversion for {self.upstream_name}.")
```

**Pre-converted Model Detection:**
```python
# BEFORE
logger.info(f"Path ends with {download_dir.split('.')[-1]}, assuming it is a pre-converted model: {download_dir}")

# AFTER
if is_rank_zero():
    print(f"Path ends with {download_dir.split('.')[-1]}, assuming it is a pre-converted model: {download_dir}")
```

**Pruning Units Information:**
```python
# BEFORE
logger.info(f'Enabled pruning units: {pruning_set}')

# AFTER
if is_rank_zero():
    print(f'Enabled pruning units: {pruning_set}')
```

**Model Loading Results:**
```python
# BEFORE
logger.info(
    'Loaded pretrained ckpt to upstream: missing=%s, unexpected=%s',
    result.missing_keys,
    result.unexpected_keys,
)

# AFTER
if is_rank_zero():
    print(f'[{self.upstream_name}] Loaded pretrained ckpt to upstream: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')
```

**Main Function Messages:**
```python
# BEFORE
logger.info('Initializing HuggingfaceFrontend with config: %s', upstream_config)
logger.info('Model initialized successfully.')
logger.info('Number of parameters: %d', net.get_num_params())
logger.info('Testing forward pass...')
logger.info('Output shape: %s', output.shape)

# AFTER
if is_rank_zero():
    print(f'Initializing HuggingfaceFrontend with config: {upstream_config}')
    print('Model initialized successfully.')
    print(f'Number of parameters: {net.get_num_params():,}')
    print('Testing forward pass...')
    print(f'Output shape: {output.shape}')
```

## Benefits of Unification

### 1. Consistent Output Format
- Both modules now use the same output mechanism
- Easier to read and debug across different components
- Uniform message formatting

### 2. Simplified Dependencies
- Removed complex logging setup
- No need for logger configuration
- Reduced import dependencies

### 3. Distributed Training Support
- All output is properly controlled by `is_rank_zero()`
- Clean output in multi-GPU environments
- Consistent behavior across modules

### 4. Better Performance
- Direct print statements are faster than logging
- No overhead from logger setup and formatting
- Reduced memory usage

## Output Format Consistency

Both modules now follow the same pattern:

```python
# Model loading
if is_rank_zero():
    print(f"Loading model: {model_id}")

# Progress messages
if is_rank_zero():
    print(f"[{model_name}] Starting conversion...")

# Results with counts
if is_rank_zero():
    print(f"[{model_name}] Missing keys: {len(missing)}")
    print(f"[{model_name}] Unexpected keys: {len(unexpected)}")

# Configuration information
if is_rank_zero():
    print(f'Enabled pruning units: {pruning_set}')
```

## Testing

### Manual Testing
```bash
# Test with different ranks
RANK=0 python -c "from wedefense.frontend.get_hf_ssl_pruning import main; main()"
RANK=1 python -c "from wedefense.frontend.get_hf_ssl_pruning import main; main()"
```

### Expected Behavior
- **RANK=0**: All print statements should be visible
- **RANK=1,2,3...**: All print statements should be suppressed
- **No RANK**: All print statements should be visible (non-distributed)

## Backward Compatibility

The changes maintain backward compatibility:
- Same functionality for model loading and conversion
- Same API for HuggingfaceFrontend class
- Same configuration options
- Only output format has changed (from logging to print)

## Future Considerations

### Potential Enhancements
1. **Structured Output**: Consider JSON or structured logging for machine-readable output
2. **Verbosity Levels**: Add verbosity control for different output levels
3. **Output Redirection**: Support for output redirection to files
4. **Timing Information**: Add timing information for performance monitoring

### Monitoring
- Monitor output behavior in production distributed environments
- Ensure no performance regressions from logging changes
- Verify output suppression works correctly across different PyTorch versions

## Conclusion

The logging format unification successfully standardizes output behavior across both SSL model conversion modules. This provides a cleaner, more consistent user experience while maintaining all existing functionality and improving performance in distributed training scenarios.
