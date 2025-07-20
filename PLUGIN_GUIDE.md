# Dumbo Plugin Development Guide

Dumbo uses a plugin-based architecture where all functionality is provided through plugins. This guide explains how to create and use plugins.

## Plugin System Overview

### How Plugins Work

1. **Discovery**: Plugins are loaded from `src/dumbo/plugins/` or external modules
2. **Registration**: Each plugin module defines `AVAILABLE_PLUGINS` - a list of plugin classes
3. **Hooks**: Plugins provide functionality through hook functions registered in `hooks()`
4. **Integration**: The framework calls hooks at appropriate times during training
5. **Metrics**: Plugins can provide metrics collectors via `get_metrics_collector()` method

### Plugin Types

- **ModelLoaderPlugin**: Loads models (`transformers_model`, etc.)
- **TokenizerLoaderPlugin**: Loads tokenizers (`transformers_tokenizer`, etc.)
- **ModelPatcherPlugin**: Modifies models (Liger kernel, LoRA, etc.)
- **LoggingPlugin**: Provides logging capabilities (wandb, tensorboard, etc.)
- **MetricsProvider**: Provides metrics collection via `MetricsCollector`
- **Custom Plugins**: Any functionality through hook system

## Plugin Base Classes

### BasePlugin
All plugins inherit from `BasePlugin`:

```python
from dumbo.plugin_loader import BasePlugin

class MyPlugin(BasePlugin):
    config_key = "my_plugin"  # YAML config key
    provides = ["my_feature"]  # Capabilities this plugin provides
    requires = []  # Capabilities this plugin needs
    conflicts = []  # Capabilities this plugin conflicts with
    
    def hooks(self) -> dict[str, Callable]:
        return {}  # Hook functions
```

### ModelLoaderPlugin
Loads models and provides model functionality.

```python
from dumbo.plugin_loader import ModelLoaderPlugin
from dumbo.result import Result, Ok

class MyModelLoader(ModelLoaderPlugin):
    config_key = "model"
    provides = ["my_model_type"]
    
    def load_model(self, config: dict[str, Any]) -> Result[Model]:
        # Load and return model
        return Ok(model)
```

### TokenizerLoaderPlugin
Loads tokenizers with optional model reference for embedding resizing.

```python
from dumbo.plugin_loader import TokenizerLoaderPlugin
from dumbo.result import Result, Ok

class MyTokenizerLoader(TokenizerLoaderPlugin):
    config_key = "model"
    provides = ["my_tokenizer_type"]
    
    def load_tokenizer(self, config: dict[str, Any], model=None) -> Result[Tokenizer]:
        # Load and return tokenizer
        return Ok(tokenizer)
```

### ModelPatcherPlugin
Modifies or patches existing models.

```python
from dumbo.plugin_loader import ModelPatcherPlugin
from dumbo.result import Result, Ok

class MyModelPatcher(ModelPatcherPlugin):
    provides = ["model"]
    
    def patch_model(self, model: Model, config: dict[str, Any]) -> Result[Model]:
        # Modify model and return
        return Ok(modified_model)
```

### LoggingPlugin
Provides logging capabilities throughout training with metrics integration.

```python
from dumbo.plugin_loader import LoggingPlugin
from dumbo.result import Result, Ok

class MyLoggingPlugin(LoggingPlugin):
    config_key = "my_logger"
    provides = ["logging", "metrics_collector"]
    
    def initialize(self, config: dict[str, Any]) -> Result[None]:
        # Initialize logging system
        return Ok(None)
    
    def log_metrics(self, metrics: dict[str, Any], step: int) -> Result[None]:
        # Log metrics
        return Ok(None)
    
    def finish(self) -> Result[None]:
        # Cleanup logging
        return Ok(None)
```

## Metrics Integration

### MetricsCollector Interface
Plugins can provide metrics collectors by implementing the abstract `MetricsCollector` class:

```python
from dumbo.metrics import MetricsCollector, MetricEvent
from dumbo.result import Result, Ok

class MyMetricsCollector(MetricsCollector):
    def __init__(self, config):
        self.config = config
    
    def log_metric(self, event: MetricEvent) -> None:
        # Log a single metric
        pass
    
    def log_metrics(self, events: List[MetricEvent]) -> None:
        # Log multiple metrics
        pass
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        # Log hyperparameters
        pass
    
    def log_model_info(self, info: Dict[str, Any]) -> None:
        # Log model information
        pass
    
    def finalize(self) -> None:
        # Cleanup
        pass

class MyPlugin(LoggingPlugin):
    config_key = "my_plugin"
    provides = ["logging", "metrics_collector"]
    
    def get_metrics_collector(self) -> Result[Any]:
        return Ok(MyMetricsCollector(self.config))
```

### Using the Metrics Registry
The framework automatically registers metrics collectors from plugins:

```python
from dumbo.metrics import get_metrics_registry

# In your plugin
registry = get_metrics_registry()
collector = MyMetricsCollector(config)
registry.register(collector)
```

## Hook System

### Hook Reference

| Hook Name | When Called | Parameters | Return Type |
|-----------|-------------|------------|-------------|
| `model_loader` | Load model | `config: dict[str, Any]` | `Result[Model]` |
| `tokenizer_loader` | Load tokenizer | `config: dict[str, Any], model=None` | `Result[Tokenizer]` |
| `model_patcher` | Patch model | `model: Model, config: dict[str, Any]` | `Result[Model]` |
| `log_init` | Initialize logging | `config: dict[str, Any]` | `Result[None]` |
| `log_model` | Log model info | `model_info: dict[str, Any]` | `Result[None]` |
| `log_dataset` | Log dataset info | `dataset_info: dict[str, Any]` | `Result[None]` |
| `log_hyperparameters` | Log hyperparameters | `hparams: dict[str, Any]` | `Result[None]` |
| `log_training_start` | Training start | `config: dict[str, Any]` | `Result[None]` |
| `log_training_end` | Training end | `summary: dict[str, Any]` | `Result[None]` |
| `log_step` | Individual step | `step_info: dict[str, Any]` | `Result[None]` |
| `log_metrics` | Log metrics | `metrics: dict[str, Any], step: int` | `Result[None]` |
| `metrics_collector` | Get collector | None | `Result[MetricsCollector]` |
| `finish` | Cleanup | None | `Result[None]` |

### Hook Implementation

Plugins can provide hooks via the `hooks()` method:

```python
from typing import Callable
from dumbo.result import Result, Ok

class MyPlugin(BasePlugin):
    def hooks(self) -> dict[str, Callable]:
        return {
            "log_init": self.initialize,
            "log_metrics": self.log_metrics,
            "metrics_collector": self.get_metrics_collector,
            "finish": self.finish,
        }
```

## Configuration

Plugins are configured in YAML files under their `config_key`:

```yaml
# Example configuration
my_plugin:
    parameter1: value1
    parameter2: value2

plugins:
    - my_plugin  # Load the plugin
```

## Advanced Features

### Plugin Dependencies
Specify what your plugin needs or conflicts with:

```python
class MyPlugin(BasePlugin):
    config_key = "my_plugin"
    provides = ["logging", "advanced_logging"]
    requires = ["model"]  # Needs model to be loaded first
    conflicts = ["basic_logging"]  # Doesn't work with basic logging
```

### Error Handling
Always return `Result` types for proper error handling:

```python
from dumbo.result import Result, Ok, Err

def my_hook(self, data: dict[str, Any]) -> Result[None]:
    try:
        # Do something
        return Ok(None)
    except Exception as e:
        logger.error(f"My plugin failed: {e}")
        return Err(e)
```

### External Plugin Loading
Plugins can be loaded from external modules:

```python
# In your external plugin module
class ExternalPlugin(BasePlugin):
    config_key = "external"
    provides = ["external_feature"]

AVAILABLE_PLUGINS = [ExternalPlugin]
```

Use in config:
```yaml
plugins:
    - my_external_plugin  # Will try to import my_external_plugin
```

## Complete Example: Logging Plugin with Metrics

```python
"""Complete logging plugin with metrics collection."""

import json
import time
from typing import Any, List
from dumbo.result import Result, Ok
from dumbo.plugin_loader import LoggingPlugin
from dumbo.metrics import MetricsCollector, MetricEvent, get_metrics_registry
from dumbo.logger import get_logger

logger = get_logger()

class FileMetricsCollector(MetricsCollector):
    """Metrics collector that logs to file."""
    
    def __init__(self, log_path: str):
        self.log_file = open(log_path, "w")
        self.start_time = time.time()
    
    def log_metric(self, event: MetricEvent) -> None:
        """Log a single metric event."""
        self.log_file.write(f"[{event.timestamp}] {event.name}: {event.value} (step {event.step})\n")
        self.log_file.flush()
    
    def log_metrics(self, events: List[MetricEvent]) -> None:
        """Log multiple metric events."""
        for event in events:
            self.log_metric(event)
    
    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.log_file.write(f"Hyperparameters: {json.dumps(params)}\n")
        self.log_file.flush()
    
    def log_model_info(self, info: dict[str, Any]) -> None:
        """Log model information."""
        self.log_file.write(f"Model info: {json.dumps(info)}\n")
        self.log_file.flush()
    
    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.log_file.close()

class FileLoggingPlugin(LoggingPlugin):
    config_key = "file_logger"
    provides = ["logging", "metrics_collector"]
    
    def __init__(self):
        super().__init__()
        self.config = None
        self.collector = None
    
    def initialize(self, config: dict[str, Any]) -> Result[None]:
        """Initialize file logging."""
        self.config = config
        log_path = config.get("log_path", "training.log")
        self.collector = FileMetricsCollector(log_path)
        
        # Register with global registry
        registry = get_metrics_registry()
        registry.register(self.collector)
        
        logger.info(f"Initialized file logging to {log_path}")
        return Ok(None)
    
    def get_metrics_collector(self) -> Result[Any]:
        """Get metrics collector."""
        return Ok(self.collector)
    
    def finish(self) -> Result[None]:
        """Cleanup file logging."""
        if self.collector:
            self.collector.finalize()
        return Ok(None)
    
    def hooks(self) -> dict[str, Any]:
        """Provide hooks for plugin system."""
        return {
            "log_init": self.initialize,
            "metrics_collector": self.get_metrics_collector,
            "finish": self.finish,
        }

# Required for plugin discovery
AVAILABLE_PLUGINS = [FileLoggingPlugin]
```

## Testing Your Plugin

### 1. Basic Test

```python
# test_my_plugin.py
from dumbo.plugins.my_plugin import FileLoggingPlugin
from dumbo.metrics import MetricEvent

plugin = FileLoggingPlugin()
result = plugin.initialize({"log_path": "test.log"})
assert result.is_ok()

# Test metrics collection
collector = plugin.get_metrics_collector().unwrap()
collector.log_metric(MetricEvent("loss", 0.5, 1, {}))

result = plugin.finish()
assert result.is_ok()
```

### 2. Integration Test

```yaml
# test_config.yaml
model:
    base_model: roneneldan/TinyStories-1M

file_logger:
    log_path: "test_run.log"

plugins:
    - transformers
    - transformers_trainer
    - file_logger
```

Run with: `uv run dumbo test_config.yaml`

## Plugin Directory Structure

```
src/dumbo/plugins/
├── __init__.py          # Empty (plugins discovered via import)
├── my_plugin.py         # Your plugin
├── transformers.py      # Model loading with metrics
├── transformers_trainer.py  # Training setup
├── liger.py            # Liger kernel optimizations
├── wandb.py            # Weights & Biases with metrics
├── polars.py           # Dataset loading
├── jinja_formatter.py  # Data formatting
└── trl.py              # TRL training
```

## Best Practices

1. **Always return `Result` types** from hook functions
2. **Handle errors gracefully** - don't crash the training process
3. **Use the logger** for debug/info messages
4. **Follow naming conventions** for config keys and hook names
5. **Document your plugin** with clear docstrings
6. **Test thoroughly** before using in production
7. **Check for conflicts** with other plugins
8. **Keep dependencies minimal**
9. **Use the metrics system** for consistent logging
10. **Handle cleanup properly** in `finalize()` methods