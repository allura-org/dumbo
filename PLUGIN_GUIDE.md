# Dumbo Plugin Development Guide

Dumbo uses a plugin-based architecture where all functionality is provided through plugins. This guide explains how to create and use plugins.

## Plugin System Overview

### How Plugins Work

1. **Discovery**: Plugins are loaded from `src/dumbo/plugins/` or external modules
2. **Registration**: Each plugin module defines `AVAILABLE_PLUGINS` - a list of plugin classes
3. **Hooks**: Plugins provide functionality through hook functions registered in `hooks()`
4. **Integration**: The framework calls hooks at appropriate times during training

### Plugin Types

- **ModelLoaderPlugin**: Loads models (`transformers_model`, etc.)
- **TokenizerLoaderPlugin**: Loads tokenizers (`transformers_tokenizer`, etc.)
- **ModelPatcherPlugin**: Modifies models (Liger kernel, LoRA, etc.)
- **LoggingPlugin**: Provides logging capabilities (wandb, tensorboard, etc.)
- **Custom Plugins**: Any functionality through hook system

## Creating a Plugin

### 1. Plugin Structure

```python
from dumbo.result import Result, Ok
from dumbo.plugin_loader import [PluginType]

class MyPlugin([PluginType]):
    config_key = "my_plugin"  # YAML config key
    provides = ["my_feature"]  # Capabilities this plugin provides
    requires = []  # Capabilities this plugin needs
    conflicts = []  # Capabilities this plugin conflicts with
    
    def hooks(self) -> dict[str, Callable]:
        return {
            "hook_name": self.my_hook_function,
            # ... other hooks
        }
    
    def my_hook_function(self, *args) -> Result[Any]:
        # Implementation
        return Ok(result)

# Required: List of plugin classes in this module
AVAILABLE_PLUGINS = [MyPlugin]
```

### 2. Plugin Base Classes

#### ModelLoaderPlugin
Loads models and provides model functionality.

```python
from dumbo.plugin_loader import ModelLoaderPlugin

class MyModelLoader(ModelLoaderPlugin):
    config_key = "model"
    provides = ["my_model_type"]
    
    def load_model(self, config: dict[str, Any]) -> Result[Model]:
        # Load and return model
        return Ok(model)
```

#### TokenizerLoaderPlugin
Loads tokenizers with optional model reference.

```python
from dumbo.plugin_loader import TokenizerLoaderPlugin

class MyTokenizerLoader(TokenizerLoaderPlugin):
    config_key = "model"
    provides = ["my_tokenizer_type"]
    
    def load_tokenizer(self, config: dict[str, Any], model=None) -> Result[Tokenizer]:
        # Load and return tokenizer
        return Ok(tokenizer)
```

#### ModelPatcherPlugin
Modifies or patches existing models.

```python
from dumbo.plugin_loader import ModelPatcherPlugin

class MyModelPatcher(ModelPatcherPlugin):
    provides = ["model"]
    
    def patch_model(self, model: Model, config: dict[str, Any]) -> Result[Model]:
        # Modify model and return
        return Ok(modified_model)
```

#### LoggingPlugin
Provides logging capabilities throughout training.

```python
from dumbo.plugin_loader import LoggingPlugin

class MyLoggingPlugin(LoggingPlugin):
    config_key = "my_logger"
    provides = ["logging"]
    
    def initialize(self, config: dict[str, Any]) -> Result[None]:
        # Initialize logging system
        return Ok(None)
    
    def log_metrics(self, metrics: dict[str, Any], step: int) -> Result[None]:
        # Log metrics
        return Ok(None)
    
    def log_training_start(self, config: dict[str, Any]) -> Result[None]:
        # Log training start
        return Ok(None)
    
    def finish(self) -> Result[None]:
        # Cleanup logging
        return Ok(None)
```

### 3. Hook Reference

#### LoggingPlugin Hooks

| Hook Name | When Called | Parameters |
|-----------|-------------|------------|
| `log_init` | Before any logging | `config: dict[str, Any]` |
| `log_model` | After model loading | `model_info: dict[str, Any]` |
| `log_dataset` | After dataset loading | `dataset_info: dict[str, Any]` |
| `log_hyperparameters` | After config loading | `hparams: dict[str, Any]` |
| `log_training_start` | Before training begins | `config: dict[str, Any]` |
| `log_training_end` | After training completes | `summary: dict[str, Any]` |
| `log_step` | During training | `step_info: dict[str, Any]` |
| `log_metrics` | When metrics are ready | `metrics: dict[str, Any], step: int` |
| `finish` | Cleanup phase | None |

### 4. Configuration

Plugins are configured in YAML files under their `config_key`:

```yaml
# Example configuration
my_plugin:
    parameter1: value1
    parameter2: value2

plugins:
    - my_plugin  # Load the plugin
```

## Example: Creating a Logging Plugin

### 1. Create the Plugin File

Create `src/dumbo/plugins/my_logger.py`:

```python
"""My custom logging plugin."""

import json
from dumbo.result import Result, Ok
from dumbo.plugin_loader import LoggingPlugin
from dumbo.logger import get_logger

logger = get_logger()

class MyLoggingPlugin(LoggingPlugin):
    config_key = "my_logger"
    provides = ["logging", "my_logging"]
    
    def __init__(self):
        super().__init__()
        self.log_file = None
    
    def initialize(self, config: dict[str, Any]) -> Result[None]:
        """Initialize logging to file."""
        log_path = config.get("log_path", "training.log")
        self.log_file = open(log_path, "w")
        logger.info(f"Initialized logging to {log_path}")
        return Ok(None)
    
    def log_metrics(self, metrics: dict[str, Any], step: int) -> Result[None]:
        """Log metrics to file."""
        if self.log_file:
            self.log_file.write(f"Step {step}: {json.dumps(metrics)}\n")
            self.log_file.flush()
        return Ok(None)
    
    def log_training_start(self, config: dict[str, Any]) -> Result[None]:
        """Log training start."""
        if self.log_file:
            self.log_file.write("Training started\n")
            self.log_file.write(f"Config: {json.dumps(config)}\n")
        return Ok(None)
    
    def finish(self) -> Result[None]:
        """Close log file."""
        if self.log_file:
            self.log_file.close()
        return Ok(None)

# Required for plugin discovery
AVAILABLE_PLUGINS = [MyLoggingPlugin]
```

### 2. Use the Plugin

```yaml
# config.yaml
model:
    base_model: roneneldan/TinyStories-1M

my_logger:
    log_path: "my_training.log"

plugins:
    - transformers
    - transformers_trainer
    - my_logger
```

## Advanced Plugin Features

### Dependencies

Specify what your plugin needs or conflicts with:

```python
class MyPlugin(LoggingPlugin):
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

### Multiple Hooks

A single plugin can provide multiple hooks:

```python
def hooks(self) -> dict[str, Callable]:
    return {
        "log_init": self.initialize,
        "log_metrics": self.log_metrics,
        "custom_hook": self.custom_function,
    }
```

## Testing Your Plugin

### 1. Basic Test

```python
# test_my_plugin.py
from dumbo.plugins.my_plugin import MyLoggingPlugin

plugin = MyLoggingPlugin()
result = plugin.initialize({"log_path": "test.log"})
assert result.is_ok()

result = plugin.log_metrics({"loss": 0.5}, step=1)
assert result.is_ok()

result = plugin.finish()
assert result.is_ok()
```

### 2. Integration Test

```yaml
# test_config.yaml
model:
    base_model: roneneldan/TinyStories-1M

my_logger:
    log_path: "test_run.log"

plugins:
    - transformers
    - transformers_trainer
    - my_logger
```

Run with: `uv run dumbo test_config.yaml`

## Best Practices

1. **Always return `Result` types** from hook functions
2. **Handle errors gracefully** - don't crash the training process
3. **Use the logger** for debug/info messages
4. **Follow naming conventions** for config keys and hook names
5. **Document your plugin** with clear docstrings
6. **Test thoroughly** before using in production
7. **Check for conflicts** with other plugins
8. **Keep dependencies minimal**

## Plugin Directory Structure

```
src/dumbo/plugins/
├── __init__.py          # Empty (plugins discovered via import)
├── my_plugin.py         # Your plugin
├── transformers.py      # Example: Model loading
├── wandb.py            # Example: Logging plugin
└── ...
```