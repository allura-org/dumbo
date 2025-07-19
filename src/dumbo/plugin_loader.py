from importlib import import_module
from msgspec import Struct
from typing import Any, Callable

from .logger import get_logger
from .result import Result, Ok, Err

logger = get_logger()

class BasePlugin:
    config_key: str
    provides: list[str] = []
    requires: list[str] = []
    conflicts: list[str] = []

    def __init__(self):
        pass

    def config_extension(self) -> Struct | None:
        return None

    def hooks(self) -> dict[str, Callable]:
        return {}

Model = Any # slightly better type hints
Tokenizer = Any # slightly better type hints
class ModelLoaderPlugin(BasePlugin):
    config_key = "model"

    def __init__(self):
        self.provides += ["model"]

    def hooks(self) -> dict[str, Callable]:
        return {
            "model_loader": self.load_model,
        }
    
    def load_model(self, config: dict[str, Any]) -> Result[Model]:
        ...

class TokenizerLoaderPlugin(BasePlugin):
    config_key = "model"

    def __init__(self):
        self.provides += ["tokenizer"]

    def hooks(self) -> dict[str, Callable]:
        return {
            "tokenizer_loader": self.load_tokenizer,
        }
    
    def load_tokenizer(self, config: dict[str, Any]) -> Result[Tokenizer]:
        ...

class ModelPatcherPlugin(BasePlugin):
    provides = ["model"]
    
    def hooks(self) -> dict[str, Callable]:
        return {
            "model_patcher": self.patch_model,
        }
    
    def patch_model(self, model: Model, config: dict[str, Any]) -> Result[Model]:
        ...

class LoggingPlugin(BasePlugin):
    config_key = "logging"
    provides = ["logging"]
    
    def hooks(self) -> dict[str, Callable]:
        return {
            "log_init": self.initialize,
            "log_metrics": self.log_metrics,
            "log_model": self.log_model,
            "log_dataset": self.log_dataset,
            "log_training_start": self.log_training_start,
            "log_training_end": self.log_training_end,
            "log_step": self.log_step,
            "log_hyperparameters": self.log_hyperparameters,
            "finish": self.finish,
        }
    
    def initialize(self, config: dict[str, Any]) -> Result[None]:
        """Initialize logging system."""
        return Ok(None)
    
    def log_metrics(self, metrics: dict[str, Any], step: int) -> Result[None]:
        """Log training metrics."""
        return Ok(None)
    
    def log_model(self, model_info: dict[str, Any]) -> Result[None]:
        """Log model information."""
        return Ok(None)
    
    def log_dataset(self, dataset_info: dict[str, Any]) -> Result[None]:
        """Log dataset information."""
        return Ok(None)
    
    def log_training_start(self, config: dict[str, Any]) -> Result[None]:
        """Log training start."""
        return Ok(None)
    
    def log_training_end(self, summary: dict[str, Any]) -> Result[None]:
        """Log training completion."""
        return Ok(None)
    
    def log_step(self, step_info: dict[str, Any]) -> Result[None]:
        """Log individual training step."""
        return Ok(None)
    
    def log_hyperparameters(self, hparams: dict[str, Any]) -> Result[None]:
        """Log hyperparameters."""
        return Ok(None)
    
    def finish(self) -> Result[None]:
        """Cleanup logging system."""
        return Ok(None)

def import_plugin(name: str) -> Result[Any]:
    try:
        try:
            module = import_module(f"dumbo.plugins.{name}")
        except ImportError:
            module = import_module(name)
        logger.info(f"Successfully imported plugin `{name}`.")
        return Ok(module)
    except ImportError as e:
        logger.error(f"Failed to import plugin `{name}`! Could not find built-in plugin or external plugin `{name}`.")
        return Err(e)
