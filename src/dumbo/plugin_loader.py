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

def import_plugin(name: str) -> Result[BasePlugin]:
    try:
        try:
            module = import_module(name)
        except ImportError:
            module = import_module(f"dumbo.plugins.{name}")
        logger.info(f"Successfully imported plugin `{name}`.")
        return Ok(module)
    except ImportError as e:
        logger.error(f"Failed to import plugin `{name}`! Could not find built-in plugin or external plugin `{name}`.")
        return Err(e)