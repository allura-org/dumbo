from importlib import import_module
from msgspec import Struct
from typing import Any, Callable, List

from .logger import get_logger
from .result import Result, Ok, Err

logger = get_logger()

class BasePlugin:
    config_key: str
    provides: list[str] = []
    requires: list[str] = []
    conflicts: list[str] = []

    def __init__(self) -> None:
        pass

    def config_extension(self) -> Struct | None:
        return None

    def hooks(self) -> dict[str, Callable]:
        return {}

Model = Any  # slightly better type hints
Tokenizer = Any  # slightly better type hints
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

class DatasetLoaderPlugin(BasePlugin):
    config_key = "datasets"

    def __init__(self) -> None:
        self.provides += ["dataset"]

    def hooks(self) -> dict[str, Callable]:
        return {
            "dataset_loader": self.load_dataset,
        }

    def load_dataset(self, config: list[dict[str, Any]]) -> Result[Any]:
        ...


class DatasetFormatterPlugin(BasePlugin):
    provides = ["dataset"]

    def hooks(self) -> dict[str, Callable]:
        return {
            "dataset_formatter": self.format_dataset,
        }

    def format_dataset(self, dataset: Any, config: list[dict[str, Any]]) -> Result[Any]:
        ...


class TrainerPlugin(BasePlugin):
    config_key = "trl"

    def hooks(self) -> dict[str, Callable]:
        return {
            "trainer": self.train,
        }

    def train(self, model: Model, tokenizer: Tokenizer, dataset: Any, config: dict[str, Any]) -> Result[Any]:
        ...

def import_plugin(name: str) -> Result[List[BasePlugin]]:
    try:
        try:
            module = import_module(f"dumbo.plugins.{name}")
        except ImportError:
            module = import_module(name)

        plugin_classes = getattr(module, "AVAILABLE_PLUGINS", None)
        if not plugin_classes:
            logger.error(f"Plugin module `{name}` is missing AVAILABLE_PLUGINS")
            return Err(Exception(f"No plugins found in {name}"))

        plugins = [cls() for cls in plugin_classes]
        logger.info(f"Successfully imported plugin `{name}`.")
        return Ok(plugins)
    except ImportError as e:
        logger.error(
            f"Failed to import plugin `{name}`! Could not find built-in plugin or external plugin `{name}`."
        )
        return Err(e)

