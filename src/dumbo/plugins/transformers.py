from typing import Any

from dumbo.logger import get_logger
from dumbo.plugin_loader import (
    Model,
    ModelLoaderPlugin,
    Tokenizer,
    TokenizerLoaderPlugin,
)
from dumbo.result import Ok, Err, Result

try:  # heavy dependency; import lazily if available
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

logger = get_logger()

class TransformersModelLoaderPlugin(ModelLoaderPlugin):
    provides = ["transformers_model"]

    def load_model(self, config: dict[str, Any]) -> Result[Model]:
        logger.info(f"Loading model from {config['base_model']}...")
        if AutoModelForCausalLM is None:
            logger.error("transformers is not installed")
            return Err(ImportError("transformers is not installed"))

        try:
            model = AutoModelForCausalLM.from_pretrained(
                config["base_model"],
                device_map="auto",
            )
            logger.info("Model loaded successfully")
            return Ok(model)
        except Exception as e:  # pragma: no cover - runtime failures
            logger.error(f"Failed to load model: {e}")
            return Err(e)


class TransformersTokenizerLoaderPlugin(TokenizerLoaderPlugin):
    provides = ["transformers_tokenizer"]

    def load_tokenizer(self, config: dict[str, Any]) -> Result[Tokenizer]:
        logger.info(f"Loading tokenizer from {config['base_model']}...")
        if AutoTokenizer is None:
            logger.error("transformers is not installed")
            return Err(ImportError("transformers is not installed"))

        try:
            tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
            return Ok(tokenizer)
        except Exception as e:  # pragma: no cover - runtime failures
            logger.error(f"Failed to load tokenizer: {e}")
            return Err(e)

AVAILABLE_PLUGINS = [
    TransformersModelLoaderPlugin,
    TransformersTokenizerLoaderPlugin,
]

