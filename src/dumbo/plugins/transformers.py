from transformers import AutoModelForCausalLM, AutoTokenizer
from dumbo.result import Result, Ok
from dumbo.plugin_loader import ModelLoaderPlugin, TokenizerLoaderPlugin
from dumbo.plugin_loader import Model, Tokenizer
from typing import Any
from dumbo.logger import get_logger

logger = get_logger()

class TransformersModelLoaderPlugin(ModelLoaderPlugin):
    provides = ["transformers_model"]

    def load_model(self, config: dict[str, Any]) -> Result[Model]:
        logger.info(f"Loading model from {config['base_model']}...")
        model = AutoModelForCausalLM.from_pretrained(config["base_model"])
        logger.info("Model loaded successfully")
        return Ok(model)


class TransformersTokenizerLoaderPlugin(TokenizerLoaderPlugin):
    provides = ["transformers_tokenizer"]

    def load_tokenizer(self, config: dict[str, Any]) -> Result[Tokenizer]:
        logger.info(f"Loading tokenizer from {config['base_model']}...")
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        return Ok(tokenizer)

AVAILABLE_PLUGINS = [
    TransformersModelLoaderPlugin,
    TransformersTokenizerLoaderPlugin,
]