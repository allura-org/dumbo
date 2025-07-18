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
        model = AutoModelForCausalLM.from_pretrained(config["base_model"])
        
        # Store tokenizer config for potential embedding resizing
        self.tokenizer_config = config.get("tokenizer", {})
        return Ok(model)
    
    def resize_embeddings(self, model: Model, tokenizer: Tokenizer) -> Result[Model]:
        """Resize model embeddings if tokenizer has new tokens"""
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        return Ok(model)


class TransformersTokenizerLoaderPlugin(TokenizerLoaderPlugin):
    provides = ["transformers_tokenizer"]

    def load_tokenizer(self, config: dict[str, Any], model=None) -> Result[Tokenizer]:
        """Load tokenizer with optional model for embedding resizing"""
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        
        # Handle tokenizer configuration
        tokenizer_config = config.get("tokenizer", {})
        
        # Track if we need to resize embeddings
        original_vocab_size = len(tokenizer)
        
        # Configure pad token
        pad_token = tokenizer_config.get("pad_token")
        if pad_token:
            if pad_token in tokenizer.get_vocab():
                tokenizer.pad_token = pad_token
            else:
                tokenizer.add_special_tokens({"pad_token": pad_token})
        elif tokenizer.pad_token is None:
            # Default: use eos token as pad token if pad token doesn't exist
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure other special tokens
        for token_type in ["eos_token", "bos_token", "unk_token"]:
            token_value = tokenizer_config.get(token_type)
            if token_value:
                if token_value in tokenizer.get_vocab():
                    setattr(tokenizer, token_type, token_value)
                else:
                    tokenizer.add_special_tokens({token_type: token_value})
        
        # Add additional special tokens
        additional_special_tokens = tokenizer_config.get("additional_special_tokens", [])
        if additional_special_tokens:
            new_tokens = [t for t in additional_special_tokens if t not in tokenizer.get_vocab()]
            if new_tokens:
                tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        
        # Resize model embeddings if tokens were added
        new_vocab_size = len(tokenizer)
        if new_vocab_size > original_vocab_size and model is not None:
            model.resize_token_embeddings(new_vocab_size)
        
        return Ok(tokenizer)

AVAILABLE_PLUGINS = [
    TransformersModelLoaderPlugin,
    TransformersTokenizerLoaderPlugin,
]
