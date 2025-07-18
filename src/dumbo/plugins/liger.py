from dumbo.plugin_loader import BasePlugin
from dumbo.result import Result, Ok
from typing import Any, Callable
from transformers import AutoModelForCausalLM
from dumbo.logger import get_logger

logger = get_logger()

try:
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
except ImportError:
    logger.warning("liger_kernel not available, using fallback")
    _apply_liger_kernel_to_instance = None

class LigerPlugin(BasePlugin):
    config_key = "liger"
    requires = ["transformers_model"]
    
    def hooks(self) -> dict[str, Callable]:
        return {
            "model_patcher": self.patch_model,
        }
    
    def patch_model(self, model: AutoModelForCausalLM, config: dict[str, Any] | None) -> Result[AutoModelForCausalLM]:
        logger.info("Patching model...")
        if _apply_liger_kernel_to_instance is not None and config:
            _apply_liger_kernel_to_instance(model, **config)
        else:
            logger.warning("Skipping liger kernel patches")
        return Ok(model)

AVAILABLE_PLUGINS = [LigerPlugin]
