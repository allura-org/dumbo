from dumbo.plugin_loader import BasePlugin
from dumbo.result import Result
from typing import Any, Callable
from transformers import AutoModelForCausalLM
from dumbo.logger import get_logger

logger = get_logger()

from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

class LigerPlugin(BasePlugin):
    config_key = "liger"
    requires = ["transformers_model"]
    
    def hooks(self) -> dict[str, Callable]:
        return {
            "model_patcher": self.patch_model,
        }
    
    def patch_model(self, model: AutoModelForCausalLM, config: dict[str, Any] | None) -> Result[AutoModelForCausalLM]:
        logger.info("Patching model...")
        _apply_liger_kernel_to_instance(model, **config)
        return Ok(model)