from typing import Any, Callable

from dumbo.logger import get_logger
from dumbo.plugin_loader import BasePlugin, Model
from dumbo.result import Ok, Result

logger = get_logger()

try:
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
except Exception:  # pragma: no cover - optional dependency
    _apply_liger_kernel_to_instance = None  # type: ignore


class LigerPlugin(BasePlugin):
    config_key = "liger"
    requires = ["transformers_model"]

    def hooks(self) -> dict[str, Callable]:
        return {
            "model_patcher": self.patch_model,
        }

    def patch_model(self, model: Model, config: dict[str, Any] | None) -> Result[Model]:
        logger.info("Patching model...")
        if _apply_liger_kernel_to_instance is None:
            logger.warning("liger-kernel not installed; skipping patch")
            return Ok(model)
        _apply_liger_kernel_to_instance(model, **(config or {}))
        return Ok(model)

AVAILABLE_PLUGINS = [LigerPlugin]

