from dumbo.plugin_loader import ModelPatcherPlugin
from dumbo.result import Result, Ok
from typing import Any, Callable
from transformers import AutoModelForCausalLM
from dumbo.logger import get_logger

logger = get_logger()

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    logger.warning("PEFT not available, LoRA patches will be skipped")
    PEFT_AVAILABLE = False

class PEFTPlugin(ModelPatcherPlugin):
    config_key = "peft"
    requires = ["transformers_model"]
    
    def hooks(self) -> dict[str, Callable]:
        return {
            "model_patcher": self.patch_model,
        }
    
    def patch_model(self, model: AutoModelForCausalLM, config: dict[str, Any] | None) -> Result[AutoModelForCausalLM]:
        logger.info("Applying PEFT LoRA patches...")
        
        if not PEFT_AVAILABLE:
            logger.warning("PEFT library not available, skipping LoRA patches")
            return Ok(model)
        
        if not config or not config.get("lora", {}).get("enabled", False):
            logger.info("LoRA not enabled in config, skipping patches")
            return Ok(model)
        
        lora_config = config["lora"]
        
        # Build LoraConfig from config parameters
        lora_kwargs = {
            "task_type": TaskType.CAUSAL_LM,
            "r": lora_config.get("r", 16),
            "lora_alpha": lora_config.get("lora_alpha", 32),
            "lora_dropout": lora_config.get("lora_dropout", 0.1),
            "bias": lora_config.get("bias", "none"),
            "target_modules": lora_config.get("target_modules", None),
            "modules_to_save": lora_config.get("modules_to_save", None),
        }
        
        # Filter out None values
        lora_kwargs = {k: v for k, v in lora_kwargs.items() if v is not None}
        
        # Handle target_modules string patterns
        if "target_modules" in lora_kwargs:
            target_modules = lora_kwargs["target_modules"]
            if isinstance(target_modules, str):
                # Handle common patterns like "all-linear", "q_proj,v_proj", etc.
                if target_modules == "all-linear":
                    lora_kwargs["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                elif "," in target_modules:
                    lora_kwargs["target_modules"] = [m.strip() for m in target_modules.split(",")]
                else:
                    lora_kwargs["target_modules"] = [target_modules]
        
        try:
            peft_config = LoraConfig(**lora_kwargs)
            model = get_peft_model(model, peft_config)
            
            # Log trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA applied: {trainable_params:,} trainable parameters out of {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
            
            return Ok(model)
            
        except Exception as e:
            logger.error(f"Failed to apply LoRA patches: {e}")
            return Ok(model)  # Return original model if patching fails

AVAILABLE_PLUGINS = [PEFTPlugin]