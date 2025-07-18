from dumbo.plugin_loader import BasePlugin
from dumbo.result import Result, Ok, Err
from typing import Any, Dict, Callable
from dumbo.logger import get_logger
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

logger = get_logger()

class TRLTrainingPlugin(BasePlugin):
    config_key = "trl"
    provides = ["trainer"]
    requires = ["model", "tokenizer", "dataset_loader"]
    
    def hooks(self) -> Dict[str, Any]:
        return {
            "trainer": self.create_trainer,
            "train": self.train_model,
        }
    
    def create_trainer(self, model: Any, tokenizer: Any, datasets: Any, config: Dict[str, Any]) -> Result[Any]:
        try:
            trl_config = config
            trainer_type = trl_config.get("trainer_type", "sft")
            
            if trainer_type != "sft":
                return Err(ValueError(f"Unsupported trainer type: {trainer_type}"))
            
            # Get training arguments
            train_args_config = trl_config.get("arguments", {})
            
            # Calculate gradient accumulation steps
            batch_size = train_args_config.get("batch_size", 16)
            physical_batch_size = train_args_config.get("physical_batch_size", 1)
            
            # Calculate based on available GPUs
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            grad_accum_steps = max(1, batch_size // (physical_batch_size * num_gpus))
            
            # Ensure learning_rate is a float
            learning_rate = float(train_args_config.get("learning_rate", 1e-4))
            num_epochs = int(train_args_config.get("num_epochs", 3))
            
            sft_config = SFTConfig(
                output_dir="./output",
                per_device_train_batch_size=physical_batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                fp16=torch.cuda.is_available(),
                bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
                logging_steps=10,
                save_strategy="epoch",
                eval_strategy="no",
                report_to=None,
                dataset_text_field="text",
                max_seq_length=512,
                **{k: v for k, v in train_args_config.items() if k not in ["batch_size", "physical_batch_size", "learning_rate", "num_epochs"]}
            )
            
            # Use the first dataset
            train_dataset = datasets[0] if datasets else None
            
            # Use the dataset as-is since formatting is handled by jinja_formatter
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )
            
            logger.info("Trainer created successfully")
            return Ok(trainer)
            
        except Exception as e:
            logger.error(f"Failed to create trainer: {e}")
            return Err(e)
    
    def train_model(self, trainer: Any, config: Dict[str, Any]) -> Result[Any]:
        try:
            logger.info("Starting training...")
            trainer.train()
            logger.info("Training completed successfully")
            return Ok(trainer)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return Err(e)

AVAILABLE_PLUGINS = [TRLTrainingPlugin]