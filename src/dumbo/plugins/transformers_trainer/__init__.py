from dumbo.plugin_loader import BasePlugin
from dumbo.result import Result, Ok, Err
from typing import Any, Dict
from dumbo.logger import get_logger
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
import torch

logger = get_logger()

class TransformersTrainerPlugin(BasePlugin):
    config_key = "transformers"
    provides = ["trainer"]
    requires = ["model", "tokenizer", "dataset_loader"]
    
    def hooks(self) -> Dict[str, Any]:
        return {
            "trainer": self.create_trainer,
            "train": self.train_model,
        }
    
    def create_trainer(self, model: Any, tokenizer: Any, datasets: Any, config: Dict[str, Any]) -> Result[Any]:
        try:
            trl_config = config["trainer"]
            
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
            
            training_args = TrainingArguments(
                per_device_train_batch_size=physical_batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                fp16=torch.cuda.is_available(),
                bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
                eval_strategy="no",
                report_to="none",
                remove_unused_columns=False,
                **{k: v for k, v in train_args_config.items() if k not in ["batch_size", "physical_batch_size", "learning_rate", "num_epochs"]}
            )
            
            # Use the first dataset
            train_dataset = datasets[0] if datasets else None
            
            # Tokenize the dataset
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=512,
                    return_tensors=None
                )
            
            tokenized_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False  # We're doing causal LM, not masked LM
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                callbacks=self._get_callbacks(model, tokenizer)
            )
            
            return Ok(trainer)
            
        except Exception as e:
            logger.error(f"Failed to create trainer: {e}")
            return Err(e)
    
    def _get_callbacks(self, model, tokenizer):
        """Get list of callbacks including metrics adapter."""
        callbacks = []
        
        # Add the metrics adapter callback
        try:
            from dumbo.plugins.transformers_trainer.metrics_adapter import get_trainer_metrics_callback
            metrics_callback = get_trainer_metrics_callback()
            callbacks.append(metrics_callback)
            logger.info("Added metrics adapter callback to trainer")
        except Exception as e:
            logger.warning(f"Failed to add metrics adapter: {e}")
        
        return callbacks

    def train_model(self, trainer: Any, config: Dict[str, Any]) -> Result[Any]:
        try:
            trainer.train()
            return Ok(trainer)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return Err(e)

AVAILABLE_PLUGINS = [TransformersTrainerPlugin]
