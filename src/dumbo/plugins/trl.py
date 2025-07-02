from typing import Any, Callable

from dumbo.logger import get_logger
from dumbo.plugin_loader import TrainerPlugin, Model, Tokenizer
from dumbo.result import Ok, Err, Result

try:
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
except Exception:  # pragma: no cover - optional dependency
    SFTTrainer = None  # type: ignore
    TrainingArguments = None  # type: ignore
    Dataset = None  # type: ignore

logger = get_logger()


class TRLPlugin(TrainerPlugin):
    """Train the model with `trl`'s SFT trainer if available."""

    def train(
        self,
        model: Model,
        tokenizer: Tokenizer,
        dataset: Any,
        config: dict[str, Any],
    ) -> Result[None]:
        if SFTTrainer is None or TrainingArguments is None or Dataset is None:
            logger.error("trl or transformers is not installed; skipping training")
            return Err(ImportError("trl or transformers not available"))

        if model is None or tokenizer is None or dataset is None:
            return Err(ValueError("model, tokenizer and dataset must not be None"))

        try:
            logger.info("Starting training with TRL...")
            args_cfg = config.get("arguments", {})
            training_args = TrainingArguments(
                output_dir="./outputs",
                num_train_epochs=1,
                **args_cfg,
            )

            if not isinstance(dataset, Dataset):
                dataset = Dataset.from_pandas(dataset.to_pandas())

            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=training_args,
            )
            trainer.train()
            logger.info("Training completed")
            return Ok(None)
        except Exception as e:  # pragma: no cover - runtime failures
            logger.error(f"Training failed: {e}")
            return Err(e)

AVAILABLE_PLUGINS = [TRLPlugin]

