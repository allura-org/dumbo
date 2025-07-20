"""Adapter layer to connect transformers Trainer to abstract metrics system."""

from typing import Any, Dict, List, Optional
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from dumbo.metrics import MetricsRegistry, MetricEvent, get_metrics_registry
from dumbo.logger import get_logger

logger = get_logger()


class MetricsAdapterCallback(TrainerCallback):
    """Adapter that converts Trainer events to abstract metrics."""
    
    def __init__(self, metrics_registry: Optional[MetricsRegistry] = None):
        super().__init__()
        self.metrics_registry = metrics_registry or get_metrics_registry()
        self._step_counter = 0
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when trainer initialization ends."""
        # Log training args as hyperparameters
        hparams = {}
        for key, value in args.to_dict().items():
            if isinstance(value, (int, float, str, bool)):
                hparams[key] = value
            else:
                hparams[key] = str(value)
        
        self.metrics_registry.log_hyperparameters(hparams)
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when training begins."""
        model = kwargs.get('model')
        tokenizer = kwargs.get('tokenizer')
        
        if model and tokenizer:
            model_info = {
                "model_type": type(model).__name__,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else None,
            }
            self.metrics_registry.log_model_info(model_info)
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Called when logs are ready to be logged."""
        if logs is None:
            return
            
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second", 
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]
        
        events = []
        non_scalar_logs = {}
        
        for k, v in logs.items():
            if k in single_value_scalars:
                # These are summary metrics
                events.append(MetricEvent(
                    name=k,
                    value=float(v),
                    step=state.global_step,
                    tags={"type": "summary", "source": "trainer"}
                ))
            else:
                # Regular metrics
                if isinstance(v, (int, float)):
                    non_scalar_logs[k] = float(v)
        
        # Log regular metrics with basic rewriting
        for k, v in non_scalar_logs.items():
            # Basic log rewriting like transformers does
            if k.startswith("eval_"):
                # Evaluation metrics stay as-is
                name = "eval/" + k
            elif k.startswith("train_") or k in ["loss", "learning_rate", "epoch", "grad_norm"]:
                name = "train/" + k
            else:
                # Ensure consistent naming
                name = k
            
            events.append(MetricEvent(
                name=name,
                value=v,
                step=state.global_step,
                tags={"source": "trainer"}
            ))
        
        # Add global step as a metric
        events.append(MetricEvent(
            name="train/global_step",
            value=float(state.global_step),
            step=state.global_step,
            tags={"type": "system", "source": "trainer"}
        ))
        
        if events:
            logger.debug("Attempting to log metrics to registry")
            self.metrics_registry.log_metrics(events)
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):
        """Called after evaluation."""
        events = []
        for key, value in metrics.items():
            events.append(MetricEvent(
                name=f"eval/{key}",
                value=float(value),
                step=state.global_step,
                tags={"source": "evaluation"}
            ))
        
        if events:
            self.metrics_registry.log_metrics(events)
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when model is saved."""
        self.metrics_registry.log_metric(MetricEvent(
            name="checkpoint_saved",
            value=1.0,
            step=state.global_step,
            tags={"checkpoint": f"checkpoint-{state.global_step}"}
        ))
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when training ends."""
        summary_events = []
        
        if hasattr(state, 'global_step'):
            summary_events.append(MetricEvent(
                name="total_steps",
                value=float(state.global_step),
                step=state.global_step,
                tags={"type": "summary"}
            ))
        
        if hasattr(state, 'epoch') and state.epoch is not None:
            summary_events.append(MetricEvent(
                name="total_epochs",
                value=float(state.epoch),
                step=state.global_step,
                tags={"type": "summary"}
            ))
        
        if state.log_history:
            final_loss = state.log_history[-1].get("loss")
            if final_loss is not None:
                summary_events.append(MetricEvent(
                    name="final_loss",
                    value=float(final_loss),
                    step=state.global_step,
                    tags={"type": "summary"}
                ))
        
        if summary_events:
            self.metrics_registry.log_metrics(summary_events)
        
        # Finalize metrics collection
        self.metrics_registry.finalize()


def get_trainer_metrics_callback() -> MetricsAdapterCallback:
    """Get the trainer metrics callback."""
    return MetricsAdapterCallback()
