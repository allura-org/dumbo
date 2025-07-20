"""Weights & Biases logging plugin for Dumbo training framework."""

import os
import json
import time
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass

import wandb
from dumbo.result import Result, Ok
from dumbo.plugin_loader import LoggingPlugin
from dumbo.metrics import MetricsCollector, MetricEvent
from dumbo.logger import get_logger

logger = get_logger()


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases integration."""
    
    project: str = "dumbo-training"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[list] = None
    notes: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    save_code: bool = True
    save_code_path: Optional[str] = None
    log_model: bool = False
    log_gradients: bool = False
    log_model_stats: bool = True
    step_metric: str = "step"
    resume: Union[bool, str] = False
    id: Optional[str] = None
    anonymous: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "training"
    offline: bool = False


class WandbMetricsCollector(MetricsCollector):
    """Weights & Biases metrics collector."""
    
    def __init__(self, wandb_run: Any):
        self.run = wandb_run
        self._model_info_logged = False
    
    def log_metric(self, event: MetricEvent) -> None:
        """Log a single metric event."""
        if not self.run:
            return
        
        try:
            wandb.log({event.name: event.value}, step=event.step)
        except Exception as e:
            logger.error(f"Failed to log metric {event.name}: {e}")
    
    def log_metrics(self, events: List[MetricEvent]) -> None:
        """Log multiple metric events."""
        if not self.run:
            return
        
        try:
            metrics = {event.name: event.value for event in events}
            step = events[0].step if events else 0
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if not self.run:
            return
        
        try:
            wandb.config.update(params)
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")
    
    def log_model_info(self, info: Dict[str, Any]) -> None:
        """Log model information."""
        if not self.run or self._model_info_logged:
            return
        
        try:
            wandb.log({"model_info": info})
            self._model_info_logged = True
        except Exception as e:
            logger.error(f"Failed to log model info: {e}")
    
    def finalize(self) -> None:
        """Finalize metrics collection."""
        if not self.run:
            return
        
        try:
            wandb.finish()
        except Exception as e:
            logger.error(f"Failed to finalize wandb: {e}")


class WandbLoggingPlugin(LoggingPlugin):
    """Weights & Biases logging plugin for Dumbo training framework."""
    
    config_key = "wandb"
    provides = ["logging", "wandb_logging", "metrics_collector"]
    
    def __init__(self):
        super().__init__()
        self.config: Optional[WandbConfig] = None
        self.run: Optional[wandb.sdk.wandb_run.Run] = None
        self._step_counter = 0
        self._collector: Optional[WandbMetricsCollector] = None
        
    def initialize(self, config: dict[str, Any]) -> Result[None]:
        """Initialize wandb logging system."""
        try:
            self.config = WandbConfig(**config)
            
            # Set environment variables
            if self.config.offline:
                os.environ["WANDB_MODE"] = "offline"
                
            if "api_key" in config:
                os.environ["WANDB_API_KEY"] = config["api_key"]
            
            # Initialize wandb run
            init_kwargs = {
                "project": self.config.project,
                "entity": self.config.entity,
                "name": self.config.name,
                "tags": self.config.tags or [],
                "notes": self.config.notes,
                "config": self._prepare_config(),
                "save_code": self.config.save_code,
                "resume": self.config.resume,
                "id": self.config.id,
                "anonymous": self.config.anonymous,
                "group": self.config.group,
                "job_type": self.config.job_type,
            }
            
            # Filter None values
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
            
            self.run = wandb.init(**init_kwargs)
            logger.info("Initialized Weights & Biases logging")
            
            return Ok(None)
            
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            return Ok(None)  # Continue without logging
    
    def _prepare_config(self) -> Dict[str, Any]:
        """Prepare configuration dictionary for wandb."""
        if not self.config:
            return {}
            
        config = self.config.config or {}
        
        # Add system info
        config.update({
            "framework": "dumbo",
            "plugin_version": "1.0.0",
            "timestamp": int(time.time()),
        })
        
        return config
    
    def log_metrics(self, metrics: dict[str, Any], step: int) -> Result[None]:
        """Log training metrics to wandb."""
        if not self.run:
            return Ok(None)
            
        try:
            wandb.log(metrics, step=step)
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            return Ok(None)
    
    def log_model(self, model_info: dict[str, Any]) -> Result[None]:
        """Log model information to wandb."""
        if not self.run:
            return Ok(None)
            
        try:
            wandb.log({"model_info": model_info})
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to log model info: {e}")
            return Ok(None)
    
    def log_dataset(self, dataset_info: dict[str, Any]) -> Result[None]:
        """Log dataset information to wandb."""
        if not self.run:
            return Ok(None)
            
        try:
            # Create a table for dataset info
            dataset_table = wandb.Table(
                columns=["key", "value"],
                data=[[k, str(v)] for k, v in dataset_info.items()]
            )
            wandb.log({"dataset_info": dataset_table})
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to log dataset info: {e}")
            return Ok(None)
    
    def log_training_start(self, config: dict[str, Any]) -> Result[None]:
        """Log training start to wandb."""
        if not self.run:
            return Ok(None)
            
        try:
            wandb.log({"training_start": {"timestamp": int(time.time())}})
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to log training start: {e}")
            return Ok(None)
    
    def log_training_end(self, summary: dict[str, Any]) -> Result[None]:
        """Log training completion to wandb."""
        if not self.run:
            return Ok(None)
            
        try:
            wandb.log({"training_end": {"timestamp": int(time.time()), **summary}})
            
            # Add to run summary
            for k, v in summary.items():
                wandb.run.summary[k] = v
                
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to log training end: {e}")
            return Ok(None)
    
    def log_step(self, step_info: dict[str, Any]) -> Result[None]:
        """Log individual training step to wandb."""
        if not self.run:
            return Ok(None)
            
        try:
            step = step_info.get("step", self._step_counter)
            metrics = step_info.get("metrics", {})
            
            wandb.log(metrics, step=step)
            self._step_counter = step + 1
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to log step: {e}")
            return Ok(None)
    
    def log_hyperparameters(self, hparams: dict[str, Any]) -> Result[None]:
        """Log hyperparameters to wandb."""
        if not self.run:
            return Ok(None)
            
        try:
            wandb.config.update(hparams)
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")
            return Ok(None)
    
    def finish(self) -> Result[None]:
        """Cleanup wandb logging system."""
        try:
            if self.run:
                wandb.finish()
                self.run = None
                logger.info("Finished Weights & Biases logging")
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to finish wandb: {e}")
            return Ok(None)
    
    def get_metrics_collector(self) -> Result[Any]:
        """Get a metrics collector instance."""
        try:
            if not self.run:
                return Ok(None)
            
            if not self._collector:
                self._collector = WandbMetricsCollector(self.run)
            
            return Ok(self._collector)
        except Exception as e:
            logger.error(f"Failed to create metrics collector: {e}")
            return Ok(None)
    
    def hooks(self) -> Dict[str, Any]:
        """Provide hooks for plugin system."""
        return {
            "log_init": self.initialize,
            "metrics_collector": self.get_metrics_collector,
        }


# Available plugins for this module
AVAILABLE_PLUGINS = [WandbLoggingPlugin]
