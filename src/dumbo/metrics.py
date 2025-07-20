"""Abstract metrics collection system for Dumbo training framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from dumbo.logger import get_logger

logger = get_logger()


@dataclass
class MetricEvent:
    """Represents a single metric event."""
    name: str
    value: float
    step: int
    tags: Dict[str, Any]
    timestamp: Optional[float] = None


class MetricsCollector(ABC):
    """Abstract base class for metrics collection."""
    
    @abstractmethod
    def log_metric(self, event: MetricEvent) -> None:
        """Log a single metric event."""
        pass
    
    @abstractmethod
    def log_metrics(self, events: List[MetricEvent]) -> None:
        """Log multiple metric events."""
        pass
    
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass
    
    @abstractmethod
    def log_model_info(self, info: Dict[str, Any]) -> None:
        """Log model information."""
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """Finalize metrics collection."""
        pass


class MetricsRegistry:
    """Registry for metrics collectors."""
    
    def __init__(self):
        self._collectors: List[MetricsCollector] = []
    
    def register(self, collector: MetricsCollector) -> None:
        """Register a metrics collector."""
        logger.debug(f"Registered {collector.unwrap()} collector")
        self._collectors.append(collector.unwrap())
    
    def log_metric(self, event: MetricEvent) -> None:
        """Log a metric to all collectors."""
        for collector in self._collectors:
            try:
                collector.log_metric(event)
            except Exception as e:
                # Log error but don't fail
                pass
    
    def log_metrics(self, events: List[MetricEvent]) -> None:
        """Log multiple metrics to all collectors."""
        for collector in self._collectors:
            try:
                logger.debug(f"Attempting to log metrics to {collector}")
                collector.log_metrics(events)
            except Exception as e:
                # Log error but don't fail
                logger.error(f"Failed to log to {collector}: {e}")
                pass
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to all collectors."""
        for collector in self._collectors:
            try:
                collector.log_hyperparameters(params)
            except Exception as e:
                # Log error but don't fail
                pass
    
    def log_model_info(self, info: Dict[str, Any]) -> None:
        """Log model info to all collectors."""
        for collector in self._collectors:
            try:
                collector.log_model_info(info)
            except Exception as e:
                # Log error but don't fail
                pass
    
    def finalize(self) -> None:
        """Finalize all collectors."""
        for collector in self._collectors:
            try:
                collector.finalize()
            except Exception as e:
                # Log error but don't fail
                pass


# Global metrics registry
_metrics_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _metrics_registry
