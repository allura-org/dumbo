from dumbo.plugin_loader import BasePlugin
from dumbo.result import Result, Ok, Err
from typing import Any, Dict, List
from dumbo.logger import get_logger
import polars as pl
from datasets import Dataset

logger = get_logger()

class PolarsDatasetPlugin(BasePlugin):
    config_key = "datasets"
    provides = ["dataset_loader"]
    
    def hooks(self) -> Dict[str, Any]:
        return {
            "dataset_loader": self.load_datasets,
        }
    
    def load_datasets(self, config: List[Dict[str, Any]]) -> Result[List[Dataset]]:
        datasets = []
        
        for dataset_config in config:
            try:
                dataset_type = dataset_config.get("type", "huggingface_polars")
                
                if dataset_type == "huggingface_polars":
                    dataset = self._load_huggingface_polars(dataset_config)
                elif dataset_type == "csv_polars":
                    dataset = self._load_csv_polars(dataset_config)
                elif dataset_type == "json_polars":
                    dataset = self._load_json_polars(dataset_config)
                elif dataset_type == "parquet_polars":
                    dataset = self._load_parquet_polars(dataset_config)
                else:
                    return Err(ValueError(f"Unsupported dataset type: {dataset_type}"))
                
                datasets.append(dataset)
                
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                return Err(e)
        
        return Ok(datasets)
    
    def _load_huggingface_polars(self, config: Dict[str, Any]) -> Dataset:
        from datasets import load_dataset
        
        path = config["path"]
        logger.info(f"Loading HuggingFace dataset: {path}")
        
        dataset = load_dataset(path)
        
        # Convert to the expected format based on data_format
        if config.get("data_format") == "alpaca":
            dataset = self._format_alpaca(dataset)
        
        return dataset["train"]
    
    def _load_csv_polars(self, config: Dict[str, Any]) -> Dataset:
        path = config["path"]
        logger.info(f"Loading CSV dataset: {path}")
        
        df = pl.read_csv(path)
        return Dataset.from_polars(df)
    
    def _load_json_polars(self, config: Dict[str, Any]) -> Dataset:
        path = config["path"]
        logger.info(f"Loading JSON dataset: {path}")
        
        df = pl.read_json(path)
        return Dataset.from_polars(df)
    
    def _load_parquet_polars(self, config: Dict[str, Any]) -> Dataset:
        path = config["path"]
        logger.info(f"Loading Parquet dataset: {path}")
        
        df = pl.read_parquet(path)
        return Dataset.from_polars(df)
    
    def _format_alpaca(self, dataset: Any) -> Any:
        # Convert alpaca format to messages format
        def format_example(example):
            if "instruction" in example and "input" in example and "output" in example:
                instruction = example["instruction"]
                input_text = example["input"]
                output_text = example["output"]
                
                if input_text:
                    user_content = f"{instruction}\n\n{input_text}"
                else:
                    user_content = instruction
                
                return {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output_text}
                    ]
                }
            return example
        
        return dataset.map(format_example)

AVAILABLE_PLUGINS = [PolarsDatasetPlugin]