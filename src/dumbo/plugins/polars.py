from typing import Any, List

try:
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None  # type: ignore

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

from dumbo.logger import get_logger
from dumbo.plugin_loader import DatasetLoaderPlugin
from dumbo.result import Ok, Err, Result

logger = get_logger()


class PolarsDatasetPlugin(DatasetLoaderPlugin):
    """Load datasets using Polars and optionally the 🤗 *datasets* library."""

    def load_dataset(self, config: List[dict[str, Any]]) -> Result[Any]:
        if pl is None:
            logger.error("polars is not installed")
            return Err(ImportError("polars is not installed"))

        cfg = config[0] if config else {}
        ds_type = cfg.get("type", "huggingface_polars")
        path = cfg.get("path")

        logger.info(f"Loading dataset type '{ds_type}' from {path}...")

        try:
            if ds_type == "huggingface_polars":
                if load_dataset is None:
                    raise ImportError("datasets library is not installed")
                dataset = load_dataset(path, split="train[:100]")
                df = pl.from_pandas(dataset.to_pandas())
            elif ds_type == "csv_polars":
                df = pl.read_csv(path)
            elif ds_type == "json_polars":
                df = pl.read_json(path)
            elif ds_type == "parquet_polars":
                df = pl.read_parquet(path)
            else:
                raise ValueError(f"Unsupported dataset type: {ds_type}")

            if cfg.get("data_format") == "alpaca":
                logger.info("Converting Alpaca format to messages list...")
                df = df.with_columns(
                    pl.struct(["instruction", "input", "output"]).map_elements(
                        lambda row: [
                            {
                                "role": "user",
                                "content": f"{row['instruction']}\n{row['input']}".strip(),
                            },
                            {"role": "assistant", "content": row["output"]},
                        ]
                    ).alias("messages")
                ).select("messages")

            return Ok(df)
        except Exception as e:  # pragma: no cover - runtime failures
            logger.error(f"Failed to load dataset: {e}")
            return Err(e)

AVAILABLE_PLUGINS = [PolarsDatasetPlugin]

