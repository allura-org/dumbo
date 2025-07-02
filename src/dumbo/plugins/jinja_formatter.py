from __future__ import annotations

from typing import Any, Callable, List, TYPE_CHECKING

try:
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None  # type: ignore
if TYPE_CHECKING:
    import polars as pl  # type: ignore

try:
    from jinja2 import Template
except Exception:  # pragma: no cover - optional dependency
    Template = None  # type: ignore

from dumbo.logger import get_logger
from dumbo.plugin_loader import DatasetFormatterPlugin
from dumbo.result import Ok, Err, Result

logger = get_logger()


class JinjaFormatterPlugin(DatasetFormatterPlugin):
    """Apply a Jinja template to each dataset row."""

    def format_dataset(self, dataset: Any, config: List[dict[str, Any]]) -> Result[Any]:
        if pl is None or Template is None:
            logger.error("Required dependencies for JinjaFormatter are missing")
            return Err(ImportError("polars or jinja2 not installed"))
        if not dataset.shape[0]:
            return Ok(dataset)
        fmt = config[0].get("train_format", {})
        template_str = fmt.get("template", "{{ messages }}")
        template = Template(template_str)
        logger.info("Formatting dataset with Jinja template...")
        rendered = [template.render(messages=row[0]) for row in dataset.select("messages").iter_rows()]
        df = dataset.with_columns(pl.Series("text", rendered))
        return Ok(df)

AVAILABLE_PLUGINS = [JinjaFormatterPlugin]

