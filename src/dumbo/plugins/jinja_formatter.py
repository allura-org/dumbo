from dumbo.plugin_loader import BasePlugin
from dumbo.result import Result, Ok
from typing import Any, Dict, Callable
from dumbo.logger import get_logger
from jinja2 import Template

logger = get_logger()

class JinjaFormatterPlugin(BasePlugin):
    config_key = "datasets"
    provides = ["formatter"]
    
    def hooks(self) -> Dict[str, Any]:
        return {
            "text_formatter": self.format_text,
        }
    
    def format_text(self, dataset: Any, config: Dict[str, Any]) -> Result[Any]:
        if "train_format" not in config:
            return Ok(dataset)
        
        format_config = config["train_format"]
        
        if format_config.get("type") != "jinja_messages":
            return Ok(dataset)
        
        template_str = format_config.get("template", "")
        template = Template(template_str)
        
        def format_example(example):
            if "messages" in example:
                formatted = template.render(messages=example["messages"])
                return {"text": formatted.strip()}
            return {"text": str(example.get("text", ""))}
        
        logger.info("Applying Jinja2 formatting to dataset...")
        formatted_dataset = dataset.map(format_example)
        
        return Ok(formatted_dataset)

AVAILABLE_PLUGINS = [JinjaFormatterPlugin]