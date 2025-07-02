from yaml import load, SafeLoader

from dataclasses import dataclass
from simple_parsing import field as simple_field, parse as simple_parse

from .result import Result, Ok, is_err
from .logger import setup_root, get_logger
from .plugin_loader import import_plugin

logger = get_logger()

@dataclass
class Args:
    config: str = simple_field(positional=True)

def main(args: Args) -> Result[None]:
    logger.info("Haiiiii :3")

    logger.info(f"Loading config from {args.config}")
    config = load(open(args.config), Loader=SafeLoader)
    logger.info("Config loaded successfully")

    logger.info("Loading plugins...")
    plugins = {}

    for plugin_name in config["plugins"]:
        plugin_result = import_plugin(plugin_name)
        if is_err(plugin_result):
            return plugin_result
        plugins[plugin_name] = plugin_result.unwrap()
    logger.info("Plugins loaded successfully")

    return Ok(None)

def real_main() -> None:
    args = simple_parse(Args)
    setup_root()
    main(args)