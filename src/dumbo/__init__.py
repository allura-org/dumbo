from yaml import load, SafeLoader

from dataclasses import dataclass
from simple_parsing import field as simple_field, parse as simple_parse

from .result import Result, Ok, is_err
from .logger import setup_root, get_logger
from .plugin_loader import (
    import_plugin,
    ModelLoaderPlugin,
    TokenizerLoaderPlugin,
    ModelPatcherPlugin,
    DatasetLoaderPlugin,
    DatasetFormatterPlugin,
    TrainerPlugin,
)

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
    plugin_instances = []

    for plugin_name in config["plugins"]:
        plugin_result = import_plugin(plugin_name)
        if is_err(plugin_result):
            return plugin_result
        plugin_instances.extend(plugin_result.unwrap())
    logger.info("Plugins loaded successfully")

    model = None
    tokenizer = None
    dataset = None

    for plugin in plugin_instances:
        if isinstance(plugin, ModelLoaderPlugin):
            result = plugin.load_model(config.get("model", {}))
            if is_err(result):
                return result
            model = result.unwrap()

    for plugin in plugin_instances:
        if isinstance(plugin, TokenizerLoaderPlugin):
            result = plugin.load_tokenizer(config.get("model", {}))
            if is_err(result):
                return result
            tokenizer = result.unwrap()

    for plugin in plugin_instances:
        if isinstance(plugin, ModelPatcherPlugin) and model is not None:
            result = plugin.patch_model(model, config.get(plugin.config_key))
            if is_err(result):
                return result
            model = result.unwrap()

    for plugin in plugin_instances:
        if isinstance(plugin, DatasetLoaderPlugin):
            result = plugin.load_dataset(config.get("datasets", []))
            if is_err(result):
                return result
            dataset = result.unwrap()

    for plugin in plugin_instances:
        if isinstance(plugin, DatasetFormatterPlugin) and dataset is not None:
            result = plugin.format_dataset(dataset, config.get("datasets", []))
            if is_err(result):
                return result
            dataset = result.unwrap()

    for plugin in plugin_instances:
        if isinstance(plugin, TrainerPlugin):
            result = plugin.train(model, tokenizer, dataset, config.get("trl", {}))
            if is_err(result):
                return result

    return Ok(None)

def real_main() -> None:
    args = simple_parse(Args)
    setup_root()
    main(args)
