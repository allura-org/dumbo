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
    config = load(open(args.config), Loader=SafeLoader)
    plugins = {}
    
    # Import all plugin classes from their modules
    for plugin_name in config["plugins"]:
        plugin_result = import_plugin(plugin_name)
        if is_err(plugin_result):
            return plugin_result
        plugin_module = plugin_result.unwrap()

        # Get available plugins from the module
        if hasattr(plugin_module, 'AVAILABLE_PLUGINS'):
            for PluginClass in plugin_module.AVAILABLE_PLUGINS:
                plugin_instance = PluginClass()
                plugins[plugin_instance.__class__.__name__] = plugin_instance

    # Load model and tokenizer
    model = None
    tokenizer = None
    
    # Load model first
    model_plugins = [p for p in plugins.values() if hasattr(p, 'load_model')]
    for plugin in model_plugins:
        result = plugin.load_model(config.get(plugin.config_key, {}))
        model = result.unwrap()
        break
    
    # Load tokenizer with model reference for embedding resizing
    tokenizer_plugins = [p for p in plugins.values() if hasattr(p, 'load_tokenizer')]
    for plugin in tokenizer_plugins:
        import inspect
        sig = inspect.signature(plugin.load_tokenizer)
        if 'model' in sig.parameters:
            result = plugin.load_tokenizer(config.get(plugin.config_key, {}), model=model)
        else:
            result = plugin.load_tokenizer(config.get(plugin.config_key, {}))
        tokenizer = result.unwrap()
        break
    
    # Apply model patches
    patcher_plugins = [p for p in plugins.values() if hasattr(p, 'patch_model')]
    for plugin in patcher_plugins:
        result = plugin.patch_model(model, config.get(plugin.config_key))
        model = result.unwrap()
    
    # Load datasets
    datasets = None
    dataset_plugins = [p for p in plugins.values() if hasattr(p, 'load_datasets')]
    for plugin in dataset_plugins:
        result = plugin.load_datasets(config.get(plugin.config_key, []))
        datasets = result.unwrap()
    
    # Format datasets
    formatter_plugins = [p for p in plugins.values() if hasattr(p, 'format_text')]
    if datasets:
        for i, dataset in enumerate(datasets):
            for plugin in formatter_plugins:
                if len(config.get("datasets", [])) > i:
                    result = plugin.format_text(dataset, config[plugin.config_key][i])
                    datasets[i] = result.unwrap()
    
    # Create trainer and train
    trainer = None
    trainer_plugins = [p for p in plugins.values() if hasattr(p, 'create_trainer')]
    for plugin in trainer_plugins:
        result = plugin.create_trainer(model, tokenizer, datasets, config.get(plugin.config_key, {}))
        trainer = result.unwrap()
        break
    
    # Start training
    train_plugins = [p for p in plugins.values() if hasattr(p, 'train_model')]
    for plugin in train_plugins:
        result = plugin.train_model(trainer, config.get(plugin.config_key, {}))
        break

    return Ok(None)

def real_main() -> None:
    args = simple_parse(Args)
    setup_root()
    main(args)
