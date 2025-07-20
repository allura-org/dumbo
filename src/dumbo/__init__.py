from yaml import load, SafeLoader

from dataclasses import dataclass
from simple_parsing import field as simple_field, parse as simple_parse

from .result import Result, Ok, is_err, is_ok
from .logger import setup_root, get_logger
from .plugin_loader import import_plugin
from .metrics import get_metrics_registry
from .__constants__ import __version__, __motd__

logger = get_logger()

# ascii art by joan stark <3
art = """
  __QQ  
 (_)_"> 
_)      """.split("\n") + [f"version {__version__}", __motd__]

text = """
▓█████▄  █    ██  ███▄ ▄███▓ ▄▄▄▄    ▒█████  
▒██▀ ██▌ ██  ▓██▒▓██▒▀█▀ ██▒▓█████▄ ▒██▒  ██▒
░██   █▌▓██  ▒██░▓██    ▓██░▒██▒ ▄██▒██░  ██▒
░▓█▄   ▌▓▓█  ░██░▒██    ▒██ ▒██░█▀  ▒██   ██░
░▒████▓ ▒▒█████▓ ▒██▒   ░██▒░▓█  ▀█▓░ ████▓▒░
 ▒▒▓  ▒ ░▒▓▒ ▒ ▒ ░ ▒░   ░  ░░▒▓███▀▒░ ▒░▒░▒░ 
 ░ ▒  ▒ ░░▒░ ░ ░ ░  ░      ░▒░▒   ░   ░ ▒ ▒░ 
 ░ ░  ░  ░░░ ░ ░ ░      ░    ░    ░ ░ ░ ░ ▒  
   ░       ░            ░    ░          ░ ░  """.split("\n")

@dataclass
class Args:
    config: str = simple_field(positional=True)

def main(args: Args) -> Result[None]:
    for idx, text_line in enumerate(text):
        print(text_line, end=" ")
        if dict(enumerate(art)).get(idx, None):
            print(art[idx], end="")
        print("")
        

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

    # Initialize logging
    logging_plugins = [p for p in plugins.values() if "logging" in p.provides]
    for plugin in logging_plugins:
        hooks = plugin.hooks()
        if "log_init" in hooks:
            result = hooks["log_init"](config.get(plugin.config_key, {}))
            if is_err(result):
                return result

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
    
    # Initialize metrics registry and register collectors
    registry = get_metrics_registry()
    
    # Register metrics collectors from plugins
    for plugin_name, plugin in plugins.items():
        if hasattr(plugin, 'get_metrics_collector'):
            try:
                collector_result = plugin.get_metrics_collector()
                if is_ok(collector_result) and collector_result.value:
                    registry.register(collector_result.value)
                    logger.info(f"Registered metrics collector from plugin: {plugin_name}")
            except Exception as e:
                logger.warning(f"Failed to register collector from {plugin_name}: {e}")
        
        # Allow plugins to provide collectors via hooks
        if hasattr(plugin, 'hooks'):
            hooks = plugin.hooks()
            if 'metrics_collector' in hooks:
                try:
                    collector = hooks['metrics_collector']()
                    if collector:
                        registry.register(collector)
                        logger.info(f"Registered metrics collector via hook from plugin: {plugin_name}")
                except Exception as e:
                    logger.warning(f"Failed to register collector via hook from {plugin_name}: {e}")
    
    # Log model info
    model_info = {
        "model_type": str(type(model)),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "config": model.config.to_dict() if hasattr(model, 'config') else {}
    }
    for plugin in logging_plugins:
        hooks = plugin.hooks()
        if "log_model" in hooks:
            result = hooks["log_model"](model_info)
            if is_err(result):
                return result
    
    # Log dataset info
    if datasets:
        dataset_info = {
            "num_datasets": len(datasets),
            "dataset_types": [str(type(d)) for d in datasets]
        }
        for plugin in logging_plugins:
            hooks = plugin.hooks()
            if "log_dataset" in hooks:
                result = hooks["log_dataset"](dataset_info)
                if is_err(result):
                    return result
    
    # Log hyperparameters
    for plugin in logging_plugins:
        hooks = plugin.hooks()
        if "log_hyperparameters" in hooks:
            result = hooks["log_hyperparameters"](config)
            if is_err(result):
                return result
    
    # Log training start
    for plugin in logging_plugins:
        hooks = plugin.hooks()
        if "log_training_start" in hooks:
            result = hooks["log_training_start"](config)
            if is_err(result):
                return result
    
    # Start training
    train_plugins = [p for p in plugins.values() if hasattr(p, 'train_model')]
    for plugin in train_plugins:
        result = plugin.train_model(trainer, config.get(plugin.config_key, {}))
        if is_err(result):
            return result
        break

    # Log training completion
    training_summary = {}  # Could be populated by trainer
    for plugin in logging_plugins:
        hooks = plugin.hooks()
        if "log_training_end" in hooks:
            result = hooks["log_training_end"](training_summary)
            if is_err(result):
                return result
    
    # Cleanup logging
    for plugin in logging_plugins:
        hooks = plugin.hooks()
        if "finish" in hooks:
            result = hooks["finish"]()
            if is_err(result):
                return result

    return Ok(None)

def real_main() -> None:
    args = simple_parse(Args)
    setup_root()
    main(args)
