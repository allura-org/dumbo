# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Dumbo is a modular training framework built on top of Transformers for fine-tuning language models. It uses a plugin-based architecture where functionality is provided through plugins that can be mixed and matched.

## Architecture

### Core Components
- **Plugin System**: All functionality is provided through plugins loaded from `src/dumbo/plugins/`
- **Configuration**: YAML-based configuration files define model, datasets, and training parameters
- **Pipeline**: Model → Tokenizer → Dataset → Training → Output

### Plugin Types
- **Model Loader**: Loads models (`transformers_model`, etc.)
- **Tokenizer Loader**: Loads tokenizers (`transformers_tokenizer`, etc.)
- **Model Patcher**: Applies modifications to models (Liger kernel, LoRA, etc.)
- **Dataset Loader**: Loads datasets (`polars`, etc.)
- **Formatter**: Formats data for training (`jinja_formatter`, etc.)
- **Trainer**: Creates training setup (`transformers_trainer`, etc.)

## Usage

### Running Training
```bash
# Basic usage
uv run dumbo examples/smollm/small.yaml

# With custom config
uv run dumbo my_config.yaml
```

### Common Commands
```bash
# Install dependencies
uv sync

# Run specific example
uv run dumbo examples/tinystories_ci/1m_sft.yaml

# Development mode
uv run python -m dumbo examples/smollm/small.yaml
```

### Configuration Structure
```yaml
model:
  base_model: HuggingFaceTB/SmolLM2-135M
  tokenizer:
    pad_token: "<|pad|>"
    eos_token: "<|im_end|>"
  liger:
    rope: true
    cross_entropy: false

datasets:
  - path: tatsu-lab/alpaca
    type: huggingface_polars
    data_format: alpaca
    train_format:
      type: jinja_messages
      template: "..."

trainer:
  arguments:
    batch_size: 16
    physical_batch_size: 1
    learning_rate: 1e-4

plugins:
  - transformers
  - transformers_trainer
  - liger
  - polars
  - jinja_formatter
```

## Plugin Development

### Creating a Plugin
1. Create file in `src/dumbo/plugins/`
2. Inherit from appropriate base class (`ModelLoaderPlugin`, etc.)
3. Implement required methods
4. Add to `AVAILABLE_PLUGINS`

### Plugin Loading Order
1. Model loaded first
2. Tokenizer loaded with model reference for embedding resizing
3. Model patches applied
4. Datasets loaded and formatted
5. Trainer created
6. Training executed

## Key Files
- `src/dumbo/__init__.py`: Main entry point and orchestration
- `src/dumbo/plugin_loader.py`: Plugin system base classes
- `src/dumbo/plugins/transformers.py`: Model/tokenizer loading
- `src/dumbo/plugins/transformers_trainer.py`: Training setup
- `src/dumbo/plugins/liger.py`: Liger kernel optimizations