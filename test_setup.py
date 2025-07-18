#!/usr/bin/env python3
"""
Simple test to verify the training setup works correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dumbo import main
from dataclasses import dataclass
import tempfile

# Test configuration
TEST_CONFIG = """
model:
    base_model: HuggingFaceTB/SmolLM2-135M
    liger:
        rope: true
        cross_entropy: false
        fused_linear_cross_entropy: true
        rms_norm: true
        swiglu: true

datasets:
    - path: tatsu-lab/alpaca
      type: huggingface_polars
      data_format: alpaca
      train_format:
        type: jinja_messages
        template: |
            {% for message in messages -%}
            <|im_start|>{{ message.role }}
            {{ message.content }}<|im_end|>
            {% endfor %}

trl:
    trainer_type: sft
    arguments:
        batch_size: 1
        physical_batch_size: 1
        learning_rate: 1e-4
        num_epochs: 1
        max_steps: 5

plugins:
    - transformers
    - liger
    - polars
    - jinja_formatter
    - trl
"""

@dataclass 
class TestArgs:
    config: str

def test_end_to_end():
    """Test the complete training pipeline"""
    print("Testing end-to-end training setup...")
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(TEST_CONFIG)
        config_path = f.name
    
    try:
        args = TestArgs(config=config_path)
        result = main(args)
        
        if hasattr(result, 'unwrap'):
            result.unwrap()
            print("✓ Training setup completed successfully!")
        else:
            print("✓ Training setup completed successfully!")
            
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(config_path)
    
    return True

if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)