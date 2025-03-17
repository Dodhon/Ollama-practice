#!/bin/bash
set -e

# Check if input file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_file>"
    echo "Example: $0 data/sample.txt"
    exit 1
fi

# Setup environment
python3 -m pip install -r requirements_minimal.txt

# Run finetuning
python3 src/minimal_finetune.py --input "$1" --output "./lora_model"

echo "Finetuning complete! Model saved to ./lora_model" 