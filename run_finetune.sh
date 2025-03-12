#!/bin/bash
set -e

# Check if input files are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_file1> [input_file2 ...]"
    echo "Example: $0 data/my_text1.txt data/my_text2.txt"
    exit 1
fi

# Detect Mac and Apple Silicon
IS_MAC=false
IS_APPLE_SILICON=false
if [[ "$(uname)" == "Darwin" ]]; then
    IS_MAC=true
    if [[ "$(uname -m)" == "arm64" ]]; then
        IS_APPLE_SILICON=true
        echo "Detected Apple Silicon Mac"
    else
        echo "Detected Intel Mac"
    fi
else
    echo "Detected non-Mac system"
fi

# Setup environment
echo "Setting up environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install bitsandbytes for Mac if possible
if $IS_MAC; then
    echo "Attempting to install Mac-compatible bitsandbytes..."
    if $IS_APPLE_SILICON; then
        pip install bitsandbytes-apple || echo "Warning: Could not install bitsandbytes-apple. Training will use CPU only."
    else
        pip install bitsandbytes || echo "Warning: Could not install bitsandbytes. Training will use CPU only."
    fi
else
    pip install bitsandbytes
fi

# Create dataset
echo "Creating dataset..."
python src/create_dataset.py --input "$@" --format instruction

# Check if dataset was created successfully
if [ ! -f "data/training_data.json" ]; then
    echo "Error: Dataset creation failed. Please check your input files."
    exit 1
fi

# Run finetuning
echo "Starting LoRA finetuning..."
python src/finetune_lora.py

# Check if model was saved
if [ -d "./lora_finetuned_model" ]; then
    echo "Finetuning complete! Model saved to ./lora_finetuned_model"
    echo "To use with Ollama, you'll need to convert the model to GGUF format."
else
    echo "Warning: Model directory not found. Finetuning may have failed."
    exit 1
fi 