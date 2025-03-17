# Minimal LoRA Finetuning

A proof-of-concept for finetuning language models using LoRA.

## Quick Start

1. Create a text file with instruction/response pairs separated by `###`:

```
Write a poem about AI.
###
Circuits of thought in silicon dreams,
Intelligence flows in digital streams.
```

2. Run the finetuning script:

```bash
./finetune_minimal.sh your_data.txt
```

3. The finetuned model will be saved to `./lora_model`

## Requirements

- Python 3.8+
- Mac with 8GB+ RAM (uses TinyLlama 1.1B model)
- Internet connection (to download the base model)

## How It Works

This script:
1. Loads your training data
2. Downloads a small language model (TinyLlama 1.1B)
3. Applies LoRA adapters to make training efficient
4. Finetunes on your data
5. Saves the resulting model

The entire process is contained in a single Python file for simplicity. 