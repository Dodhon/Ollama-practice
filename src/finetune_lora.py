#!/usr/bin/env python3
import os
import torch
import platform
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Basic configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Base model to finetune
OUTPUT_DIR = "./lora_finetuned_model"
DATASET_PATH = "data/training_data.json"  # Path to your dataset
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 2  # Reduced for Mac
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to compensate for smaller batch size
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 512

# Detect if running on Mac
IS_MAC = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MAC and platform.processor() == "arm"

def prepare_dataset(tokenizer, dataset_path):
    """Load and prepare the dataset for training."""
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please create it first.")
    
    # Load dataset - expects a JSON file with a "text" field
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def train():
    """Main training function."""
    print(f"Starting LoRA finetuning on {MODEL_NAME}")
    print(f"Detected system: {'Mac with Apple Silicon' if IS_APPLE_SILICON else 'Mac with Intel' if IS_MAC else 'Non-Mac'}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Mac-specific model loading adjustments
    if IS_MAC:
        print("Using Mac-specific settings for model loading")
        # On Mac, especially Apple Silicon, we need to be more conservative
        model_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
        # For Apple Silicon, we can use MPS (Metal Performance Shaders)
        if IS_APPLE_SILICON and torch.backends.mps.is_available():
            print("Using MPS (Metal Performance Shaders) for acceleration")
            device_map = "mps"
            # Note: 8-bit quantization may not be fully supported on MPS
            use_8bit = False
        else:
            print("Using CPU for training (this will be slow)")
            device_map = "cpu"
            use_8bit = False
    else:
        # Standard CUDA settings for non-Mac
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "load_in_8bit": True,
        }
        use_8bit = True
    
    # Load model with appropriate settings
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to CPU with minimal settings")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Use float32 for better compatibility
            device_map="cpu",
            load_in_8bit=False,
        )
        use_8bit = False
    
    # Prepare model for training if using 8-bit quantization
    if use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Typical for transformer models
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Prepare dataset
    try:
        tokenized_dataset = prepare_dataset(tokenizer, DATASET_PATH)
    except FileNotFoundError as e:
        print(e)
        return
    
    # Training arguments with Mac-specific adjustments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=5,
        save_strategy="epoch",
        fp16=not IS_APPLE_SILICON,  # fp16 might not work well on Apple Silicon
        optim="adamw_torch",  # More compatible optimizer
        report_to="none",  # Disable wandb and other reporting that might cause issues
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Training started...")
    try:
        trainer.train()
        training_successful = True
    except Exception as e:
        print(f"Training error: {e}")
        print("Trying to save the model anyway in case of partial training...")
        training_successful = False
    
    # Save the model
    try:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {OUTPUT_DIR}")
        
        if not training_successful:
            print("Note: The model was saved but training did not complete successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print("To use this model with Ollama, you'll need to convert it to GGUF format.")

if __name__ == "__main__":
    train() 