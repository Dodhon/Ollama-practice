#!/usr/bin/env python3
import os
import torch
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Minimal LoRA finetuning")
    parser.add_argument("--input", "-i", required=True, help="Input text file for training")
    parser.add_argument("--output", "-o", default="./lora_model", help="Output directory for model")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load training data
    print(f"Loading training data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into instruction/response pairs
    examples = []
    for pair in text.split("\n\n"):
        if "###" in pair:
            parts = pair.split("###")
            if len(parts) >= 2:
                instruction = parts[0].strip()
                response = "###".join(parts[1:]).strip()
                examples.append({
                    "text": f"Instruction: {instruction}\nResponse: {response}"
                })
    
    if not examples:
        print("Error: No valid instruction/response pairs found")
        return
    
    print(f"Found {len(examples)} training examples")
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    
    # Load model and tokenizer (using TinyLlama for speed and memory efficiency)
    print("Loading model and tokenizer")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Check if Mac with Apple Silicon for MPS support
    device_map = "auto"
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) for acceleration")
        device_map = "mps"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    print(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    print("Finetuning complete!")

if __name__ == "__main__":
    main() 