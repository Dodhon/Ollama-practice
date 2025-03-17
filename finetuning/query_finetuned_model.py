#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(model_path):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}")
    
    # Check if Mac with Apple Silicon for MPS support
    device_map = "auto"
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) for acceleration")
        device_map = "mps"
    
    # Load the base model
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the model with LoRA weights
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_length=512):
    """Generate a response for the given instruction."""
    # Format the input as expected by the model
    prompt = f"Instruction: {instruction}\nResponse:"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode and return the response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part (after "Response:")
    response_parts = full_output.split("Response:")
    if len(response_parts) > 1:
        return response_parts[1].strip()
    else:
        return full_output.strip()

def main():
    # Path to your fine-tuned model
    model_path = "lora_finetuned_small_model"
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_path)
    
    print("\nModel loaded successfully! Type 'exit' to quit.")
    
    # Interactive loop
    while True:
        # Get instruction from user
        instruction = input("\nType your question: ")
        
        if instruction.lower() == 'exit':
            break
        
        # Generate and print response
        response = generate_response(model, tokenizer, instruction)
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main() 