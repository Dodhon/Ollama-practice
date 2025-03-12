#!/usr/bin/env python3
import os
import json
import argparse

def create_dataset(input_files, output_file, format_type="basic"):
    """
    Create a dataset for finetuning from text files.
    
    Args:
        input_files: List of input text files
        output_file: Path to save the JSON dataset
        format_type: How to format the data (basic, instruction, chat)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    dataset = []
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping.")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            print(f"Warning: File {file_path} is empty, skipping.")
            continue
            
        # Format the data based on the specified type
        if format_type == "basic":
            # Simple text format
            dataset.append({"text": content})
        elif format_type == "instruction":
            # Instruction format (useful for instruction tuning)
            # Assumes each file has instructions and responses separated by "###"
            parts = content.split("###")
            if len(parts) >= 2:
                instruction = parts[0].strip()
                response = "###".join(parts[1:]).strip()
                dataset.append({
                    "text": f"Instruction: {instruction}\nResponse: {response}"
                })
            else:
                print(f"Warning: File {file_path} doesn't have the expected instruction format, skipping.")
        elif format_type == "chat":
            # Chat format (useful for conversation models)
            # Assumes each line starts with "User:" or "Assistant:"
            formatted_text = ""
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith("User:") or line.startswith("Assistant:"):
                    formatted_text += line + "\n"
            if formatted_text:
                dataset.append({"text": formatted_text})
            else:
                print(f"Warning: File {file_path} doesn't have the expected chat format, skipping.")
    
    if not dataset:
        print("Error: No valid data found in the input files.")
        return False
        
    # Save the dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Dataset created with {len(dataset)} examples and saved to {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create a dataset for LLM finetuning")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Input text files")
    parser.add_argument("--output", "-o", default="data/training_data.json", help="Output JSON file")
    parser.add_argument("--format", "-f", choices=["basic", "instruction", "chat"], 
                        default="basic", help="Format type for the dataset")
    
    args = parser.parse_args()
    create_dataset(args.input, args.output, args.format)

if __name__ == "__main__":
    main() 