import requests
import json

def simple_ask(question, model='mistral'):
    """Send a simple question to Ollama."""
    print(f"Asking: {question}")
    
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': model,
            'prompt': question
        },
        stream=True 
    )
    
    # Handle streaming 
    full_response = ""
    for line in response.iter_lines():
        if line:
            json_response = json.loads(line)
            if 'response' in json_response:
                full_response += json_response['response']
                # Print each chunk as it arrives
                print(json_response['response'], end='', flush=True)
    
    return full_response

def get_question():
    # Get question from user
    question = input("Enter a question: ")
    return question

def main():
    # Simple test question
    question = get_question()
    
    try:
        print("\nAnswer:")
        answer = simple_ask(question)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()