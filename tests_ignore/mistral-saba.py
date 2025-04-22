import os
import requests


def call_mistral_saba(prompt: str, max_tokens: int = 1000) -> str:
    """
    Make an API call to Mistral Saba through OpenRouter
    
    Args:
        prompt (str): The input prompt
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The model's response
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    data = {
        "model": "mistralai/mistral-saba",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.text}")
        
    return response.json()["choices"][0]["message"]["content"]

# Example usage
if __name__ == "__main__":
    try:
        response = call_mistral_saba("are you the best model for franco-arabic translation?")
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")
