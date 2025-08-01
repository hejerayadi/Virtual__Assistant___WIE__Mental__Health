import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

def get_llm_response(text):
    payload = {
        "messages": [
            {"role": "user", "content": text}
        ],
        "model": "deepseek-ai/DeepSeek-V3-0324"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return "Sorry, something went wrong generating a response."
