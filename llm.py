import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

SYSTEM_PROMPT = (
    "You are an empathetic virtual assistant for breast cancer patients. "
    "Your replies will be read aloud, so keep them short and conversational. "
    "Use a gentle and understanding tone. "
)

# Initialize chat history with the system prompt
chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

def get_llm_response(user_input, emotions=None):
    # Inject detected emotions into user message
    if emotions:
        emotion_info = (
            f"{user_input}\n\n"
            f"Detected emotions:\n"
            f"- From voice: {emotions.get('voice', 'unknown')}\n"
            f"- From facial expression: {emotions.get('face', 'unknown')}\n"
            f"- From text: {emotions.get('text', 'unknown')}\n"
            "Consider these to tailor your emotional tone accordingly."
        )
    else:
        emotion_info = user_input

    # Add the user's message to chat history
    chat_history.append({"role": "user", "content": emotion_info})

    # Prepare payload
    payload = {
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "messages": chat_history
    }

    # Send request
    response = requests.post(API_URL, headers=HEADERS, json=payload)

    try:
        content = response.json()["choices"][0]["message"]["content"]
        # Add assistant's response to history
        chat_history.append({"role": "assistant", "content": content})
        return content
    except Exception as e:
        return "Sorry, something went wrong generating a response."
