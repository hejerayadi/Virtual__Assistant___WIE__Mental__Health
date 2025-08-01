from text_emotion import detect_emotions
from llm import query_llm  # We need to slightly modify llm.py to make this work
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    user_input = "i am feelinf sad and stressed give me some tips to feel better"
    
    # Step 1: detect emotions
    emotions = detect_emotions(user_input)
    print("Detected emotions:", emotions)
    
    # Step 2: build prompt for LLM that includes emotion info
    prompt = (
        f"User said: {user_input}\n"
        f"User emotions: {emotions}\n"
        "Based on this, give helpful advice to feel better."
    )
    
    # Step 3: query LLM with prompt
    response = query_llm(prompt)
    print("LLM response:", response)

if __name__ == "__main__":
    main()
