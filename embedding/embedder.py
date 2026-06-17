# from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import requests
import json

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))

# def get_embedding(text, model=os.getenv("MODEL_EMB")):
#     # text = text.replace("\n", " ")
#     return client.embeddings.create(input = [text], model=model, dimensions=512).data[0].embedding

def get_embedding(text_input):
    url = "http://10.10.61.29:11434/api/embed"
    
    payload = {
        "model": "bge-m3",
        "input": text_input
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        vector = data.get("embeddings")[0]
        
        return vector

    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối đến Ollama: {e}")
        return None

# --- Chạy thử ---