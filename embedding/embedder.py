from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))

def get_embedding(text, model=os.getenv("MODEL_EMB")):
    # text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model, dimensions=512).data[0].embedding