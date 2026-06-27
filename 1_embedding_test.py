import os 
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key = api_key)

text = "What is the meaning of life"
result = client.models.embed_content(
    model = "gemini-embedding-001",
    contents = text
)
embedding = result.embeddings[0].values
print(f"text = {text}")
print(f"Embedding length: {len(embedding)}")
print(f"First 5 numbers: {embedding[:5]}")
