import os
import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

pdf_path = "AI_Notes.pdf"
reader = PdfReader(pdf_path)
print(f"Number of pages: {len(reader.pages)}")
full_text = ""
for page in reader.pages: 
    full_text += page.extract_text()
print(f"Number of characters extracted: {len(full_text)}")
print("First 500 characters: ")
print(full_text[:500])

def chunk_text(text, chunk_size= 500,overlap = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
chunks = chunk_text(full_text)
print(f"Created {len(chunks)} chunks")

print("Embedding chunks, this may take a minute....")
embeddings = []

for i, chunk in enumerate(chunks):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    vector = result.embeddings[0].values
    embeddings.append(vector)
    print(f"Embedded chunk {i+1}/{len(chunks)}")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="ai_notes")
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"\nStored {collection.count()} chunks in chromadb")
