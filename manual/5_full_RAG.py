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

embeddings = []

for i, chunk in enumerate(chunks):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    vector = result.embeddings[0].values
    embeddings.append(vector)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="ai_notes")
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print("Setup complete! Ready to search ")
question = input("Ask a question: ")
question_embed = client.models.embed_content(
    model = "gemini-embedding-001",
    contents = question
)
question_vector= question_embed.embeddings[0].values
results = collection.query(
    query_embeddings=[question_vector],
    n_results=4
)
print(f"Quesion: {question}")
print("Top 4 chunks found: \n")
for i,doc in enumerate(results["documents"][0]):
    print(f"... Match {i+1} ...")
    print(doc)
    print()

retrieved_chunks= results["documents"][0]
context = "\n\n".join(retrieved_chunks)
prompt = f"""Answer the question in your own words, in a full sentence, using ONLY the context below.
If the answer isn't in the context,say "I don't know based on the given notes."

Context:{context}
Question:{question}
Answer:"""
answer_response= client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
print("="*50)
print("FINAL ANSWER")
print(answer_response.text)