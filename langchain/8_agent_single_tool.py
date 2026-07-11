from dotenv import load_dotenv
from pypdf import PdfReader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from google import genai
from google.genai import types
import chromadb

load_dotenv()
api_key= os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
search_notes_tool = {
    "name": "search_notes",
    "description": "Searches the user's AI course notes for relevant information. Use this when the question is about AI/ML concepts like CSP, search algorithms, adversarial search, etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up in the notes"
            }
        },
        "required": ["query"]
    }
}
tools = types.Tool(function_declarations=[search_notes_tool])
config = types.GenerateContentConfig(tools=[tools])
def build_vector_store():
    reader = PdfReader("AI_Notes.pdf")
    full_text = "\n".join([page.extract_text() for page in reader.pages])
    merged_doc = Document(page_content=full_text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks = splitter.split_documents([merged_doc])
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("ai_notes")
    print("Embedding chunks in Rag...")
    embeddings = []
    
    for chunk in chunks:
        result = client.models.embed_content(
            model = "gemini-embedding-001",
            contents = chunk.page_content
        )
        embeddings.append(result.embeddings[0].values)
    collection.add(
        documents=[c.page_content for c in chunks],
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("RAG Ready\n")
    return collection
def search_notes(query : str, collection) -> str:
    question_embedding= client.models.embed_content(
        model= "gemini-embedding-001",
        contents=query
    ).embeddings[0].values
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results = 4
    )
    return "\n\n".join(results["documents"][0])
collection = build_vector_store()
while True:
    question = input("Ask a question (or exit!): ")
    if question.lower() == "exit":
        break
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=config
    )
    part = response.candidates[0].content.parts[0]
    if part.function_call:
        print(f"Calling tool: {part.function_call.name}")
        args = dict(part.function_call.args)
        tool_result = search_notes(args["query"], collection)

        final_response = client.models.generate_content(
            model= "gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=question)]),
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)]),
                types.Content(role="tool", parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name="search_notes",
                        response={"result": tool_result}
                    )
                )])
            ],
            config=config
        )
        print(f"\nFinal Answer: \n{final_response.text}\n")
    else:
        print(f"Direct Answer: {part.text}")