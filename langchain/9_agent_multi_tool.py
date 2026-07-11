import os 
import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types
import chromadb
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

search_notes_tool = {
    "name": "search_notes",
    "description": "Searches the user's AI course notes for relevant information. Use this when the question is about AI/ML concepts like CSP, search algorithms, adversarial search, etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description":"The search query to look up in the notes"
            }
        },
        "required": ["query"]
}}
query_database_tool = {
    "name": "query_database",
    "description": "Runs a SQL query against a football matches database containing 49,505 international matches from 1872 to 2024. Columns are: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral. Use this for any question about football match results, scores, teams, or tournaments.",
        "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description":"A valid SQLite SQL query to run against the 'matches' table"
            }
        },
        "required": ["sql"]
}}
tools = types.Tool(function_declarations=[search_notes_tool, query_database_tool])
config = types.GenerateContentConfig(
    tools=[tools],
    system_instruction=(
        "You are a helpful assistant with two tools: " 
        "1) search_notes for AI/ML concept questions"
        "2) query_database for ANY football statistics questions — always use this tool for football questions, never answer from your own knowledge. "
        "Answer maths and general knowledge questions directlt. "
        "Always use the most appropiate tool for the questions. "
    )
)
def build_vector_store():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("ai_notes")
    if collection.count() > 0:
        print(f"Loaded existing vector store ({collection.count()} chunks)\n")
        return collection
    
    reader = PdfReader("AI_Notes.pdf")
    full_text = "\n".join([page.extract_text() for page in reader.pages])
    merged_doc = Document(page_content=full_text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([merged_doc])
   
    print("first_run - Embedding chunks...")
    embeddings = []
    for chunk in chunks:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk.page_content
        )
        embeddings.append(result.embeddings[0].values)
    collection.add(
        documents=[c.page_content for c in chunks],
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("\nRAG Ready")
    return collection
def search_notes(query: str, collection) -> str:
    question_embedding= client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    ).embeddings[0].values
    results=collection.query(
        query_embeddings=[question_embedding],
        n_results=4
    )
    return "\n\n".join(results["documents"][0])
def query_database(sql: str) -> str:
    try:
        conn= sqlite3.connect("football.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        if not rows:
            return "No results found"
        result = ", ".join(columns) + "\n"
        for row in rows[:20]:
            result += ", ".join(str(val) for val in row) + "\n"
        return result
    except Exception as e:
        return f"SQL Error {str(e)}"
collection = build_vector_store()
while True:
    question = input("Ask a question (or exit): ")
    if question.lower() == "exit":
        break
    response= client.models.generate_content(
        model= "gemini-2.5-flash",
        contents=question,
        config=config
    )
    part = response.candidates[0].content.parts[0]
    if part.function_call:
        tool_name = part.function_call.name
        args = dict(part.function_call.args)
        print(f"Calling tool: {tool_name}")
        print(f"Arguments: {args}")
        if tool_name == "search_notes":
            tool_result = search_notes(args["query"], collection)
        elif tool_name == "query_database":
            tool_result = query_database(args["sql"])
            print(f"Raw DB result: \n{tool_result}")
        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=question)]),
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)]),
                types.Content(role="tool", parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name=tool_name,
                        response={"result": tool_result}
                    )
                )])
            ],
            config=config
        )
        print(f"\nFinal Answer:\n{final_response.text}\n")

    else:
        print(f"\nDirect Answer:\n{part.text}\n")