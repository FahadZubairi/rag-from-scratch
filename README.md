# RAG From Scratch

A Retrieval Augmented Generation (RAG) pipeline built twice: first entirely
by hand with no frameworks, then rebuilt using LangChain — to understand
exactly what the framework abstracts before relying on it.

## What This Project Does

Takes a PDF (in this case, my AI course notes), splits it into chunks,
converts those chunks into embeddings, stores them in a vector database,
and answers questions about the document using only retrieved context —
not the model's general training knowledge.

## Two Implementations

**`manual/`** — Built entirely from scratch using the Gemini SDK directly,
ChromaDB, and pypdf. Every embedding call, the chunking function, and the
retrieval logic are written by hand. No LangChain.

**`langchain/`** — The same pipeline rebuilt using LangChain, after the
manual version was fully working and understood. Demonstrates what
LangChain automates: document loading, splitting, embedding, vector
storage, and chaining retrieval with generation.

## Pipeline (Both Versions)

1. Load PDF and extract text
2. Split into overlapping chunks
3. Generate embeddings for each chunk (Gemini `gemini-embedding-001`)
4. Store embeddings in ChromaDB
5. Embed the user's question, retrieve the most similar chunks via cosine similarity
6. Generate a grounded answer using only the retrieved context (Gemini `gemini-2.5-flash`)

## Setup

1. Clone this repo
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_key_here
   ```
5. Place your own PDF in the root folder and update the filename inside the scripts
6. Run either version:
   ```
   python manual/5_full_RAG.py
   python langchain/7_langchain_embedandstore.py
   ```

## What I Learned

- **Embeddings**: representing meaning as numbers, where similar ideas produce similar vectors
- **Chunking strategy**: balancing chunk size and overlap to avoid losing context at boundaries, and how chunking method (single merged string vs per-page splitting) changes the number and quality of chunks
- **Vector similarity search**: retrieving information by meaning, not keywords, using cosine similarity
- **Prompt grounding**: instructing a model to answer only from given context to reduce hallucination
- **What LangChain actually abstracts**: having built the manual version first, I could map every LangChain function (`PyPDFLoader`, `RecursiveCharacterTextSplitter`, `GoogleGenerativeAIEmbeddings`, `Chroma.from_documents`, `create_retrieval_chain`) back to the exact manual code it replaces
- **Real-world engineering friction**: university account restrictions blocking API access, a model deprecation breaking a previously working script overnight, rate limits triggered by a framework sending requests faster than my manual code did, and a LangChain version change that moved `create_stuff_documents_chain` into a separate `langchain-classic` package

## Tech Stack

Python, Google Gemini API, ChromaDB, pypdf, LangChain
