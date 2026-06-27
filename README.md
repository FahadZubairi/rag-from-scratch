# RAG From Scratch

A Retrieval Augmented Generation (RAG) pipeline built manually, 
without frameworks, to understand every step before abstracting with LangChain later.

## What it does
Takes a PDF, splits it into chunks, embeds them using Google's Gemini API,
stores them in ChromaDB, and answers questions using only retrieved context.

## Pipeline
1. Load PDF and extract text
2. Split into overlapping chunks
3. Generate embeddings for each chunk (Gemini `gemini-embedding-001`)
4. Store embeddings in ChromaDB
5. Embed user question, retrieve top matching chunks via cosine similarity
6. Generate a grounded answer using Gemini `gemini-2.5-flash`

## Setup
1. Clone this repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it and run: `pip install -r requirements.txt`
4. Create a `.env` file with: `GEMINI_API_KEY=your_key_here`
5. Place your PDF in the root folder and update the filename in the script
6. Run: `python 5_full_RAG.py`

## What I learned
- How embeddings represent meaning numerically
- Chunking strategy tradeoffs (size vs overlap)
- Manual cosine similarity search vs framework abstractions
- Prompt grounding to reduce hallucination
- Real-world API issues: org account restrictions, model deprecation mid-project