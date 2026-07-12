# RAG From Scratch + Multi-Tool AI Agent

Two projects built progressively — first a RAG pipeline built twice (manually
then with LangChain), then a multi-tool AI agent that uses the RAG pipeline
as one of its tools. Every piece built manually before using any framework,
so every abstraction is understood from the inside.

---

## Project 1 — RAG Pipeline

A Retrieval Augmented Generation system that answers questions about a PDF
using only retrieved context, not the model's general training knowledge.

### Two Implementations

**`manual/`** — Built entirely from scratch using the Gemini SDK directly.
Every embedding call, the chunking function, and the retrieval logic written
by hand. No LangChain.

**`langchain/`** — The same pipeline rebuilt using LangChain after the manual
version was fully working and understood. Demonstrates exactly what the
framework abstracts: document loading, splitting, embedding, vector storage,
and chaining retrieval with generation.

### RAG Pipeline (Both Versions)

1. Load PDF and extract raw text
2. Split into overlapping chunks manually
3. Generate embeddings for each chunk (Gemini `gemini-embedding-001`)
4. Store embeddings in ChromaDB (persistent across runs)
5. Embed the user's question, retrieve most similar chunks via cosine similarity
6. Generate a grounded answer using only retrieved context (Gemini `gemini-2.5-flash`)

---

## Project 2 — Multi-Tool AI Agent

An AI agent built on top of the RAG pipeline that routes questions to the
right tool automatically using Gemini's native function calling API.
No agent frameworks — the decision loop is built manually.

### Tools

| Tool | Triggers when | Data source |
|---|---|---|
| `search_notes` | AI/ML concept questions | RAG pipeline over course notes |
| `query_database` | Football statistics questions | 49,505 international matches (SQLite) |
| Direct answer | Math and general knowledge | Model's own reasoning |

### Agent Decision Loop

1. User asks a question
2. Model decides: call `search_notes`, call `query_database`, or answer directly
3. If tool called: run the real function, feed result back to model
4. Model generates final answer grounded in tool output

### What the Agent Can Do

- Generate complex SQL from schema description alone — no hardcoded queries
- Route correctly between two tools based on question semantics
- Handle out-of-scope questions gracefully without hallucinating
- Maintain persistent vector storage so embeddings are computed once, not every run

---

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
4. Create a `.env` file in the root:
   ```
   GEMINI_API_KEY=your_key_here
   ```
5. Add your own PDF to the root folder and update the filename in the scripts
6. Download the football dataset from Kaggle
   (`martj42/international-football-results-from-1872-to-2017`)
   and place `results.csv` in the root, then run:
   ```
   python setup_database.py
   ```
7. Run either project:
   ```
   python manual/5_full_RAG.py
   python langchain/7_langchain_embedandstore.py
   python langchain/9_agent_multi_tool.py
   ```

---

## What I Learned

**RAG fundamentals**
- Embeddings: representing meaning as numbers where similar ideas produce similar vectors
- Chunking tradeoffs: chunk size and overlap directly affect retrieval quality,
  and splitting strategy (merged string vs per-page) changes chunk count significantly
- Vector similarity search: retrieving by meaning not keywords using cosine similarity
- Prompt grounding: instructing a model to answer only from context to reduce hallucination

**LangChain internals**
- Having built the manual version first, every LangChain abstraction maps directly
  to the code it replaces — `PyPDFLoader`, `RecursiveCharacterTextSplitter`,
  `GoogleGenerativeAIEmbeddings`, `Chroma.from_documents`, `create_retrieval_chain`

**Agent design**
- Gemini function calling: defining tool schemas, reading model decisions,
  feeding results back in the correct 3-part conversation format
- Tool descriptions control agent behavior — scope and routing emerge from
  how you describe tools, not from hardcoded logic
- System prompts override the model's tendency to answer from training knowledge
  when you want it to always verify from a source

**Real-world engineering friction**
- University org account restrictions blocking Gemini API access
- Model deprecation breaking a working script overnight (gemini-2.0-flash → 2.5-flash)
- LangChain version change moving core chains into `langchain-classic`
- Rate limits triggered by frameworks sending requests faster than manual code
- Persistent ChromaDB eliminating 1-2 minute re-embedding on every run

---

## Tech Stack

Python, Google Gemini API, ChromaDB, LangChain, SQLite, pypdf, pandas

## Evaluation

Tested the agent on 8 questions covering all three routing paths.

| Question | Expected Tool | Actual Tool | Correct |
|---|---|---|---|
| What is backtracking search? | search_notes | search_notes | ✅ |
| What is a zero-sum game? | search_notes | search_notes | ✅ |
| What is arc consistency? | search_notes | search_notes | ✅ |
| How many FIFA World Cup matches in database? | query_database | query_database | ✅ |
| Which team played most FIFA World Cup matches? | query_database | query_database | ✅ |
| How many matches did Brazil play in World Cup? | query_database | query_database | ✅ |
| What is 17 times 13? | direct | direct | ✅ |
| What is the capital of France? | direct | direct | ✅ |

**Tool routing accuracy: 8/8 (100%)**
