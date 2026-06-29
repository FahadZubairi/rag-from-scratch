import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
pdf_path = "AI_Notes.pdf"
Loader = PyPDFLoader(pdf_path)
pages = Loader.load()
full_text = "\n".join([page.page_content for page in pages])
merged_doc = Document(page_content=full_text)
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = splitter.split_documents([merged_doc])

load_dotenv()
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model
)
print("Stored Chunks in Chroma successfully.")
question = input("Ask a question: ")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
system_prompt = (
    "Answer the question using the context below and give a detailed answer with an example from context. "
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
retriever = vector_store.as_retriever(search_kwargs={"k":4})
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke({"input": question})
print("\nFinal Answer")
print(response["answer"])