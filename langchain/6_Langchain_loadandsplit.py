from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "AI_Notes.pdf"
Loader = PyPDFLoader(pdf_path)
documents = Loader.load()

print(f"Number of pages: {len(documents)}")
print(f"First page content: {documents[0].page_content[:300]}")
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = splitter.split_documents(documents)
print(f"Number of chunks created: {len(chunks)}")
print(f"Content of first chunk: {chunks[0].page_content}")

