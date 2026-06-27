from pypdf import PdfReader
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
print(f"\nNumber of chunks created: {len(chunks)}")
print("First Chunk: ")
print(chunks[0])
print("\nSecond chunk: ")
print(chunks[1])
