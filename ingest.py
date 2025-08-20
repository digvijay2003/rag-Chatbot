# ingest.py
import argparse
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Define the path to your URL list file
URLS_FILE = "urls.txt"

def build_index(urls, index_dir="faiss_index", chunk_size=800, chunk_overlap=150):
    # ... (rest of the function is the same as before) ...
    # The function body remains identical to the updated version above
    
    print(f"Loading {len(urls)} URL(s)…")
    loader = WebBaseLoader(urls)
    docs = loader.load()

    print(f"Splitting into chunks… (size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    new_chunks = splitter.split_documents(docs)
    print(f"Total new chunks: {len(new_chunks)}")

    print("Embedding chunks (MiniLM)…")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(index_dir):
        print(f"Loading existing FAISS index from {index_dir}…")
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        print(f"Adding new documents to the existing index...")
        vs.add_documents(new_chunks)
    else:
        print(f"Building a new FAISS index → {index_dir}")
        vs = FAISS.from_documents(new_chunks, embedding=embeddings)
        
    vs.save_local(index_dir)
    print("✅ Done.")

if __name__ == "__main__":
    if not os.path.exists(URLS_FILE):
        print(f"Error: URL list file not found at {URLS_FILE}")
    else:
        with open(URLS_FILE, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        build_index(urls)