# ingest.py
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# Define Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
URLS_FILE = "urls.txt"

def ingest_to_pinecone(urls):
    print(f"Loading {len(urls)} URL(s)…")
    loader = WebBaseLoader(urls)
    docs = loader.load()

    print("Splitting into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    print("Embedding chunks and pushing to Pinecone…")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # This line connects to Pinecone and upserts the documents.
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=INDEX_NAME
    )
    print("✅ Done.")

if __name__ == "__main__":
    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, INDEX_NAME]):
        raise RuntimeError("Pinecone environment variables not set.")
    
    if not os.path.exists(URLS_FILE):
        print(f"Error: URL list file not found at {URLS_FILE}")
    else:
        with open(URLS_FILE, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        ingest_to_pinecone(urls)