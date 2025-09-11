import os
import time
from dotenv import load_dotenv
from pathlib import Path
from typing import List
import requests
from tqdm import tqdm

from bs4 import BeautifulSoup

# LangChain parts
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# New Pinecone client
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
URLS_FILE = "urls.txt"
USER_AGENT = os.getenv("USER_AGENT", "FeedHopeBot/1.0 (+https://yourdomain.example)")

if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, GOOGLE_API_KEY]):
    raise RuntimeError(
        "Set PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME and GOOGLE_API_KEY in .env"
    )

# ---------------------
# Google Embeddings adapter for LangChain vectorstores
# ---------------------
class GoogleEmbeddingsAdapter:
    """
    Google Gemini API embeddings adapter that works with LangChain vectorstores.
    Uses the Google AI Studio API for text embeddings.
    """
    def __init__(self, model_name: str, api_key: str, batch_size: int = 100, max_retries: int = 3, backoff: float = 1.0):
        self.model = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.backoff = backoff
        
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _call_google_api(self, texts: List[str]):
        """Make API call to Google's embedding endpoint"""
        url = f"{self._base_url}/{self.model}:batchEmbedContents"
        
        requests_data = []
        for text in texts:
            requests_data.append({
                "model": self.model,
                "content": {"parts": [{"text": text}]},
                "taskType": "RETRIEVAL_DOCUMENT", 
                "outputDimensionality": 768
            })
        
        payload = {"requests": requests_data}
        params = {"key": self.api_key}
        
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url, 
                    json=payload, 
                    params=params,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    try:
                        error_body = response.json()
                    except:
                        error_body = response.text
                    raise RuntimeError(f"Google API status {response.status_code}: {error_body}")
                    
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait_time = self.backoff * (2 ** (attempt - 1))
                    print(f"Attempt {attempt} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
        raise RuntimeError(f"Google API failed after {self.max_retries} attempts: {last_exc}")

    def _call_google_api_single(self, text: str):
        """Make API call for single text embedding"""
        url = f"{self._base_url}/{self.model}:embedContent"
        
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY", 
            "outputDimensionality": 768
        }
        
        params = {"key": self.api_key}
        
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url, 
                    json=payload, 
                    params=params,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    try:
                        error_body = response.json()
                    except:
                        error_body = response.text
                    raise RuntimeError(f"Google API status {response.status_code}: {error_body}")
                    
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait_time = self.backoff * (2 ** (attempt - 1))
                    print(f"Attempt {attempt} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
        raise RuntimeError(f"Google API failed after {self.max_retries} attempts: {last_exc}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using Google's batch API"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1} with {len(batch)} texts...")
            
            try:
                response = self._call_google_api(batch)
                
                embeddings = []
                for embedding_response in response.get("embeddings", []):
                    values = embedding_response.get("values", [])
                    if not values:
                        raise RuntimeError("Empty embedding returned from Google API")
                    embeddings.append(values)
                
                if len(embeddings) != len(batch):
                    raise RuntimeError(f"Mismatch: requested {len(batch)} embeddings, got {len(embeddings)}")
                
                all_embeddings.extend(embeddings)
                
            except Exception as e:
                print(f"Batch failed, falling back to individual requests: {e}")
                for text in batch:
                    try:
                        response = self._call_google_api_single(text)
                        embedding = response.get("embedding", {}).get("values", [])
                        if not embedding:
                            raise RuntimeError("Empty embedding returned from Google API")
                        all_embeddings.append(embedding)
                    except Exception as single_e:
                        print(f"Failed to embed text: {text[:50]}... Error: {single_e}")
                        raise single_e
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Google's API"""
        response = self._call_google_api_single(text)
        embedding = response.get("embedding", {}).get("values", [])
        if not embedding:
            raise RuntimeError("Empty embedding returned from Google API")
        return embedding


# ---------------------
# helper: fetch & chunk
# ---------------------
def fetch_text_from_url(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html5lib")
    for s in soup(["script", "style", "header", "footer", "nav", "form"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n\n".join(lines)

def chunk_text_and_make_documents(text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    documents = []
    for i, chunk in enumerate(chunks):
        meta = {"source": source, "chunk": i, "text": chunk}
        documents.append(Document(page_content=chunk, metadata=meta))
    return documents


# ---------------------
# Pinecone helpers (modern client)
# ---------------------
def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

def ensure_index_exists(pc: Pinecone, index_name: str, dim: int, metric: str = "cosine"):
    idx_list = pc.list_indexes()
    if isinstance(idx_list, dict) and "indexes" in idx_list:
        existing_names = [i.get("name") for i in idx_list["indexes"]]
    elif isinstance(idx_list, list):
        if all(isinstance(x, str) for x in idx_list):
            existing_names = idx_list
        else:
            existing_names = [x.get("name") for x in idx_list]
    else:
        existing_names = []

    if index_name in existing_names:
        print(f"Pinecone index '{index_name}' already exists.")
        return

    print(f"Creating Pinecone index '{index_name}' (dim={dim}) ...")
    pc.create_index(name=index_name, dimension=dim, metric=metric,
                    spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION","us-east-1")))
    time.sleep(2)
    print("Index created.")

# ---------------------
# ingestion pipeline
# ---------------------
def ingest_urls(urls: List[str]):
    all_docs: List[Document] = []
    for url in urls:
        try:
            text = fetch_text_from_url(url)
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue
        docs = chunk_text_and_make_documents(text, source=url)
        print(f"-> {url} -> {len(docs)} chunks")
        all_docs.extend(docs)

    if not all_docs:
        print("No documents to ingest.")
        return

    embeddings = GoogleEmbeddingsAdapter(
        model_name=GOOGLE_EMBED_MODEL, 
        api_key=GOOGLE_API_KEY, 
        batch_size=50  
    )

    sample_text = all_docs[0].page_content[:1000]
    print(f"Testing embedding with sample text: {sample_text[:100]}...")
    sample_embedding = embeddings.embed_query(sample_text)
    dim = len(sample_embedding)
    print(f"Embedding dimension: {dim}")

    pc = get_pinecone_client()
    ensure_index_exists(pc, PINECONE_INDEX_NAME, dim, metric="cosine")

    index_handle = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(embedding=embeddings, index=index_handle)

    print("Uploading documents to Pinecone via PineconeVectorStore.add_documents() ...")
    ids = vector_store.add_documents(all_docs)
    print(f"Upserted {len(ids)} documents. ✅ Ingest complete.")


# ---------------------
# CLI entry
# ---------------------
if __name__ == "__main__":
    if not Path(URLS_FILE).exists():
        raise RuntimeError(f"{URLS_FILE} not found")
    with open(URLS_FILE, "r") as f:
        urls = [ln.strip() for ln in f if ln.strip()]
    ingest_urls(urls)