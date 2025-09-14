# app.py
import os
import time
import requests
import json
import asyncio
import logging
import uuid
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import bleach

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pinecone import Pinecone
from rate_limiter import get_client_ip, rate_limit_dependency

load_dotenv()

# Import configured loggers from separate module
from basic_logging import (
    app_logger, embedding_logger, vectorstore_logger, 
    llm_logger, auth_logger, rate_limit_logger
)

# --- Config / env ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
TOP_K = int(os.getenv("TOP_K", "4"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")

# API key for the HTTP API (set this in .env). Minimal auth for demo.
API_KEY = os.getenv("API_KEY", None)
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Validation with logging
app_logger.info("Initializing FeedHope RAG Chatbot...")
if not GROQ_API_KEY:
    app_logger.critical("GROQ_API_KEY not found in environment")
    raise RuntimeError("Set GROQ_API_KEY in environment or .env")
if not PINECONE_API_KEY:
    app_logger.critical("Pinecone credentials not found in environment")
    raise RuntimeError("Set Pinecone credentials in environment or .env")
if not GOOGLE_API_KEY:
    app_logger.critical("GOOGLE_API_KEY not found in environment")
    raise RuntimeError("Set GOOGLE_API_KEY in environment or .env")

app_logger.info(f"Configuration loaded - Model: {MODEL_NAME}, Embedding: {GOOGLE_EMBED_MODEL}, Top-K: {TOP_K}")

# -----------------------------
# Request Tracking Middleware
# -----------------------------
class RequestTrackingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        client_ip = await get_client_ip(request)
        app_logger.info(
            f"Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": client_ip,
                "user_agent": request.headers.get("user-agent", "unknown"),
                "content_length": request.headers.get("content-length", 0)
            }
        )
        
        try:
            response = await call_next(request)

            response_time = time.time() - request.state.start_time
            
            app_logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "success": response.status_code < 400
                }
            )
            
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            response_time = time.time() - request.state.start_time
            app_logger.error(
                f"Request failed with exception",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "response_time_ms": round(response_time * 1000, 2)
                },
                exc_info=True
            )
            raise

# -----------------------------
# GoogleEmbeddingsAdapter with Logging
# -----------------------------
class GoogleEmbeddingsAdapter:
    """
    Sync Google embedding adapter with async wrappers and comprehensive logging.
    """

    def __init__(self, model_name: str, api_key: str, batch_size: int = 100, max_retries: int = 3, backoff: float = 1.0):
        self.model = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.backoff = backoff
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        embedding_logger.info(
            f"Initialized Google Embeddings Adapter",
            extra={
                "model": model_name,
                "batch_size": batch_size,
                "max_retries": max_retries,
                "backoff": backoff
            }
        )

    def _make_text_scalar(self, text: Any) -> str:
        """Ensure the 'text' value is always a scalar string before sending to Google."""
        if isinstance(text, str):
            return text
        try:
            result = str(text)
            embedding_logger.debug(f"Converted non-string input to string: {type(text)} -> str")
            return result
        except Exception as e:
            embedding_logger.warning(f"Failed to convert text to string, using JSON: {e}")
            return json.dumps(text, default=str)

    def _call_google_api_single(self, text: Any):
        """Call the single-embed endpoint with text sanitized."""
        request_start = time.time()
        url = f"{self._base_url}/{self.model}:embedContent"
        text_scalar = self._make_text_scalar(text)
        text_length = len(text_scalar)
        
        embedding_logger.debug(
            f"Starting single embedding request",
            extra={
                "text_length": text_length,
                "model": self.model
            }
        )
        
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": text_scalar}]},
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                
                # fallback if header auth didn't work, try query param
                if response.status_code == 401 and self.api_key:
                    embedding_logger.debug("Header auth failed, trying query param auth")
                    response = requests.post(url, json=payload, params={"key": self.api_key},
                                             headers={"Content-Type": "application/json"}, timeout=30)

                response_time = time.time() - request_start
                
                if response.status_code == 200:
                    embedding_logger.info(
                        f"Single embedding request successful",
                        extra={
                            "attempt": attempt,
                            "response_time_ms": round(response_time * 1000, 2),
                            "text_length": text_length,
                            "status_code": response.status_code
                        }
                    )
                    return response.json()
                else:
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = response.text
                    
                    embedding_logger.error(
                        f"Google API error",
                        extra={
                            "attempt": attempt,
                            "status_code": response.status_code,
                            "error_body": error_body,
                            "response_time_ms": round(response_time * 1000, 2)
                        }
                    )
                    raise RuntimeError(f"Google API status {response.status_code}: {error_body}")

            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait_time = self.backoff * (2 ** (attempt - 1)) + (0.1 * attempt)
                    embedding_logger.warning(
                        f"Embedding attempt failed, retrying",
                        extra={
                            "attempt": attempt,
                            "max_retries": self.max_retries,
                            "wait_time": wait_time,
                            "error": str(exc)
                        }
                    )
                    time.sleep(wait_time)
                else:
                    embedding_logger.error(
                        f"All embedding attempts failed",
                        extra={
                            "max_retries": self.max_retries,
                            "final_error": str(exc)
                        }
                    )

        raise RuntimeError(f"Google API failed after {self.max_retries} attempts: {last_exc}")

    def _call_google_api(self, texts: List[Any]):
        """Call batchEmbedContents endpoint. Each text is sanitized."""
        request_start = time.time()
        url = f"{self._base_url}/{self.model}:batchEmbedContents"
        batch_size = len(texts)
        total_text_length = sum(len(self._make_text_scalar(text)) for text in texts)
        
        embedding_logger.info(
            f"Starting batch embedding request",
            extra={
                "batch_size": batch_size,
                "total_text_length": total_text_length,
                "model": self.model
            }
        )
        
        requests_data = []
        for text in texts:
            text_scalar = self._make_text_scalar(text)
            requests_data.append({
                "model": self.model,
                "content": {"parts": [{"text": text_scalar}]},
            })
        payload = {"requests": requests_data}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                
                if response.status_code == 401 and self.api_key:
                    embedding_logger.debug("Batch: Header auth failed, trying query param auth")
                    response = requests.post(url, json=payload, params={"key": self.api_key},
                                             headers={"Content-Type": "application/json"}, timeout=60)

                response_time = time.time() - request_start
                
                if response.status_code == 200:
                    embedding_logger.info(
                        f"Batch embedding request successful",
                        extra={
                            "attempt": attempt,
                            "batch_size": batch_size,
                            "response_time_ms": round(response_time * 1000, 2),
                            "status_code": response.status_code
                        }
                    )
                    return response.json()
                else:
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = response.text
                    
                    embedding_logger.error(
                        f"Google batch API error",
                        extra={
                            "attempt": attempt,
                            "status_code": response.status_code,
                            "batch_size": batch_size,
                            "error_body": error_body,
                            "response_time_ms": round(response_time * 1000, 2)
                        }
                    )
                    raise RuntimeError(f"Google API status {response.status_code}: {error_body}")

            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait_time = self.backoff * (2 ** (attempt - 1)) + (0.1 * attempt)
                    embedding_logger.warning(
                        f"Batch embedding attempt failed, retrying",
                        extra={
                            "attempt": attempt,
                            "max_retries": self.max_retries,
                            "batch_size": batch_size,
                            "wait_time": wait_time,
                            "error": str(exc)
                        }
                    )
                    time.sleep(wait_time)
                else:
                    embedding_logger.error(
                        f"All batch embedding attempts failed",
                        extra={
                            "max_retries": self.max_retries,
                            "batch_size": batch_size,
                            "final_error": str(exc)
                        }
                    )

        raise RuntimeError(f"Google API failed after {self.max_retries} attempts: {last_exc}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents with fallback and logging."""
        start_time = time.time()
        total_texts = len(texts)
        
        embedding_logger.info(
            f"Starting document embedding",
            extra={
                "total_documents": total_texts,
                "batch_size": self.batch_size
            }
        )
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_texts + self.batch_size - 1) // self.batch_size
            
            embedding_logger.debug(f"Processing batch {batch_num}/{total_batches} with {len(batch)} documents")
            
            try:
                response = self._call_google_api(batch)
                embeddings = []
                for emb_resp in response.get("embeddings", []):
                    vals = emb_resp.get("values", [])
                    if not vals:
                        raise RuntimeError("Empty embedding returned from Google batch API")
                    embeddings.append(vals)
                    
                if len(embeddings) != len(batch):
                    raise RuntimeError(f"Mismatch: requested {len(batch)} embeddings, got {len(embeddings)}")
                    
                all_embeddings.extend(embeddings)
                embedding_logger.debug(f"Batch {batch_num} completed successfully")
                
            except Exception as e:
                embedding_logger.warning(
                    f"Batch {batch_num} failed, falling back to individual requests",
                    extra={"error": str(e), "batch_size": len(batch)}
                )
                
                # fall back to individual calls
                for j, text in enumerate(batch):
                    try:
                        resp = self._call_google_api_single(text)
                        embedding = resp.get("embedding", {}).get("values", [])
                        if not embedding:
                            raise RuntimeError("Empty embedding returned from Google single API")
                        all_embeddings.append(embedding)
                        
                    except Exception as single_e:
                        embedding_logger.error(
                            f"Failed to embed individual document",
                            extra={
                                "batch_num": batch_num,
                                "document_index": j,
                                "error": str(single_e),
                                "text_preview": text[:100] if text else "empty"
                            }
                        )
                        raise single_e
        
        total_time = time.time() - start_time
        embedding_logger.info(
            f"Document embedding completed",
            extra={
                "total_documents": total_texts,
                "total_embeddings": len(all_embeddings),
                "total_time_ms": round(total_time * 1000, 2),
                "avg_time_per_doc_ms": round((total_time / total_texts) * 1000, 2) if total_texts > 0 else 0
            }
        )
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with logging."""
        start_time = time.time()
        text_length = len(text)
        
        embedding_logger.info(
            f"Starting query embedding",
            extra={"text_length": text_length}
        )
        
        try:
            resp = self._call_google_api_single(text)
            embedding = resp.get("embedding", {}).get("values", [])
            if not embedding:
                raise RuntimeError("Empty embedding returned from Google single API")
            
            total_time = time.time() - start_time
            embedding_logger.info(
                f"Query embedding completed",
                extra={
                    "text_length": text_length,
                    "embedding_dimension": len(embedding),
                    "total_time_ms": round(total_time * 1000, 2)
                }
            )
            
            return embedding
            
        except Exception as e:
            embedding_logger.error(
                f"Query embedding failed",
                extra={
                    "text_length": text_length,
                    "error": str(e),
                    "text_preview": text[:100] if text else "empty"
                }
            )
            raise

    # ---- async wrappers ----
    async def aembed_query(self, text: str) -> List[float]:
        embedding_logger.debug("Running async query embedding")
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embedding_logger.debug(f"Running async document embedding for {len(texts)} documents")
        return await asyncio.to_thread(self.embed_documents, texts)


# -----------------------------
# Initialize vectorstore & LLM
# -----------------------------
app_logger.info("Initializing Google embeddings...")
embeddings = GoogleEmbeddingsAdapter(model_name=GOOGLE_EMBED_MODEL, api_key=GOOGLE_API_KEY)

app_logger.info("Connecting to Pinecone...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index_handle = pc.Index(PINECONE_INDEX_NAME)
    vectorstore_logger.info(
        f"Connected to Pinecone index",
        extra={
            "index_name": PINECONE_INDEX_NAME,
            "environment": PINECONE_ENVIRONMENT
        }
    )
except Exception as e:
    vectorstore_logger.error(f"Failed to connect to Pinecone: {e}")
    raise

# vectorstore = PineconeVectorStore(embedding=embeddings, index=index_handle)
# retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

app_logger.info("Initializing LLM...")
try:
    llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME)
    llm_logger.info(f"Connected to Groq LLM with model: {MODEL_NAME}")
except Exception as e:
    llm_logger.error(f"Failed to initialize LLM: {e}")
    raise

# -----------------------------
# Prompt & format
# -----------------------------
BASE_PROMPT_TEMPLATE = """You are a helpful assistant for FeedHope, a platform that connects volunteers with food distribution opportunities. Use ONLY the provided context to answer the question.
If the answer is not in the context, then answer according to your understanding based on the question, but be clear about what information comes from the context vs. your general knowledge.

Context:
{context}

Question: {question}

Answer (be concise and cite sources if possible):"""
prompt = ChatPromptTemplate.from_template(BASE_PROMPT_TEMPLATE)

def format_docs(docs: list[Document]) -> str:
    formatted = "\n\n".join([f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in docs])
    vectorstore_logger.debug(f"Formatted {len(docs)} documents for context")
    return formatted

format_docs_runnable = RunnableLambda(format_docs)

# rag_chain = (
#     {
#         "question": RunnablePassthrough(),
#         "context": retriever | format_docs_runnable,
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# -----------------------------
# FastAPI app & models
# -----------------------------
app = FastAPI(title="FeedHope RAG Chatbot", description="RAG-powered chatbot for FeedHope platform")

# Add middleware
app.add_middleware(RequestTrackingMiddleware)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 20000:
            raise ValueError('Message too long (max 20,000 characters)')
        # Sanitize HTML/script content
        sanitized = bleach.clean(v.strip(), strip=True)
        if sanitized != v.strip():
            app_logger.warning("Sanitized user input - potential security risk detected")
        return sanitized
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if v and (len(v) > 100 or not v.replace('-', '').replace('_', '').isalnum()):
            raise ValueError('Invalid session ID format')
        return v

class SourceInfo(BaseModel):
    source: str
    chunk: Optional[int] = None
    snippet: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    request_id: Optional[str] = None

conversation_store: Dict[str, List[Dict[str, str]]] = {}

# -----------------------------
# Auth dependency
# -----------------------------
async def get_api_key(request: Request, api_key_header_value: Optional[str] = Depends(api_key_header)):
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    if API_KEY is None:
        auth_logger.debug("No API key required", extra={"request_id": request_id})
        return None
        
    if not api_key_header_value:
        auth_logger.warning("Missing API key", extra={"request_id": request_id})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API Key")
        
    if api_key_header_value != API_KEY:
        auth_logger.warning(
            "Invalid API key provided",
            extra={
                "request_id": request_id,
                "provided_key_prefix": api_key_header_value[:8] if api_key_header_value else None
            }
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")
    
    auth_logger.info("API key validated successfully", extra={"request_id": request_id})
    return api_key_header_value

@app.get("/")
def read_root():
    app_logger.info("Root endpoint accessed")
    return {"message": "FeedHope RAG Chatbot API is running!", "status": "healthy"}

@app.get("/health")
def health_check():
    app_logger.info("Health check requested")
    try:
        pc.list_indexes()
        app_logger.info("Health check passed")
        return {
            "status": "healthy", 
            "embedding_model": GOOGLE_EMBED_MODEL, 
            "llm_model": MODEL_NAME,
            "timestamp": int(time.time())
        }
    except Exception as e:
        app_logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded", 
            "reason": str(e),
            "timestamp": int(time.time())
        }

# -----------------------------
# Conversation history helper
# -----------------------------
def build_context_with_history(retrieved_context: str, session_id: Optional[str]) -> str:
    history = []
    if session_id:
        history = conversation_store.get(session_id, [])
        app_logger.debug(
            f"Retrieved conversation history",
            extra={
                "session_id": session_id,
                "history_length": len(history)
            }
        )
    
    history_lines = []
    for turn in history[-6:]:  # Last 6 turns
        role = turn.get("role", "user")
        text = turn.get("text", "")[:400]  # Truncate for context window
        history_lines.append(f"{role.upper()}: {text}")
    
    history_block = "\n".join(history_lines)
    if history_block:
        combined = f"Conversation history (most recent first):\n{history_block}\n\nRetrieved context:\n{retrieved_context}"
        app_logger.debug("Combined context with conversation history")
    else:
        combined = f"Retrieved context:\n{retrieved_context}"
        app_logger.debug("Using retrieved context without history")
    
    return combined

@app.get("/debug/rate-limit")
async def debug_rate_limit(request: Request, session_id: Optional[str] = None):
    """Debug endpoint to check current rate limit status"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    rate_limit_logger.info(
        "Rate limit debug request",
        extra={"request_id": request_id, "session_id": session_id}
    )
    
    now = time.time()
    minute_slot = int(now // 60)
    today_str = time.strftime("%Y-%m-%d")
    
    ip_identity = await get_client_ip(request)
    identity = session_id if session_id else ip_identity
    
    # Determine which limits apply
    if session_id:
        limit_type = "session"
        rpm_limit = int(os.getenv("SESSION_RPM", "1"))
        rpd_limit = int(os.getenv("SESSION_RPD", "500"))
    else:
        limit_type = "ip"
        rpm_limit = int(os.getenv("IP_RPM", "1"))
        rpd_limit = int(os.getenv("IP_RPD", "200"))
    
    # Redis keys
    min_key = f"rl:{limit_type}:{identity}:m:{minute_slot}"
    day_key = f"rl:{limit_type}:{identity}:d:{today_str}"
    
    # Get current counts from Redis
    from rate_limiter import redis
    current_min = await redis.get(min_key) or 0
    current_day = await redis.get(day_key) or 0
    
    return {
        "request_id": request_id,
        "identity": identity,
        "ip_address": ip_identity,
        "session_id": session_id,
        "limit_type": limit_type,
        "current_minute_slot": minute_slot,
        "limits": {
            "rpm": rpm_limit,
            "rpd": rpd_limit
        },
        "current_usage": {
            "minute": int(current_min),
            "day": int(current_day)
        },
        "redis_keys": {
            "minute": min_key,
            "day": day_key
        },
        "environment": {
            "SESSION_RPM": os.getenv("SESSION_RPM"),
            "SESSION_RPD": os.getenv("SESSION_RPD"),
            "IP_RPM": os.getenv("IP_RPM"),
            "IP_RPD": os.getenv("IP_RPD")
        }
    }

# Insert this above the chat endpoint (near other helpers)
async def retrieve_docs_direct(
    query_text: str,
    embeddings_adapter: GoogleEmbeddingsAdapter,
    top_k: int = 4,
    namespace: Optional[str] = None,
    max_retries: int = 3,
    base_backoff: float = 1.0,
):
    """
    Compute query embedding and call Pinecone index.query directly with retries.
    Enhanced with better error handling and initialization checks.
    """
    request_start = time.time()
    
    # Validate environment variables first
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")
    if not PINECONE_INDEX_NAME:
        raise RuntimeError("PINECONE_INDEX_NAME not set")

    # embed the query (async wrapper)
    try:
        vectorstore_logger.info(f"Starting query embedding for text length: {len(query_text)}")
        q_emb = await embeddings_adapter.aembed_query(query_text)
        vectorstore_logger.info(f"Query embedding successful, dimension: {len(q_emb) if q_emb else 'None'}")
        
        if not q_emb or len(q_emb) == 0:
            raise RuntimeError("Empty embedding vector received")
            
    except Exception as e:
        embedding_logger.error("Failed to create query embedding", exc_info=True, extra={"error": str(e)})
        raise RuntimeError(f"Embedding failed: {str(e)}")

    def _query_pinecone():
        """Create fresh sync client and query - runs in thread pool"""
        try:
            vectorstore_logger.info("Creating new Pinecone client")
            
            # Initialize based on Pinecone version - newer versions don't need environment
            if PINECONE_ENVIRONMENT:
                pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            else:
                pc = Pinecone(api_key=PINECONE_API_KEY)
                
            vectorstore_logger.info("Pinecone client created successfully")
            
            # Get index handle
            index = pc.Index(PINECONE_INDEX_NAME)
            vectorstore_logger.info("Pinecone index handle obtained")
            
            # Quick connectivity test
            try:
                stats = index.describe_index_stats()
                vectorstore_logger.info(f"Index connected - total vectors: {getattr(stats, 'total_vector_count', 'unknown')}")
            except Exception as stats_error:
                vectorstore_logger.warning(f"Could not get index stats but continuing: {stats_error}")
            
            vectorstore_logger.info(
                f"Starting Pinecone query - vector dim: {len(q_emb)}, "
                f"top_k: {top_k}, namespace: {namespace or 'default'}"
            )
            
            # Execute query
            result = index.query(
                vector=q_emb,
                top_k=top_k,
                include_metadata=True,
                include_values=False,
                namespace=namespace,
            )
            
            vectorstore_logger.info(f"Pinecone query completed - got {len(result.matches) if hasattr(result, 'matches') else 'unknown'} matches")
            return result
            
        except Exception as e:
            error_msg = f"Pinecone query error: {type(e).__name__}: {str(e)}"
            vectorstore_logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            vectorstore_logger.info(f"Attempting Pinecone query (attempt {attempt}/{max_retries})")
            
            # Run the sync Pinecone query in a thread pool
            resp = await asyncio.to_thread(_query_pinecone)
            
            # Process results
            matches = resp.matches if hasattr(resp, 'matches') else []
            vectorstore_logger.info(f"Processing {len(matches)} matches from Pinecone")
            
            docs = []
            for i, match in enumerate(matches):
                try:
                    # Extract metadata safely
                    metadata = match.metadata if hasattr(match, 'metadata') else {}
                    
                    # Try different text fields
                    text = (
                        metadata.get("text") or 
                        metadata.get("content") or 
                        metadata.get("source_text") or
                        metadata.get("snippet") or 
                        metadata.get("summary") or 
                        ""
                    )
                    
                    if not text.strip():
                        vectorstore_logger.warning(f"Match {i} has no text content, available keys: {list(metadata.keys())}")
                        text = f"[No content available - Match {i}]"
                    
                    # Create document
                    doc = Document(page_content=text.strip(), metadata=metadata)
                    docs.append(doc)
                    
                except Exception as doc_error:
                    vectorstore_logger.warning(f"Error processing match {i}: {doc_error}")
                    continue

            vectorstore_logger.info(
                "Pinecone query successful",
                extra={
                    "attempt": attempt, 
                    "documents_created": len(docs),
                    "time_ms": round((time.time()-request_start)*1000, 2)
                }
            )
            
            return docs

        except Exception as exc:
            last_exc = exc
            vectorstore_logger.error(
                f"Pinecone query failed on attempt {attempt}/{max_retries}: {type(exc).__name__}: {str(exc)}",
                exc_info=True
            )
            
            if attempt < max_retries:
                backoff = base_backoff * (2 ** (attempt - 1))
                vectorstore_logger.info(f"Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)
    
    # All attempts failed
    final_error = f"All {max_retries} Pinecone query attempts failed. Last error: {type(last_exc).__name__}: {str(last_exc)}"
    vectorstore_logger.error(final_error)
    raise RuntimeError(final_error)

# -----------------------------
# Async chat endpoint
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request_model: ChatRequest,
    request: Request,
    api_key: Optional[str] = Depends(get_api_key),
):
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    start_time = time.time()
    
    app_logger.info(
        "Chat request started",
        extra={
            "request_id": request_id,
            "session_id": request_model.session_id,
            "message_length": len(request_model.message),
            "has_api_key": api_key is not None
        }
    )
    
    try:
        # Rate limit check
        rate_limit_start = time.time()
        await rate_limit_dependency(request, request_model.session_id, text_for_token_estimate=request_model.message)
        rate_limit_time = time.time() - rate_limit_start
        
        app_logger.debug(
            "Rate limit check passed",
            extra={
                "request_id": request_id,
                "rate_limit_time_ms": round(rate_limit_time * 1000, 2)
            }
        )

        # Document retrieval using direct method (no async Pinecone client)
        retrieval_start = time.time()
        
        try:
            docs = await retrieve_docs_direct(
                query_text=request_model.message,
                embeddings_adapter=embeddings,  # Remove index parameter
                top_k=TOP_K,
                namespace=None,
                max_retries=3
            )
        except Exception as e:
            vectorstore_logger.error(
                f"Document retrieval failed",
                extra={
                    "request_id": request_id,
                    "error": str(e)
                }
            )
            raise HTTPException(
                status_code=503, 
                detail="Vector database temporarily unavailable. Please try again."
            )
        
        retrieval_time = time.time() - retrieval_start
        
        vectorstore_logger.info(
            "Document retrieval completed",
            extra={
                "request_id": request_id,
                "query_length": len(request_model.message),
                "documents_found": len(docs),
                "retrieval_time_ms": round(retrieval_time * 1000, 2)
            }
        )
        
        # Format documents and build context
        retrieved_context = format_docs(docs)
        combined_context = build_context_with_history(retrieved_context, request_model.session_id)

        # Generate answer directly with LLM (no RAG chain)
        llm_start = time.time()
        
        # Create prompt manually
        formatted_prompt = prompt.format(
            question=request_model.message,
            context=combined_context
        )
        
        # Call LLM directly
        answer = await llm.ainvoke(formatted_prompt)
        
        # Extract text content if it's a message object
        if hasattr(answer, 'content'):
            answer = answer.content
        elif isinstance(answer, dict) and 'content' in answer:
            answer = answer['content']
        
        llm_time = time.time() - llm_start
        
        llm_logger.info(
            "LLM generation completed",
            extra={
                "request_id": request_id,
                "llm_time_ms": round(llm_time * 1000, 2),
                "answer_length": len(str(answer))
            }
        )

        # Store conversation history
        if request_model.session_id:
            if request_model.session_id not in conversation_store:
                conversation_store[request_model.session_id] = []
                
            conversation_store[request_model.session_id].append({
                "role": "user", 
                "text": request_model.message
            })
            conversation_store[request_model.session_id].append({
                "role": "assistant", 
                "text": str(answer)
            })
            
            if len(conversation_store[request_model.session_id]) > 20:
                conversation_store[request_model.session_id] = conversation_store[request_model.session_id][-20:]
            
            app_logger.debug(
                "Updated conversation history",
                extra={
                    "request_id": request_id,
                    "session_id": request_model.session_id,
                    "total_turns": len(conversation_store[request_model.session_id])
                }
            )

        # Build sources
        sources = []
        for doc in docs:
            snippet = (doc.page_content[:300] + "...") if len(doc.page_content) > 300 else doc.page_content
            sources.append(SourceInfo(
                source=doc.metadata.get("source", "unknown"),
                chunk=doc.metadata.get("chunk"),
                snippet=snippet
            ))

        total_time = time.time() - start_time
        
        app_logger.info(
            "Chat request completed successfully",
            extra={
                "request_id": request_id,
                "session_id": request_model.session_id,
                "total_time_ms": round(total_time * 1000, 2),
                "rate_limit_time_ms": round(rate_limit_time * 1000, 2),
                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                "llm_time_ms": round(llm_time * 1000, 2),
                "sources_count": len(sources),
                "answer_length": len(str(answer))
            }
        )

        return ChatResponse(
            answer=str(answer),
            sources=sources,
            request_id=request_id
        )

    except HTTPException:
        raise
        
    except Exception as e:
        total_time = time.time() - start_time
        app_logger.error(
            "Chat request failed",
            extra={
                "request_id": request_id,
                "session_id": request_model.session_id,
                "error": str(e),
                "total_time_ms": round(total_time * 1000, 2)
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/debug/pinecone")
async def debug_pinecone():
    """Debug endpoint to test Pinecone connectivity"""
    debug_info = {
        "environment_check": {
            "PINECONE_API_KEY": bool(PINECONE_API_KEY and len(PINECONE_API_KEY) > 10),
            "PINECONE_ENVIRONMENT": PINECONE_ENVIRONMENT,
            "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
            "api_key_length": len(PINECONE_API_KEY) if PINECONE_API_KEY else 0
        },
        "connection_test": {}
    }
    
    try:
        # Test 1: Create Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        debug_info["connection_test"]["client_creation"] = "SUCCESS"
        
        # Test 2: List indexes
        try:
            indexes = pc.list_indexes()
            debug_info["connection_test"]["list_indexes"] = {
                "status": "SUCCESS",
                "count": len(indexes) if hasattr(indexes, '__len__') else 0,
                "index_names": []
            }
            
            # Safely extract index names
            try:
                if hasattr(indexes, '__iter__'):
                    for idx in indexes:
                        if hasattr(idx, 'name'):
                            debug_info["connection_test"]["list_indexes"]["index_names"].append(idx.name)
                        else:
                            debug_info["connection_test"]["list_indexes"]["index_names"].append(str(idx))
            except Exception as e:
                debug_info["connection_test"]["list_indexes"]["index_names_error"] = str(e)
                
        except Exception as e:
            debug_info["connection_test"]["list_indexes"] = {
                "status": "FAILED",
                "error": str(e),
                "error_type": type(e).__name__
            }
        
        # Test 3: Get index handle
        try:
            index = pc.Index(PINECONE_INDEX_NAME)
            debug_info["connection_test"]["index_handle"] = "SUCCESS"
            
            # Test 4: Get index stats
            try:
                stats = index.describe_index_stats()
                debug_info["connection_test"]["index_stats"] = {
                    "status": "SUCCESS",
                    "total_vector_count": getattr(stats, 'total_vector_count', 'unknown'),
                    "dimension": getattr(stats, 'dimension', 'unknown'),
                    "index_fullness": getattr(stats, 'index_fullness', 'unknown'),
                    "namespaces": {}
                }
                
                # Safely extract namespace info
                if hasattr(stats, 'namespaces') and stats.namespaces:
                    for ns_name, ns_data in stats.namespaces.items():
                        debug_info["connection_test"]["index_stats"]["namespaces"][ns_name] = {
                            "vector_count": getattr(ns_data, 'vector_count', 'unknown')
                        }
                        
            except Exception as e:
                debug_info["connection_test"]["index_stats"] = {
                    "status": "FAILED", 
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
        except Exception as e:
            debug_info["connection_test"]["index_handle"] = {
                "status": "FAILED",
                "error": str(e),
                "error_type": type(e).__name__
            }
            
    except Exception as e:
        debug_info["connection_test"]["client_creation"] = {
            "status": "FAILED",
            "error": str(e),
            "error_type": type(e).__name__
        }
    
    return debug_info

if __name__ == "__main__":
    import uvicorn
    app_logger.info("Starting FeedHope RAG Chatbot server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
                