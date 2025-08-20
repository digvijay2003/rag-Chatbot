# api.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough

# --- Config ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
TOP_K = int(os.getenv("TOP_K", "4"))

if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in environment or .env")

# --- Load Vector Store & Retriever ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# --- LLM (Groq) ---
llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME)

# --- Prompt: grounded answering ---
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use ONLY the provided context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer (be concise and cite sources if possible):"""
)

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        [f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in docs]
    )

# Compose a Retrieval-Augmented chain (LCEL)
rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever | format_docs,
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- FastAPI App ---
app = FastAPI()

# Pydantic model for the request body
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # Get the answer from the RAG chain
        answer = rag_chain.invoke(request.message)

        # Optional: get sources for UI
        docs = retriever.get_relevant_documents(request.message)
        sources = list({d.metadata.get("source", "unknown") for d in docs})

        return {"answer": answer, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))