# app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# ... other imports ...

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough

# --- Config ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
TOP_K = int(os.getenv("TOP_K", "4"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in environment or .env")
if not PINECONE_API_KEY:
    raise RuntimeError("Set Pinecone credentials in environment or .env")

# --- Load Vector Store & Retriever ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Initialize the Pinecone client and connect to your index
vectorstore = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ... rest of the code is the same ...
llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME)

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

rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever | format_docs,
    }
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        answer = rag_chain.invoke(request.message)
        docs = retriever.get_relevant_documents(request.message)
        sources = list({d.metadata.get("source", "unknown") for d in docs})
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))