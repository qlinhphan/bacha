import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chunking.chunks import load_data
from embedding.embedder import get_embedding
from llm.llama_client import llama_clients, llama_summary_conversation
from rechieval.rechieval import rechieval_data
from vectordb.mongodb import mongodb, save_data_mongodb
from vectordb.postgre import create_table_postgre, init_postgre, save_data_into_postgre
from vectordb.vector_store import vector_stores

load_dotenv()


def _stringify_for_history(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def bootstrap_resources() -> Dict[str, Any]:
    mycol = mongodb(address=os.getenv("MONGODB_URI"))

    file_in_docs = os.listdir("ingestion/docs")
    list_path_file = [os.path.join("ingestion/docs", fid) for fid in file_in_docs]

    all_data_in_db = list(mycol.find())
    if len(all_data_in_db) == 0:
        save_data_mongodb(
            list_path_file=list_path_file,
            load_data=load_data,
            mycol=mycol,
            get_embedding=get_embedding,
        )

    if not os.path.exists("faiss.index"):
        vector_stores(mycol=mycol)

    index_file = faiss.read_index("faiss.index")
    return {
        "mycol": mycol,
        "index_file": index_file,
    }


class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, description="ID de ghi nho hoi thoai")
    message: str = Field(..., min_length=1, description="Cau hoi cua nguoi dung")
    top_k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.55, ge=0.0, le=1.0)


class SummaryRequest(BaseModel):
    user_id: str = Field(default="user-123", min_length=1, description="ID nguoi dung")


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources = bootstrap_resources()
    postgre_cursor = init_postgre()
    create_table_postgre(postgre_cursor)
    app.state.mycol = resources["mycol"]
    app.state.index_file = resources["index_file"]
    app.state.postgre_cursor = postgre_cursor
    app.state.thread_histories: Dict[str, List[List[str]]] = {}
    app.state.thread_lock = threading.Lock()
    app.state.postgre_lock = threading.Lock()
    yield
    postgre_cursor.close()
    postgre_cursor.connection.close()


app = FastAPI(title="Medical RAG API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest) -> Dict[str, Any]:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="message khong duoc de trong")

    question_vector = get_embedding(question)
    if question_vector is None:
        raise HTTPException(status_code=502, detail="Khong tao duoc embedding")

    with app.state.thread_lock:
        history = list(app.state.thread_histories.get(payload.thread_id, []))

    knowledge = rechieval_data(
        question_vector=question_vector,
        index_file=app.state.index_file,
        top_k=payload.top_k,
        mycol=app.state.mycol,
        min_score=payload.min_score,
    )

    answer = llama_clients(knowledge, history, question)

    answer_text = _stringify_for_history(answer)
    updated_history = history + [[question, answer_text]]

    with app.state.thread_lock:
        app.state.thread_histories[payload.thread_id] = updated_history

    return {
        "thread_id": payload.thread_id,
        "question": question,
        "answer": answer
    }


@app.post("/threads/{thread_id}/summary")
def summarize_thread(thread_id: str, payload: SummaryRequest) -> Dict[str, Any]:
    with app.state.thread_lock:
        history = list(app.state.thread_histories.get(thread_id, []))

    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"Khong tim thay hoi thoai cho thread_id={thread_id}",
        )

    summary = llama_summary_conversation(history)

    with app.state.postgre_lock:
        save_data_into_postgre(
            cursor=app.state.postgre_cursor,
            sessionId=thread_id,
            userId=payload.user_id,
            content=summary,
        )

    return {
        "thread_id": thread_id,
        "user_id": payload.user_id,
        "turn_count": len(history),
        "summary": summary,
    }


@app.delete("/threads/{thread_id}")
def clear_thread(thread_id: str) -> Dict[str, str]:
    with app.state.thread_lock:
        app.state.thread_histories.pop(thread_id, None)
    return {"message": f"Da xoa history cua thread_id={thread_id}"}


# python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

#curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d "{\"thread_id\":\"thread-123\",\"message\":\"Bệnh nhân sốt cao và ho kéo dài\",\"top_k\":10,\"min_score\":0.55}"