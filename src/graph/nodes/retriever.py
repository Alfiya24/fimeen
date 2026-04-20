"""
Retriever node: embeds query, runs Atlas Vector Search, returns top-k chunks.
"""

from typing import TypedDict

import certifi
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from src.config import MONGODB_URI

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "fimeen_vector_index"
DB_NAME = "fimeen"
COLLECTION_NAME = "chunks"
DEFAULT_TOP_K = 5

# Module-level singletons — loaded once, reused across calls
_model: SentenceTransformer | None = None
_client: MongoClient | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_collection():
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    return _client[DB_NAME][COLLECTION_NAME]


def retrieve_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Embed question, run Atlas Vector Search, return top-k chunks.
    
    Each returned chunk has: chunk_id, company, text, score.
    """
    model = _get_model()
    collection = _get_collection()
    
    query_embedding = model.encode(question).tolist()
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "company": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ]
    
    return list(collection.aggregate(pipeline))


# LangGraph-compatible node signature
class RetrieverState(TypedDict, total=False):
    question: str
    retrieved_chunks: list[dict]


def retriever_node(state: RetrieverState) -> RetrieverState:
    """
    LangGraph node: takes state with 'question', returns state with 'retrieved_chunks'.
    """
    question = state["question"]
    chunks = retrieve_chunks(question, top_k=DEFAULT_TOP_K)
    return {"retrieved_chunks": chunks}