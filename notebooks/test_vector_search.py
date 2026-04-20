"""
Quick test: embed a query, run vector search, show top-k results.
Verifies that the Atlas Vector Search index is working end-to-end.
"""

import certifi
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from src.config import MONGODB_URI

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "fimeen_vector_index"
TOP_K = 3


def search(query: str, model: SentenceTransformer, collection) -> None:
    """Embed a query and run vector search. Print top-k results."""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": TOP_K,
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
    ])
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. [{doc['chunk_id']}] — {doc['company']} — score: {doc['score']:.4f}")
        print(f"   {doc['text'][:250]}...")


def main():
    print(f"Loading embedder: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"Connecting to MongoDB Atlas...")
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    collection = client["fimeen"]["chunks"]
    
    # Test queries — varied to see if retrieval works across companies
    test_queries = [
        "What was Microsoft's cloud revenue growth?",
        "What are NVIDIA's main risks from export controls?",
        "Apple product launches in fiscal 2024",
    ]
    
    for query in test_queries:
        search(query, model, collection)
    
    client.close()
    print(f"\n{'='*70}")
    print("✅ Vector search test complete.")


if __name__ == "__main__":
    main()