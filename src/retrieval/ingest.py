"""
Embed SEC 10-K chunks and ingest into MongoDB Atlas Vector Search.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings).
Free, fast, runs locally on CPU.
"""

import json
from pathlib import Path

import certifi
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from src.config import MONGODB_URI

CHUNKS_FILE = Path(__file__).parent.parent.parent / "data" / "sec_chunks" / "chunks.json"

DB_NAME = "fimeen"
COLLECTION_NAME = "chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # Load chunks
    print(f"Loading chunks from {CHUNKS_FILE}")
    chunks = json.loads(CHUNKS_FILE.read_text())
    print(f"  → {len(chunks)} chunks loaded")
    
    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  → Model ready (dim={model.get_sentence_embedding_dimension()})")
    
    # Embed all chunks in batch
    print(f"\nEmbedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"  → Embeddings shape: {embeddings.shape}")
    
    # Add embeddings to chunk dicts
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    
    # Connect to MongoDB Atlas
    print(f"\nConnecting to MongoDB Atlas...")
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Clear existing data for clean reingest
    deleted = collection.delete_many({})
    print(f"  → Cleared {deleted.deleted_count} existing documents")
    
    # Insert all chunks
    result = collection.insert_many(chunks)
    print(f"  → Inserted {len(result.inserted_ids)} documents")
    
    # Verify
    count = collection.count_documents({})
    sample = collection.find_one()
    print(f"\n✅ Collection '{DB_NAME}.{COLLECTION_NAME}' now has {count} documents")
    print(f"   Sample doc fields: {list(sample.keys())}")
    print(f"   Sample embedding dim: {len(sample['embedding'])}")
    
    client.close()


if __name__ == "__main__":
    main()