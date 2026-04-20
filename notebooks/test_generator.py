"""Quick test: retrieve + generate for one question."""

from src.graph.nodes.retriever import retrieve_chunks
from src.graph.nodes.generator import generate_answer


def main():
    question = "What was Microsoft's cloud revenue growth in fiscal 2024?"
    
    print(f"Question: {question}\n")
    print("Retrieving chunks...")
    chunks = retrieve_chunks(question, top_k=5)
    print(f"  → {len(chunks)} chunks retrieved")
    for c in chunks:
        print(f"  - [{c['chunk_id']}] ({c['company']}, score={c['score']:.3f})")
    
    print("\nGenerating answer...\n")
    answer = generate_answer(question, chunks)
    print(f"Answer:\n{answer}\n")


if __name__ == "__main__":
    main()