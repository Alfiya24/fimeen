"""
End-to-end test of the vanilla RAG pipeline.

Tests 5 questions across 3 companies to verify:
- Retrieval returns relevant chunks
- Generator produces grounded answers
- LangGraph orchestration flows correctly
"""

from src.graph.vanilla_rag import run_vanilla_rag


TEST_QUESTIONS = [
    "What was Microsoft's cloud revenue growth in fiscal 2024?",
    "What are NVIDIA's main risks from U.S. export controls?",
    "What new products did Apple launch in fiscal 2024?",
    "How did Microsoft's Azure growth compare to prior year?",
    "What is NVIDIA's policy on supply chain and demand estimates?",
]


def main():
    print(f"Running vanilla RAG on {len(TEST_QUESTIONS)} questions\n")
    print("=" * 80)
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        print("-" * 80)
        
        result = run_vanilla_rag(question)
        
        chunks = result.get("retrieved_chunks", [])
        answer = result.get("answer", "(no answer)")
        
        # Show top 3 chunk IDs retrieved
        top_ids = [c["chunk_id"] for c in chunks[:3]]
        print(f"Retrieved (top 3): {', '.join(top_ids)}")
        
        print(f"\nAnswer:\n{answer}")
        print("=" * 80)


if __name__ == "__main__":
    main()