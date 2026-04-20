"""
Test the decomposer on the 5 vanilla RAG answers from Day 3.

Shows how each grounded answer breaks down into atomic claims —
the foundation for per-claim verification in Day 4.
"""

from src.graph.vanilla_rag import run_vanilla_rag
from src.graph.nodes.decomposer import decompose_answer


TEST_QUESTIONS = [
    "What was Microsoft's cloud revenue growth in fiscal 2024?",
    "What are NVIDIA's main risks from U.S. export controls?",
    "What new products did Apple launch in fiscal 2024?",
    "How did Microsoft's Azure growth compare to prior year?",
    "What is NVIDIA's policy on supply chain and demand estimates?",
]


def main():
    print(f"Testing decomposer on {len(TEST_QUESTIONS)} vanilla RAG answers\n")
    print("=" * 80)
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        print("-" * 80)
        
        # Run vanilla RAG
        result = run_vanilla_rag(question)
        answer = result.get("answer", "")
        
        print(f"Answer (first 200 chars):\n{answer[:200]}...\n")
        
        # Decompose
        claims = decompose_answer(answer)
        
        print(f"Decomposed into {len(claims)} atomic claims:")
        for j, claim in enumerate(claims, 1):
            print(f"  {j}. {claim}")
        
        print("=" * 80)


if __name__ == "__main__":
    main()