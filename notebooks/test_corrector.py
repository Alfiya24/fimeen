"""
Test the corrector on a mixed set of verified claims.

Runs: Retrieve → Generate → Decompose → Verify → Correct
on the same red-team mix (2 true + 2 false claims).
"""

from src.graph.nodes.retriever import retrieve_chunks
from src.graph.nodes.verifier import verify_claims
from src.graph.nodes.corrector import correct_answer


def main():
    # Use the same red-team mix from verifier test
    chunks = retrieve_chunks("Microsoft cloud revenue", top_k=5)
    
    test_claims = [
        "Microsoft Cloud revenue was $137.4 billion in fiscal year 2024",
        "Microsoft Cloud revenue grew 23% in fiscal year 2024",
        "Microsoft Cloud revenue was $250 billion in fiscal year 2024",
        "Microsoft acquired OpenAI in fiscal year 2024",
    ]
    
    # Simulate the original answer
    original = (
        "Microsoft Cloud revenue was $137.4 billion in fiscal year 2024, growing 23%. "
        "However, it also reached $250 billion and Microsoft acquired OpenAI."
    )
    
    print("ORIGINAL ANSWER:")
    print(original)
    print()
    
    # Verify
    verifications = verify_claims(test_claims, chunks)
    
    # Correct
    result = correct_answer(original, verifications)
    
    print("=" * 80)
    print("CORRECTED ANSWER (FIMEEN output):")
    print("=" * 80)
    print(result["corrected_answer"])
    
    print("\n" + "=" * 80)
    print("KEPT CLAIMS (Supported):")
    print("=" * 80)
    for k in result["kept_claims"]:
        print(f"  ✅ {k['claim']}")
        print(f"     Evidence: \"{k['evidence'][:100]}...\" [{k['chunk_id']}]")
    
    print("\n" + "=" * 80)
    print("DROPPED CLAIMS:")
    print("=" * 80)
    for d in result["dropped_claims"]:
        emoji = "❌" if d["reason"] == "Contradicted" else "⚠️"
        print(f"  {emoji} [{d['reason']}] {d['claim']}")
        print(f"     Reasoning: {d['reasoning']}")
    
    print("\n" + "=" * 80)
    print("STATS:")
    print("=" * 80)
    stats = result["stats"]
    print(f"  Total claims: {stats['total_claims']}")
    print(f"  Supported:    {stats['supported']}")
    print(f"  Contradicted: {stats['contradicted']}")
    print(f"  Unsupported:  {stats['unsupported']}")
    print(f"  Faithfulness: {stats['faithfulness_rate']*100:.0f}%")


if __name__ == "__main__":
    main()