"""
End-to-end test of the full FIMEEN pipeline on one question.

Runs all 5 nodes and shows each stage's output.
"""

from src.graph.fimeen import run_fimeen


def main():
    question = "What was Microsoft's cloud revenue growth in fiscal 2024?"
    
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    result = run_fimeen(question)
    
    # Stage 1: Retrieval
    chunks = result.get("retrieved_chunks", [])
    print(f"\n[1] RETRIEVED: {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"    - [{c['chunk_id']}] score={c.get('score', 0):.3f}")
    
    # Stage 2: Generation
    answer = result.get("answer", "")
    print(f"\n[2] GENERATED ANSWER:\n{answer[:300]}{'...' if len(answer) > 300 else ''}")
    
    # Stage 3: Decomposition
    claims = result.get("claims", [])
    print(f"\n[3] DECOMPOSED: {len(claims)} atomic claims")
    for i, c in enumerate(claims, 1):
        print(f"    {i}. {c}")
    
    # Stage 4: Verification
    verifications = result.get("verifications", [])
    print(f"\n[4] VERIFIED: {len(verifications)} claims checked")
    verdict_counts = {}
    for v in verifications:
        verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1
    for verdict, count in verdict_counts.items():
        print(f"    {verdict}: {count}")
    
    # Stage 5: Correction
    corrected = result.get("corrected_output", {})
    print(f"\n[5] FIMEEN FINAL OUTPUT:")
    print("-" * 80)
    print(corrected.get("corrected_answer", ""))
    print("-" * 80)
    
    stats = corrected.get("stats", {})
    print(f"\nFaithfulness rate: {stats.get('faithfulness_rate', 0)*100:.0f}% "
          f"({stats.get('supported', 0)}/{stats.get('total_claims', 0)} claims supported)")
    
    dropped = corrected.get("dropped_claims", [])
    if dropped:
        print(f"\nDROPPED CLAIMS ({len(dropped)}):")
        for d in dropped:
            print(f"  - [{d['reason']}] {d['claim']}")


if __name__ == "__main__":
    main()