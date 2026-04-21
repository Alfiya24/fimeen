"""
Test the verifier on yesterday's decomposed claims.

Runs the full Retriever → Generator → Decomposer → Verifier chain
for one question, showing which claims pass/fail verification.
"""

from src.graph.nodes.retriever import retrieve_chunks
from src.graph.nodes.generator import generate_answer
from src.graph.nodes.decomposer import decompose_answer
from src.graph.nodes.verifier import verify_claims


def test_one_question(question: str) -> None:
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # Retrieve
    chunks = retrieve_chunks(question, top_k=5)
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"\nRetrieved chunks: {', '.join(chunk_ids)}")
    
    # Generate
    answer = generate_answer(question, chunks)
    print(f"\nANSWER:\n{answer}\n")
    
    # Decompose
    claims = decompose_answer(answer)
    print(f"Decomposed into {len(claims)} claims.\n")
    
    # Verify
    print("VERIFICATION RESULTS:")
    print("-" * 80)
    verifications = verify_claims(claims, chunks)
    
    supported = 0
    contradicted = 0
    unsupported = 0
    
    for i, v in enumerate(verifications, 1):
        emoji = {"Supported": "✅", "Contradicted": "❌", "Unsupported": "⚠️"}[v["verdict"]]
        print(f"\n{i}. {emoji} [{v['verdict']}] {v['claim']}")
        if v["evidence"]:
            print(f"   Evidence: \"{v['evidence'][:150]}...\" [{v['chunk_id']}]")
        print(f"   Reasoning: {v['reasoning']}")
        
        if v["verdict"] == "Supported":
            supported += 1
        elif v["verdict"] == "Contradicted":
            contradicted += 1
        else:
            unsupported += 1
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {supported} Supported | {contradicted} Contradicted | {unsupported} Unsupported")
    print(f"Faithfulness rate: {supported}/{len(verifications)} = {100*supported/len(verifications):.0f}%" if verifications else "No claims")
    print("=" * 80)


def main():
    # Test 1: Q5 (the conflation case we just ran — keep for comparison)
    print("\n" + "#" * 80)
    print("# TEST 1: Real Q5 (conflation case)")
    print("#" * 80)
    test_question = "What is NVIDIA's policy on supply chain and demand estimates?"
    test_one_question(test_question)
    
    # Test 2: Red-team test — inject obviously false claims and see if verifier catches them
    print("\n" + "#" * 80)
    print("# TEST 2: Red-team with artificial hallucinations")
    print("#" * 80)
    
    from src.graph.nodes.retriever import retrieve_chunks
    from src.graph.nodes.verifier import verify_claims
    
    # Retrieve chunks for a Microsoft question
    chunks = retrieve_chunks("Microsoft cloud revenue", top_k=5)
    print(f"\nRetrieved {len(chunks)} chunks for context.\n")
    
    # Inject 4 claims: 2 true, 2 false
    test_claims = [
        "Microsoft Cloud revenue was $137.4 billion in fiscal year 2024",   # TRUE — should be Supported
        "Microsoft Cloud revenue grew 23% in fiscal year 2024",              # TRUE — should be Supported
        "Microsoft Cloud revenue was $250 billion in fiscal year 2024",      # FALSE — should be Contradicted
        "Microsoft acquired OpenAI in fiscal year 2024",                     # FALSE — should be Contradicted or Unsupported
    ]
    
    print("Testing 4 claims (2 true, 2 false):")
    for c in test_claims:
        print(f"  - {c}")
    
    verifications = verify_claims(test_claims, chunks)
    
    print("\n" + "-" * 80)
    for i, v in enumerate(verifications, 1):
        emoji = {"Supported": "✅", "Contradicted": "❌", "Unsupported": "⚠️"}[v["verdict"]]
        print(f"\n{i}. {emoji} [{v['verdict']}] {v['claim']}")
        if v["evidence"]:
            print(f"   Evidence: \"{v['evidence'][:150]}...\" [{v['chunk_id']}]")
        print(f"   Reasoning: {v['reasoning']}")


if __name__ == "__main__":
    main()