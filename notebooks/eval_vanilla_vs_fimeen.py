"""
Evaluation harness: compare Vanilla RAG vs FIMEEN on the same 5 questions.

Measures:
- Vanilla RAG answer
- FIMEEN claim counts and verdicts
- Faithfulness rate
- Hallucinations caught (Contradicted + Unsupported)
- Per-pipeline latency

Output: structured results + summary table for Day 8 demo.
"""

import json
import time
from pathlib import Path

from src.graph.vanilla_rag import run_vanilla_rag
from src.graph.fimeen import run_fimeen


TEST_QUESTIONS = [
    "What was Microsoft's cloud revenue growth in fiscal 2024?",
    "What are NVIDIA's main risks from U.S. export controls?",
    "What new products did Apple launch in fiscal 2024?",
    "How did Microsoft's Azure growth compare to prior year?",
    "What is NVIDIA's policy on supply chain and demand estimates?",
]

OUTPUT_FILE = Path("docs/eval_results.json")


def evaluate_question(question: str) -> dict:
    """Run both pipelines on one question, return structured comparison."""
    
    # Vanilla RAG
    t_start = time.time()
    vanilla_result = run_vanilla_rag(question)
    vanilla_latency = time.time() - t_start
    
    # FIMEEN
    t_start = time.time()
    fimeen_result = run_fimeen(question)
    fimeen_latency = time.time() - t_start
    
    # Extract FIMEEN stats
    corrected = fimeen_result.get("corrected_output", {})
    stats = corrected.get("stats", {})
    
    return {
        "question": question,
        "vanilla_rag": {
            "answer": vanilla_result.get("answer", ""),
            "latency_sec": round(vanilla_latency, 2),
        },
        "fimeen": {
            "final_answer": corrected.get("corrected_answer", ""),
            "total_claims": stats.get("total_claims", 0),
            "supported": stats.get("supported", 0),
            "contradicted": stats.get("contradicted", 0),
            "unsupported": stats.get("unsupported", 0),
            "faithfulness_rate": round(stats.get("faithfulness_rate", 0) * 100, 1),
            "hallucinations_caught": stats.get("contradicted", 0) + stats.get("unsupported", 0),
            "dropped_claims": corrected.get("dropped_claims", []),
            "latency_sec": round(fimeen_latency, 2),
        },
    }


def print_comparison(result: dict) -> None:
    """Pretty-print one question's results."""
    print("=" * 80)
    print(f"Q: {result['question']}")
    print("=" * 80)
    
    print(f"\n[VANILLA RAG] ({result['vanilla_rag']['latency_sec']}s)")
    print(f"{result['vanilla_rag']['answer'][:250]}...")
    
    f = result["fimeen"]
    print(f"\n[FIMEEN] ({f['latency_sec']}s)")
    print(f"Claims: {f['total_claims']} | "
          f"Supported: {f['supported']} | "
          f"Contradicted: {f['contradicted']} | "
          f"Unsupported: {f['unsupported']}")
    print(f"Faithfulness: {f['faithfulness_rate']}%")
    print(f"Hallucinations caught: {f['hallucinations_caught']}")
    
    if f["dropped_claims"]:
        print(f"\nDROPPED:")
        for d in f["dropped_claims"]:
            print(f"  - [{d['reason']}] {d['claim']}")
    
    print()


def print_summary(results: list[dict]) -> None:
    """Print summary table across all questions."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    print(f"{'Q#':<4} {'Claims':<8} {'Sup':<5} {'Con':<5} {'Uns':<5} "
          f"{'Faith%':<8} {'Caught':<8} {'Van(s)':<8} {'FIM(s)':<8}")
    print("-" * 70)
    
    for i, r in enumerate(results, 1):
        f = r["fimeen"]
        v = r["vanilla_rag"]
        print(f"Q{i:<3} "
              f"{f['total_claims']:<8} "
              f"{f['supported']:<5} "
              f"{f['contradicted']:<5} "
              f"{f['unsupported']:<5} "
              f"{f['faithfulness_rate']:<8} "
              f"{f['hallucinations_caught']:<8} "
              f"{v['latency_sec']:<8} "
              f"{f['latency_sec']:<8}")
    
    # Aggregates
    total_claims = sum(r["fimeen"]["total_claims"] for r in results)
    total_supported = sum(r["fimeen"]["supported"] for r in results)
    total_caught = sum(r["fimeen"]["hallucinations_caught"] for r in results)
    avg_vanilla = sum(r["vanilla_rag"]["latency_sec"] for r in results) / len(results)
    avg_fimeen = sum(r["fimeen"]["latency_sec"] for r in results) / len(results)
    
    print("-" * 70)
    print(f"TOTAL: {total_claims} claims | "
          f"{total_supported} supported ({100*total_supported/total_claims:.1f}%) | "
          f"{total_caught} issues caught")
    print(f"AVG LATENCY: Vanilla {avg_vanilla:.2f}s | FIMEEN {avg_fimeen:.2f}s "
          f"({avg_fimeen/avg_vanilla:.1f}x slower)")


def main():
    print(f"Running comparison on {len(TEST_QUESTIONS)} questions\n")
    
    results = []
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] Processing: {q[:60]}...")
        results.append(evaluate_question(q))
    
    # Print detailed results
    print("\n")
    for r in results:
        print_comparison(r)
    
    # Print summary
    print_summary(results)
    
    # Save to disk for Day 8 demo
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✅ Full results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()