"""
Corrector node: produces a cleaned-up answer based on verification results.

Strategy (v1): deterministic, structured output.
- Kept claims: Supported claims with evidence
- Dropped claims: Contradicted + Unsupported, with reasons

This design prioritizes transparency over prose fluency —
users SEE which claims survived verification.
"""

from typing import TypedDict


def correct_answer(original_answer: str, verifications: list[dict]) -> dict:
    """
    Given the original answer and verification results, produce a corrected output.
    
    Returns a dict with:
    - corrected_answer: prose-formatted final answer (kept claims only)
    - kept_claims: list of Supported claims
    - dropped_claims: list of Contradicted/Unsupported claims with reasons
    - stats: counts of Supported/Contradicted/Unsupported
    """
    kept_claims = []
    dropped_claims = []
    
    supported_count = 0
    contradicted_count = 0
    unsupported_count = 0
    
    for v in verifications:
        verdict = v.get("verdict", "Unsupported")
        
        if verdict == "Supported":
            kept_claims.append({
                "claim": v["claim"],
                "evidence": v.get("evidence", ""),
                "chunk_id": v.get("chunk_id", ""),
            })
            supported_count += 1
        elif verdict == "Contradicted":
            dropped_claims.append({
                "claim": v["claim"],
                "reason": "Contradicted",
                "evidence": v.get("evidence", ""),
                "chunk_id": v.get("chunk_id", ""),
                "reasoning": v.get("reasoning", ""),
            })
            contradicted_count += 1
        else:  # Unsupported or anything else
            dropped_claims.append({
                "claim": v["claim"],
                "reason": "Unsupported",
                "evidence": "",
                "chunk_id": "",
                "reasoning": v.get("reasoning", "No evidence found."),
            })
            unsupported_count += 1
    
    # Build corrected prose answer: list of kept claims with citations
    if not kept_claims:
        corrected_answer = (
            "Based on the retrieved documents, no claims could be verified with "
            "confidence. The original answer contained statements that could not "
            "be supported by the available evidence."
        )
    else:
        lines = ["Based on the retrieved documents, the following can be verified:\n"]
        for i, k in enumerate(kept_claims, 1):
            citation = f" [{k['chunk_id']}]" if k['chunk_id'] else ""
            lines.append(f"{i}. {k['claim']}.{citation}")
        corrected_answer = "\n".join(lines)
    
    total = supported_count + contradicted_count + unsupported_count
    faithfulness_rate = supported_count / total if total > 0 else 0.0
    
    return {
        "original_answer": original_answer,
        "corrected_answer": corrected_answer,
        "kept_claims": kept_claims,
        "dropped_claims": dropped_claims,
        "stats": {
            "total_claims": total,
            "supported": supported_count,
            "contradicted": contradicted_count,
            "unsupported": unsupported_count,
            "faithfulness_rate": faithfulness_rate,
        },
    }


# LangGraph-compatible node
class CorrectorState(TypedDict, total=False):
    answer: str
    verifications: list[dict]
    corrected_output: dict


def corrector_node(state: CorrectorState) -> CorrectorState:
    """
    LangGraph node: takes 'answer' + 'verifications', returns 'corrected_output'.
    """
    answer = state.get("answer", "")
    verifications = state.get("verifications", [])
    corrected_output = correct_answer(answer, verifications)
    return {"corrected_output": corrected_output}