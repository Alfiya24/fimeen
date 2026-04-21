"""
Verifier node: checks each claim against retrieved chunks.

For each claim, returns a verdict:
- Supported: claim is directly backed by chunk text
- Contradicted: a chunk directly contradicts the claim
- Unsupported: no chunk backs or contradicts the claim

This is FIMEEN's core hallucination-catching step.
"""

import json
import logging
import re
from typing import TypedDict, Literal

from anthropic import Anthropic, APIStatusError, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import ANTHROPIC_API_KEY

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 500

logger = logging.getLogger(__name__)

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


Verdict = Literal["Supported", "Contradicted", "Unsupported"]


VERIFIER_SYSTEM_PROMPT = """You are a strict fact-checker for financial documents.

Your job: determine whether a specific CLAIM is supported by the provided CHUNKS.

Return ONE of three verdicts:

1. "Supported" — A chunk directly confirms the claim. Numbers, names, dates, and attributions must match exactly (paraphrases of the same fact are fine, but don't accept approximate or rounded numbers as matches for specific figures).

2. "Contradicted" — A chunk directly states something that conflicts with the claim. The claim is wrong.

3. "Unsupported" — No chunk backs or contradicts the claim. The claim may or may not be true, but cannot be verified from the provided chunks.

Rules:
- Be strict. When in doubt, prefer "Unsupported" over "Supported". Faithfulness matters more than charitability.
- A claim is "Supported" only if you can quote a specific passage from a chunk that confirms it.
- Do NOT use outside knowledge. Only the provided chunks count as evidence.
- If the claim is vaguely related to a chunk but the specific fact isn't stated, return "Unsupported".

Output format: return a JSON object with:
- "verdict": one of "Supported", "Contradicted", "Unsupported"
- "evidence": a direct quote from the chunk that supports or contradicts (empty string "" if Unsupported)
- "chunk_id": the chunk_id where evidence was found (empty string "" if Unsupported)
- "reasoning": one short sentence explaining your call

Example 1:
Claim: "Microsoft Cloud revenue was $137.4 billion in fiscal year 2024"
Chunks: [MSFT_chunk_001]: "...Microsoft Cloud revenue increased 23% to $137.4 billion..."
Output: {"verdict": "Supported", "evidence": "Microsoft Cloud revenue increased 23% to $137.4 billion", "chunk_id": "MSFT_chunk_001", "reasoning": "The chunk explicitly states the $137.4 billion figure for Microsoft Cloud revenue."}

Example 2:
Claim: "Apple's iPhone 17 was launched in fiscal year 2024"
Chunks: [AAPL_chunk_000]: "...iPhone 16, iPhone 16 Plus, iPhone 16 Pro and iPhone 16 Pro Max..."
Output: {"verdict": "Contradicted", "evidence": "iPhone 16, iPhone 16 Plus, iPhone 16 Pro and iPhone 16 Pro Max", "chunk_id": "AAPL_chunk_000", "reasoning": "The chunk specifies iPhone 16 models, not iPhone 17."}

Example 3:
Claim: "NVIDIA has a formal supply chain policy documented in their filings"
Chunks: [NVDA_chunk_005]: "...export controls may effectively exclude NVIDIA from markets..."
Output: {"verdict": "Unsupported", "evidence": "", "chunk_id": "", "reasoning": "No chunk discusses NVIDIA's formal supply chain policy."}

Return ONLY the JSON object. No preamble, no markdown fences."""


def _format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks for the verifier prompt."""
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk['chunk_id']}] (Company: {chunk['company']})\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_claude(claim: str, chunks_text: str) -> str:
    client = _get_client()
    
    user_message = f"""CLAIM to verify:
{claim}

CHUNKS (evidence sources):
{chunks_text}

Return your JSON verdict."""
    
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=VERIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    
    return response.content[0].text


def _parse_verdict(raw_response: str) -> dict:
    """Parse verifier JSON response, handling format variations."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE)
    
    try:
        parsed = json.loads(cleaned)
        # Sanity check + defaults
        return {
            "verdict": parsed.get("verdict", "Unsupported"),
            "evidence": parsed.get("evidence", ""),
            "chunk_id": parsed.get("chunk_id", ""),
            "reasoning": parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse verifier JSON: {e}")
        logger.error(f"Raw response: {raw_response[:300]}")
        return {
            "verdict": "Unsupported",
            "evidence": "",
            "chunk_id": "",
            "reasoning": "Verifier failed to return valid JSON.",
        }


def verify_claim(claim: str, chunks: list[dict]) -> dict:
    """
    Verify a single claim against the retrieved chunks.
    
    Returns a dict with: claim, verdict, evidence, chunk_id, reasoning.
    """
    if not claim.strip():
        return {
            "claim": claim,
            "verdict": "Unsupported",
            "evidence": "",
            "chunk_id": "",
            "reasoning": "Empty claim.",
        }
    
    if not chunks:
        return {
            "claim": claim,
            "verdict": "Unsupported",
            "evidence": "",
            "chunk_id": "",
            "reasoning": "No chunks provided for verification.",
        }
    
    chunks_text = _format_chunks(chunks)
    
    try:
        raw = _call_claude(claim, chunks_text)
        result = _parse_verdict(raw)
        result["claim"] = claim
        return result
    except Exception as e:
        logger.exception(f"Verification failed for claim: {e}")
        return {
            "claim": claim,
            "verdict": "Unsupported",
            "evidence": "",
            "chunk_id": "",
            "reasoning": f"Verifier error: {type(e).__name__}",
        }


def verify_claims(claims: list[str], chunks: list[dict]) -> list[dict]:
    """Verify each claim in a list against shared chunks."""
    return [verify_claim(claim, chunks) for claim in claims]


# LangGraph-compatible node
class VerifierState(TypedDict, total=False):
    claims: list[str]
    retrieved_chunks: list[dict]
    verifications: list[dict]


def verifier_node(state: VerifierState) -> VerifierState:
    """
    LangGraph node: takes 'claims' + 'retrieved_chunks', returns 'verifications'.
    """
    claims = state.get("claims", [])
    chunks = state.get("retrieved_chunks", [])
    verifications = verify_claims(claims, chunks)
    return {"verifications": verifications}