"""
Decomposer node: breaks a generated answer into atomic verifiable claims.

This is the first step of FIMEEN's self-verification loop. Each claim
must be independently verifiable against source documents.
"""

import json
import logging
import re
from typing import TypedDict

from anthropic import Anthropic, APIStatusError, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import ANTHROPIC_API_KEY

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1500

logger = logging.getLogger(__name__)

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


DECOMPOSER_SYSTEM_PROMPT = """You are a claim extraction specialist for financial documents.

Your job: break an answer into ATOMIC VERIFIABLE CLAIMS.

An atomic claim:
- Is a single, standalone factual statement
- Can be independently verified against source text
- Contains one specific fact (one number, one attribution, one event, etc.)
- Is self-contained (does not require reading other claims to understand)

Rules:
1. Preserve specific numbers, percentages, dates, company names, and product names.
2. Strip citations like [CHUNK_ID] — they are metadata, not part of the claim.
3. Ignore hedging or meta-language like "Based on the context..." or "The filing states...". Extract the underlying fact.
4. If the answer explicitly says information is missing or uncertain (e.g., "the context does not provide..."), return an empty list of claims.
5. Do NOT invent or add facts not in the answer.

Output format: return a JSON object with a single key "claims", whose value is an array of strings.

Example 1:
Answer: "Microsoft Cloud revenue increased 23% to $137.4 billion in fiscal year 2024 [MSFT_chunk_001]."
Output: {"claims": ["Microsoft Cloud revenue grew 23% year-over-year in fiscal year 2024", "Microsoft Cloud revenue was $137.4 billion in fiscal year 2024"]}

Example 2:
Answer: "Apple launched the iPhone 16, iPhone 16 Plus, iPhone 16 Pro, iPhone 16 Pro Max, Apple Watch Series 10, and AirPods 4 in the fourth quarter of fiscal 2024 [AAPL_chunk_000]."
Output: {"claims": ["Apple launched iPhone 16 in Q4 fiscal 2024", "Apple launched iPhone 16 Plus in Q4 fiscal 2024", "Apple launched iPhone 16 Pro in Q4 fiscal 2024", "Apple launched iPhone 16 Pro Max in Q4 fiscal 2024", "Apple launched Apple Watch Series 10 in Q4 fiscal 2024", "Apple launched AirPods 4 in Q4 fiscal 2024"]}

Example 3:
Answer: "The context does not provide explicit prior-year Azure revenue figures, so I cannot make a direct comparison."
Output: {"claims": []}

Return ONLY the JSON object. No preamble, no explanation, no markdown fences."""


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_claude(answer: str) -> str:
    """Single API call to Claude for decomposition, with retries."""
    client = _get_client()
    
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=DECOMPOSER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Answer to decompose:\n\n{answer}"}],
    )
    
    return response.content[0].text


def _parse_claims(raw_response: str) -> list[str]:
    """
    Extract claims list from Claude's JSON response.
    Handles minor format variations (markdown fences, preambles).
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE)
    
    try:
        parsed = json.loads(cleaned)
        claims = parsed.get("claims", [])
        # Sanity filter: drop empty or whitespace-only claims
        return [c.strip() for c in claims if c and c.strip()]
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse decomposer JSON: {e}")
        logger.error(f"Raw response: {raw_response[:500]}")
        return []


def decompose_answer(answer: str) -> list[str]:
    """
    Decompose an answer into atomic verifiable claims.
    
    Returns an empty list if decomposition fails or answer has no verifiable claims.
    """
    if not answer or not answer.strip():
        return []
    
    try:
        raw = _call_claude(answer)
        return _parse_claims(raw)
    except RateLimitError:
        logger.error("Rate limit exceeded during decomposition")
        return []
    except (APIConnectionError, APIStatusError) as e:
        logger.error(f"API error during decomposition: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error in decomposer: {e}")
        return []


# LangGraph-compatible node
class DecomposerState(TypedDict, total=False):
    answer: str
    claims: list[str]


def decomposer_node(state: DecomposerState) -> DecomposerState:
    """
    LangGraph node: takes state with 'answer', returns state with 'claims'.
    """
    answer = state.get("answer", "")
    claims = decompose_answer(answer)
    return {"claims": claims}