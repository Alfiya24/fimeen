"""
Generator node: takes question + retrieved chunks, produces grounded answer.

Uses Claude Haiku 4.5 with:
- Grounded-in-context system prompt
- Retry logic on rate limits / transient errors
- Graceful fallback on persistent failure
"""

import logging
from typing import TypedDict

from anthropic import Anthropic, APIStatusError, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import ANTHROPIC_API_KEY

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 800

logger = logging.getLogger(__name__)

# Module-level singleton
_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


GENERATOR_SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about SEC 10-K filings.

RULES:
1. Answer ONLY using information from the provided context chunks.
2. If the context does not contain the answer, say so explicitly — do not invent facts.
3. Cite chunk IDs inline using [chunk_id] format when stating specific facts.
4. Be concise. 3-5 sentences typically sufficient.
5. Never fabricate numbers, dates, or attributions. If unsure, say so."""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks for inclusion in the user prompt."""
    parts = []
    for chunk in chunks:
        parts.append(
            f"[{chunk['chunk_id']}] (Company: {chunk['company']})\n{chunk['text']}\n"
        )
    return "\n---\n".join(parts)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_claude(question: str, context: str) -> str:
    """
    Single API call to Claude with retry on transient failures.
    
    Retries up to 3 times with exponential backoff (2s, 4s, 8s).
    """
    client = _get_client()
    
    user_message = f"""Context from SEC filings:

{context}

---

Question: {question}

Answer using only the context above. Cite chunk IDs for specific facts."""
    
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=GENERATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    
    return response.content[0].text


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Generate an answer to the question using the retrieved chunks.
    
    Returns the answer text, or a fallback message if generation fails.
    """
    if not chunks:
        return "I couldn't find relevant information in the SEC filings to answer this question."
    
    context = _format_context(chunks)
    
    try:
        return _call_claude(question, context)
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded after retries: {e}")
        return "I'm currently rate-limited. Please try again in a moment."
    except APIConnectionError as e:
        logger.error(f"API connection failed after retries: {e}")
        return "I'm having trouble connecting to the model right now. Please try again."
    except APIStatusError as e:
        logger.error(f"API returned error status: {e}")
        return f"The model returned an error: {e.message}. Please try again."
    except Exception as e:
        logger.exception(f"Unexpected error in generator: {e}")
        return "An unexpected error occurred. Please try again."


# LangGraph-compatible node signature
class GeneratorState(TypedDict, total=False):
    question: str
    retrieved_chunks: list[dict]
    answer: str


def generator_node(state: GeneratorState) -> GeneratorState:
    """
    LangGraph node: takes state with 'question' and 'retrieved_chunks',
    returns state with 'answer'.
    """
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    
    answer = generate_answer(question, chunks)
    return {"answer": answer}