FIMEEN

Verification-first financial Q&A.
FIMEEN fact-checks retrieval-augmented answers before showing them.

What it does

FIMEEN answers questions on financial filings (SEC 10-K) and verifies every claim before returning the result.

Instead of trusting a generated answer, FIMEEN:

breaks the answer into atomic claims
checks each claim against source documents
removes or corrects unsupported statements

The final output contains only verified information grounded in filings.

Why this matters

Large language models are good at sounding correct — not at being correct.

In financial contexts, this is a problem:

incorrect numbers can lead to wrong decisions
hallucinated facts are hard to detect
confidence ≠ accuracy

FIMEEN addresses this by adding a verification layer on top of RAG.

Financial Q&A where every claim is checked against source filings.

How it works

FIMEEN follows a verification-first pipeline:

Retrieve
Relevant sections from SEC 10-K filings are fetched using vector search.
Generate
An initial answer is produced from retrieved context.
Decompose
The answer is split into individual factual claims.
Verify
Each claim is checked against the source documents and labeled:
Supported
Contradicted
Unsupported
Correct
Unsupported or incorrect claims are removed or rewritten.
Return
A cleaned, verified answer is shown to the user.
Data

FIMEEN currently operates on:

Apple — 10-K (FY2024)
Microsoft — 10-K (FY2024)
NVIDIA — 10-K (FY2025)

Focus is on MD&A (Management Discussion & Analysis) sections.

Tech Stack
Orchestration: LangGraph
LLM: Claude Haiku 4.5
Retrieval: MongoDB Atlas Vector Search
Embeddings: sentence-transformers
Frontend: Streamlit
Running the app

install dependencies
pip install -r requirements.txt

run the app
streamlit run app.py
Example queries
What was Microsoft's cloud revenue growth in fiscal 2024?
What are NVIDIA's main risks from U.S. export controls?
What new products did Apple launch in fiscal 2024?
How did Azure growth compare year-over-year?
Output structure

FIMEEN returns:

Verified answer (final output)
Claim statistics (faithfulness, issues caught)
Verification trace (optional, per-claim evidence)
Design principle

FIMEEN is built on a simple rule:

Do not show an answer unless it can be verified.

Future work
Expand corpus beyond 10-K filings
Improve claim decomposition and verification accuracy
Add support for portfolio-level and cross-company queries
Optimize latency for production use
License

MIT License
