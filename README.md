# 🧠 FIMEEN

**Live app:** [fimeen-ai.streamlit.app](https://fimeen-ai.streamlit.app)
**Repo:** [github.com/Alfiya24/fimeen](https://github.com/Alfiya24/fimeen)

Self-verifying retrieval-augmented generation for financial Q&A. FIMEEN fact-checks RAG answers before showing them — decomposing generated text into atomic claims, verifying each against source evidence, and dropping unsupported claims.

---

## Why FIMEEN

Standard RAG systems hallucinate facts in their answers. Users can't tell which claims are grounded in the source documents and which ones the LLM made up. For financial Q&A — where a wrong number or fabricated event has real consequences — faithfulness matters more than speed.

FIMEEN does inference-time verification without fine-tuning. It runs every generated answer through a five-stage pipeline: retrieve, generate, decompose, verify, correct. The final output is an answer with only the claims that can be directly backed by evidence, plus a transparent drop-list showing what was removed and why.

---

## Architecture

Five-node LangGraph pipeline with a shared state dict flowing through each stage.

```
START → Retriever → Generator → Decomposer → Verifier → Corrector → END
```

| Node | What it does |
|---|---|
| **Retriever** | MongoDB Atlas Vector Search with sentence-transformers embeddings over chunked SEC filings |
| **Generator** | Claude Haiku 4.5 with a strict grounded-answer prompt |
| **Decomposer** | Extracts atomic, independently verifiable claims from the answer |
| **Verifier** | Checks each claim against retrieved chunks. Returns Supported, Contradicted, or Unsupported, with a direct evidence quote |
| **Corrector** | Deterministic assembly — kept claims with citations, dropped claims with reasons, faithfulness stats |

The Corrector is rule-based (not LLM-based) to avoid introducing new hallucinations during correction.

---

## Stack

- **Orchestration:** LangGraph 0.2.50
- **LLM:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
- **Retrieval:** MongoDB Atlas Vector Search, cosine similarity
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dim, local CPU)
- **UI:** Streamlit 1.40.1
- **Error handling:** Tenacity retries with exponential backoff on all LLM calls

---

## Quickstart

```bash
git clone https://github.com/Alfiya24/fimeen.git
cd fimeen
pip install -r requirements.txt

# Set secrets in .env
echo "ANTHROPIC_API_KEY=your_key_here" > .env
echo "MONGODB_URI=your_atlas_uri_here" >> .env

streamlit run app.py
```

Open [localhost:8501](http://localhost:8501) and click an example question.

---

## Corpus

SEC 10-K MD&A sections for fiscal-year reports:

- Apple Inc. (FY2024)
- Microsoft Corporation (FY2024)
- NVIDIA Corporation (FY2025)

191 chunks total, average 1842 characters per chunk, stored in MongoDB Atlas.

---

## Evaluation

Five financial questions, 52 atomic claims evaluated across vanilla RAG and FIMEEN.

| Metric | Value |
|---|---|
| Total claims | 52 |
| Supported | 51 |
| Contradicted | 1 |
| Unsupported | 0 |
| Overall faithfulness | 98.1% |
| Avg vanilla RAG latency | 3.89s |
| Avg FIMEEN latency | 30.84s |
| Latency overhead | 7.9× |

Red-team testing with four synthetic hallucinations (two true + two fabricated claims) produced 4/4 correct verdicts. Fabricated numbers (e.g., "$250B" when actual is $137.4B) and fabricated events (e.g., "Microsoft acquired OpenAI") were caught with accurate contradicting evidence.

Full results in `docs/eval_results.json`.

---

## Project structure

```
fimeen/
├── app.py                          # Streamlit UI
├── requirements.txt
├── pyproject.toml
├── .streamlit/config.toml          # Theme config
├── src/
│   ├── config.py                   # Env loading
│   ├── graph/
│   │   ├── fimeen.py               # Full 5-node LangGraph pipeline
│   │   ├── vanilla_rag.py          # 2-node baseline
│   │   └── nodes/
│   │       ├── retriever.py
│   │       ├── generator.py
│   │       ├── decomposer.py
│   │       ├── verifier.py
│   │       └── corrector.py
│   └── retrieval/
│       ├── fetch_filings.py        # SEC EDGAR downloader
│       ├── chunker.py              # Sentence-level chunker
│       └── ingest.py               # Embed + push to MongoDB
├── notebooks/
│   ├── test_retriever.py
│   ├── test_generator.py
│   ├── test_decomposer.py
│   ├── test_verifier.py
│   ├── test_corrector.py
│   ├── test_fimeen.py              # End-to-end pipeline test
│   └── eval_vanilla_vs_fimeen.py   # Comparison harness
└── docs/
    ├── eval_results.json
    └── screenshots/
```

---

## Known limitations

**Narrative-level conflation is out of scope.** Claim-level verification cannot catch cases where every individual claim is factually supported but the synthesis misleadingly combines topics. The original RLFKV paper addresses narrative-level faithfulness through fine-tuning — FIMEEN does not.

**Non-determinism across runs.** Same question, same pipeline, different verdicts possible on different runs. This is fundamental to LLM-backed verification. FIMEEN's red-team catch rate (4/4 on synthetic adversarial claims) is the stable signal.

**Latency overhead is real.** FIMEEN makes N+2 LLM calls per query (1 Generator + 1 Decomposer + N Verifier calls, one per claim). At 7.9× the latency of vanilla RAG, this is a legitimate trade-off — acceptable for high-stakes financial Q&A where faithfulness outweighs speed.

**Cold-start delay on Streamlit Cloud.** Free-tier apps sleep after inactivity. First request after idle takes 30–90s to wake the container and load the embedding model.

---

## Security considerations

- **Prompt injection:** The strict grounded system prompt on the Generator mitigates user attempts to override instructions, but is not a full defense. FIMEEN is designed for read-only Q&A; no tool-calling or agentic actions.
- **Data handling:** All corpus data is public SEC filings. No PII, no proprietary data.
- **Secrets:** `.env` file is gitignored. Production deployment uses Streamlit Cloud secrets manager (TOML-format environment variables).
- **Rate limits:** Tenacity retries with exponential backoff handle transient Anthropic API rate limits. Singleton clients avoid connection thrash.

---

## Inspiration

FIMEEN is inspired by the RLFKV paper (February 2026) on RAG hallucination mitigation for financial filings. RLFKV trains a faithfulness reward model via reinforcement learning. FIMEEN explores the inference-time alternative — no fine-tuning, just a verification pipeline — to see how far prompt engineering and structured orchestration can push faithfulness before training-time signals are needed.

---

## License

MIT

---

Built for the Decoding Data Science "Building AI Challenge" (April 2026).