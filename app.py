"""
FIMEEN — Streamlit UI

Financial Q&A with verification-first RAG on SEC 10-K filings.
"""

import json
import time

import streamlit as st

from src.graph.vanilla_rag import run_vanilla_rag
from src.graph.fimeen import run_fimeen


# --------------------------------------------------------------------------
# Page config — must be first Streamlit command
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="FIMEEN",
    page_icon="🧠",
    layout="wide",
)

# --------------------------------------------------------------------------
# Custom styling
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }

    h2, h3 {
        letter-spacing: -0.02em;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
    }

    [data-testid="stSpinner"] > div > div {
        color: #6366F1;
    }

    .fimeen-tagline {
        font-size: 1.05rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.3rem;
    }

    .fimeen-description {
        color: #4B5563;
        font-size: 0.98rem;
        margin-bottom: 1.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🧠 FIMEEN")
st.markdown('<div class="fimeen-tagline">Verification-first financial Q&A.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="fimeen-description">FIMEEN fact-checks retrieval-augmented answers before showing them.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "FIMEEN fact-checks RAG answers before showing them.  \n"
        "Built for financial Q&A where hallucinated numbers cost real money."
    )

    st.markdown("### Corpus")
    st.markdown(
        "SEC 10-K filings  \n"
        "Apple, Microsoft, and NVIDIA fiscal year reports"
    )

    st.markdown("### Tech")
    st.markdown(
        "LangGraph • Claude Haiku 4.5 • MongoDB Atlas Vector Search"
    )

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
st.subheader("Ask a question")

example_questions = [
    "What was Microsoft's cloud revenue growth in fiscal 2024?",
    "What are NVIDIA's main risks from U.S. export controls?",
    "What new products did Apple launch in fiscal 2024?",
    "How did Microsoft's Azure growth compare to prior year?",
]

question = st.text_input(
    "Your question",
    placeholder="e.g. What was Microsoft's cloud revenue growth in fiscal 2024?",
    label_visibility="collapsed",
    key="question_input",
)

run_button = st.button("Run", type="primary", use_container_width=False)

st.markdown("**Examples**")
example_cols = st.columns(len(example_questions))
for i, eq in enumerate(example_questions):
    with example_cols[i]:
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state["pending_question"] = eq
            st.rerun()

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    run_button = True

st.markdown("---")


# --------------------------------------------------------------------------
# Verdict helpers
# --------------------------------------------------------------------------
VERDICT_ICONS = {
    "Supported": "✅",
    "Contradicted": "❌",
    "Unsupported": "⚠️",
}


# --------------------------------------------------------------------------
# Vanilla RAG renderer
# --------------------------------------------------------------------------
def render_vanilla_rag(question: str):
    st.subheader("Baseline answer")
    st.caption("Standard retrieval-augmented generation without verification.")

    with st.spinner("Running baseline answer..."):
        t_start = time.time()
        result = run_vanilla_rag(question)
        latency = time.time() - t_start

    answer = result.get("answer", "")
    st.markdown("#### Answer")
    st.markdown(answer if answer else "_(no answer generated)_")

    chunks = result.get("retrieved_chunks", [])
    with st.expander(f"Retrieved context ({len(chunks)} chunks)", expanded=False):
        for c in chunks:
            score = c.get("score", 0)
            st.markdown(
                f"**{c.get('company', 'Unknown')}** · `{c.get('chunk_id', '')}` · score `{score:.3f}`"
            )
            st.caption(c.get("text", "")[:220] + ("..." if len(c.get("text", "")) > 220 else ""))
            st.markdown("")

    st.metric("Latency", f"{latency:.2f}s")
    return result, latency


# --------------------------------------------------------------------------
# FIMEEN renderer
# --------------------------------------------------------------------------
def render_fimeen(question: str, vanilla_answer: str = ""):
    st.subheader("Verified answer")
    st.caption("Retrieval, generation, claim checking, and answer correction.")

    with st.spinner("Running FIMEEN..."):
        t_start = time.time()
        result = run_fimeen(question)
        latency = time.time() - t_start

    corrected = result.get("corrected_output", {})
    stats = corrected.get("stats", {})
    verifications = result.get("verifications", [])

    corrected_answer = corrected.get("corrected_answer", "_(no answer generated)_")
    st.markdown("#### Answer")
    st.markdown(corrected_answer)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Claims", stats.get("total_claims", 0))
    with col2:
        faithfulness = stats.get("faithfulness_rate", 0) * 100
        st.metric("Faithfulness", f"{faithfulness:.0f}%")
    with col3:
        caught = stats.get("contradicted", 0) + stats.get("unsupported", 0)
        st.metric("Issues caught", caught)

    with st.expander(f"Verification trace ({len(verifications)} claims)", expanded=True):
        for i, v in enumerate(verifications, 1):
            verdict = v.get("verdict", "Unsupported")
            icon = VERDICT_ICONS.get(verdict, "⚠️")

            st.markdown(f"**{i}. {icon} {verdict}**")
            st.markdown(v.get("claim", ""))

            evidence = v.get("evidence", "")
            chunk_id = v.get("chunk_id", "")
            reasoning = v.get("reasoning", "")

            if evidence:
                evidence_preview = evidence[:220] + ("..." if len(evidence) > 220 else "")
                st.caption(f"Evidence: “{evidence_preview}” [{chunk_id}]")

            if reasoning:
                st.caption(f"Reasoning: {reasoning}")

            st.markdown("")

    dropped = corrected.get("dropped_claims", [])
    if dropped:
        with st.expander(f"Removed claims ({len(dropped)})", expanded=False):
            for d in dropped:
                reason = d.get("reason", "Unsupported")
                icon = VERDICT_ICONS.get(reason, "⚠️")
                st.markdown(f"**{icon} {reason}**")
                st.markdown(d.get("claim", ""))
                if d.get("reasoning"):
                    st.caption(d["reasoning"])
                st.markdown("")

    st.metric("Latency", f"{latency:.2f}s")

    export_data = {
        "question": question,
        "baseline_answer": vanilla_answer,
        "fimeen": {
            "corrected_answer": corrected.get("corrected_answer", ""),
            "stats": stats,
            "verifications": verifications,
            "dropped_claims": corrected.get("dropped_claims", []),
        },
    }

    st.download_button(
        label="Download result (JSON)",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name="fimeen_result.json",
        mime="application/json",
        use_container_width=False,
    )

    return result, latency


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if run_button and question.strip():
    st.markdown("### Question")
    st.info(question)

    vanilla_result, _ = render_vanilla_rag(question)
    st.markdown("---")
    render_fimeen(question, vanilla_answer=vanilla_result.get("answer", ""))

elif run_button:
    st.warning("Please enter a question first.")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "FIMEEN • Financial Q&A with verification-first RAG"
)