"""
FIMEEN — Streamlit UI

Side-by-side comparison of vanilla RAG vs FIMEEN
for financial Q&A on SEC 10-K filings.
"""

import time

import streamlit as st

from src.graph.vanilla_rag import run_vanilla_rag
from src.graph.fimeen import run_fimeen


# --------------------------------------------------------------------------
# Page config — must be first Streamlit command
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="FIMEEN — Self-Verifying RAG for Financial Q&A",
    page_icon="🧠",
    layout="wide",
)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🧠 FIMEEN")
st.caption(
    "A self-verifying RAG system for financial Q&A. "
    "Compares vanilla RAG (generate answer, return) against FIMEEN "
    "(generate → decompose → verify → correct)."
)

st.markdown("---")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About FIMEEN")
    st.markdown(
        """
        **Corpus:** MD&A sections of  
        - Apple 10-K (FY2024)  
        - Microsoft 10-K (FY2024)  
        - NVIDIA 10-K (FY2025)
        
        **Stack:**  
        - LangGraph (orchestration)  
        - Claude Haiku 4.5 (generation + verification)  
        - MongoDB Atlas Vector Search (retrieval)  
        - sentence-transformers (embeddings)
        
        **Inspired by:** RLFKV paper on financial RAG hallucination mitigation (Feb 2026).  
        FIMEEN explores inference-time verification without fine-tuning.
        """
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

col_q, col_btn = st.columns([4, 1])

with col_q:
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What was Microsoft's cloud revenue growth in fiscal 2024?",
        label_visibility="collapsed",
        key="question_input",
    )

with col_btn:
    run_button = st.button("Run comparison", type="primary", use_container_width=True)

# Example question chips
st.markdown("**Try an example:**")
example_cols = st.columns(len(example_questions))
for i, eq in enumerate(example_questions):
    with example_cols[i]:
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state["pending_question"] = eq
            st.rerun()

# If an example was clicked, use it
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    run_button = True

st.markdown("---")


# --------------------------------------------------------------------------
# Verdict styling helpers
# --------------------------------------------------------------------------
VERDICT_COLORS = {
    "Supported": "✅",
    "Contradicted": "❌",
    "Unsupported": "⚠️",
}

VERDICT_BADGE = {
    "Supported": ":green-background[Supported]",
    "Contradicted": ":red-background[Contradicted]",
    "Unsupported": ":orange-background[Unsupported]",
}


# --------------------------------------------------------------------------
# Vanilla RAG renderer
# --------------------------------------------------------------------------
def render_vanilla_rag(question: str) -> None:
    st.subheader("🔹 Vanilla RAG")
    st.caption("Retrieve chunks → generate answer. No verification.")
    
    with st.spinner("Running vanilla RAG..."):
        t_start = time.time()
        result = run_vanilla_rag(question)
        latency = time.time() - t_start
    
    answer = result.get("answer", "")
    st.markdown("#### Answer")
    st.markdown(answer if answer else "_(no answer generated)_")
    
    chunks = result.get("retrieved_chunks", [])
    with st.expander(f"📚 Retrieved {len(chunks)} chunks", expanded=False):
        for c in chunks:
            score = c.get("score", 0)
            st.markdown(
                f"**[{c['chunk_id']}]** — {c['company']} — "
                f"score: `{score:.3f}`"
            )
            st.caption(c["text"][:200] + "...")
            st.markdown("")
    
    st.metric("⏱ Latency", f"{latency:.2f}s")


# --------------------------------------------------------------------------
# FIMEEN renderer
# --------------------------------------------------------------------------
def render_fimeen(question: str) -> None:
    st.subheader("✨ FIMEEN")
    st.caption("Retrieve → generate → decompose → verify → correct.")
    
    with st.spinner("Running FIMEEN (5 stages, may take 30-60s)..."):
        t_start = time.time()
        result = run_fimeen(question)
        latency = time.time() - t_start
    
    corrected = result.get("corrected_output", {})
    stats = corrected.get("stats", {})
    verifications = result.get("verifications", [])
    
    # -- Corrected answer
    st.markdown("#### Verified answer")
    corrected_answer = corrected.get("corrected_answer", "_(no answer)_")
    st.markdown(corrected_answer)
    
    # -- Stats row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Claims",
            stats.get("total_claims", 0),
        )
    with col2:
        faithfulness = stats.get("faithfulness_rate", 0) * 100
        st.metric(
            "Faithfulness",
            f"{faithfulness:.0f}%",
        )
    with col3:
        caught = stats.get("contradicted", 0) + stats.get("unsupported", 0)
        st.metric(
            "Hallucinations caught",
            caught,
        )
    
    # -- Verification trace
    with st.expander(f"🔍 Verification trace ({len(verifications)} claims)", expanded=True):
        for i, v in enumerate(verifications, 1):
            verdict = v.get("verdict", "Unsupported")
            emoji = VERDICT_COLORS.get(verdict, "⚠️")
            badge = VERDICT_BADGE.get(verdict, "")
            
            st.markdown(f"**{i}.** {emoji} {badge}  \n{v.get('claim', '')}")
            
            evidence = v.get("evidence", "")
            chunk_id = v.get("chunk_id", "")
            reasoning = v.get("reasoning", "")
            
            if evidence:
                st.caption(f"📎 Evidence: \"{evidence[:200]}{'...' if len(evidence) > 200 else ''}\" `[{chunk_id}]`")
            if reasoning:
                st.caption(f"💭 {reasoning}")
            
            st.markdown("")
    
    # -- Dropped claims
    dropped = corrected.get("dropped_claims", [])
    if dropped:
        with st.expander(f"🚫 Dropped claims ({len(dropped)})", expanded=False):
            for d in dropped:
                reason = d.get("reason", "Unsupported")
                emoji = VERDICT_COLORS.get(reason, "⚠️")
                st.markdown(f"{emoji} **[{reason}]** {d.get('claim', '')}")
                if d.get("reasoning"):
                    st.caption(d["reasoning"])
                st.markdown("")
    
    # -- Latency
    st.metric("⏱ Latency", f"{latency:.2f}s")


# --------------------------------------------------------------------------
# Results layout
# --------------------------------------------------------------------------
if run_button and question.strip():
    st.markdown(f"### Question")
    st.info(question)
    
    col_vanilla, col_fimeen = st.columns(2)
    
    with col_vanilla:
        render_vanilla_rag(question)
    
    with col_fimeen:
        render_fimeen(question)

elif run_button:
    st.warning("Please enter a question first.")