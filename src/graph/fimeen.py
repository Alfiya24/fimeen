"""
FIMEEN: full self-verifying RAG pipeline.

5-node LangGraph:
    START → retriever → generator → decomposer → verifier → corrector → END

Each node reads from and writes to a shared state dict.
This is the pipeline compared against vanilla RAG in evaluation.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.graph.nodes.retriever import retriever_node
from src.graph.nodes.generator import generator_node
from src.graph.nodes.decomposer import decomposer_node
from src.graph.nodes.verifier import verifier_node
from src.graph.nodes.corrector import corrector_node


class FimeenState(TypedDict, total=False):
    """Unified state dict passed between FIMEEN nodes."""
    question: str
    retrieved_chunks: list[dict]
    answer: str
    claims: list[str]
    verifications: list[dict]
    corrected_output: dict


def build_fimeen_graph():
    """
    Build the full 5-node LangGraph pipeline.
    
    Flow: retriever → generator → decomposer → verifier → corrector
    """
    graph = StateGraph(FimeenState)
    
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("decomposer", decomposer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("corrector", corrector_node)
    
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "decomposer")
    graph.add_edge("decomposer", "verifier")
    graph.add_edge("verifier", "corrector")
    graph.add_edge("corrector", END)
    
    return graph.compile()


def run_fimeen(question: str) -> dict:
    """
    Run the full FIMEEN pipeline for one question.
    
    Returns the final state dict with all intermediate artifacts:
    - question
    - retrieved_chunks
    - answer (vanilla generator output)
    - claims (decomposed atomic claims)
    - verifications (per-claim verdicts)
    - corrected_output (final FIMEEN result with kept/dropped claims + stats)
    """
    app = build_fimeen_graph()
    initial_state = {"question": question}
    result = app.invoke(initial_state)
    return result