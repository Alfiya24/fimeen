"""
Vanilla RAG: Retriever → Generator.

This is the baseline pipeline FIMEEN will be compared against.
It represents "standard" RAG: retrieve chunks, generate answer, return.
No verification, no claim decomposition, no self-correction.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.graph.nodes.retriever import retriever_node
from src.graph.nodes.generator import generator_node


class VanillaRAGState(TypedDict, total=False):
    """State passed between nodes in the vanilla RAG graph."""
    question: str
    retrieved_chunks: list[dict]
    answer: str


def build_vanilla_rag_graph():
    """
    Build a 2-node LangGraph:
    
        START → retriever → generator → END
    """
    graph = StateGraph(VanillaRAGState)
    
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)
    
    return graph.compile()


def run_vanilla_rag(question: str) -> dict:
    """
    Run the full vanilla RAG pipeline for one question.
    
    Returns the final state dict with question, retrieved_chunks, and answer.
    """
    app = build_vanilla_rag_graph()
    initial_state = {"question": question}
    result = app.invoke(initial_state)
    return result