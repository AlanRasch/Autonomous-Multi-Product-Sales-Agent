"""Wires the five stages into a LangGraph StateGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..llm import SalesLLM
from .nodes import analysis, enrichment, make_execution, make_ingestion, strategy
from .state import AgentState

STAGES = ("ingestion", "enrichment", "analysis", "strategy", "execution")


def build_graph(llm: SalesLLM):
    graph = StateGraph(AgentState)
    graph.add_node("ingestion", make_ingestion(llm))
    graph.add_node("enrichment", enrichment)
    graph.add_node("analysis", analysis)
    graph.add_node("strategy", strategy)
    graph.add_node("execution", make_execution(llm))

    graph.add_edge(START, "ingestion")
    for current, nxt in zip(STAGES, STAGES[1:]):
        graph.add_edge(current, nxt)
    graph.add_edge("execution", END)  # -> WAIT / LISTEN for the next inbound event
    return graph.compile()
