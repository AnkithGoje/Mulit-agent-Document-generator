from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.graph.nodes import AgentNodes
from app.graph.routing import after_review, after_task
from app.graph.state import AgentState
from app.llm.providers import StructuredLLM


def build_graph(settings: Settings, llm: StructuredLLM):
    nodes = AgentNodes(settings, llm)
    graph = StateGraph(AgentState)
    graph.add_node("planner", nodes.plan)
    graph.add_node("execute_task", nodes.execute_task)
    graph.add_node("compose", nodes.compose)
    graph.add_node("reviewer", nodes.review)
    graph.add_node("revise", nodes.revise)
    graph.add_node("render", nodes.render)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "execute_task")
    graph.add_conditional_edges("execute_task", after_task)
    graph.add_edge("compose", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        lambda state: after_review(state, settings.max_revisions),
        {"revise": "revise", "render": "render"},
    )
    graph.add_edge("revise", "reviewer")
    graph.add_edge("render", END)
    return graph.compile()
