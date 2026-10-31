from typing import Literal

from app.graph.state import AgentState


def after_task(state: AgentState) -> Literal["execute_task", "compose"]:
    return "execute_task" if state["task_index"] < len(state["plan"].tasks) else "compose"


def after_review(state: AgentState, max_revisions: int) -> Literal["revise", "render"]:
    if not state["review"].approved and state["revisions"] < max_revisions:
        return "revise"
    return "render"
