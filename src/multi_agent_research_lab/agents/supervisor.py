"""Deterministic supervisor and routing policy."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect shared state and append the next route."""

        if state.final_answer:
            route = "done"
        elif state.iteration >= self.settings.max_iterations:
            route = "done"
            message = f"Supervisor reached max_iterations={self.settings.max_iterations}."
            state.errors.append(message)
            state.final_answer = (
                state.analysis_notes
                or state.research_notes
                or "The workflow stopped before producing a grounded answer."
            )
        elif not state.sources or not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        else:
            route = "writer"

        state.record_route(route)
        state.add_trace_event(
            "supervisor",
            {
                "next_route": route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        return state
