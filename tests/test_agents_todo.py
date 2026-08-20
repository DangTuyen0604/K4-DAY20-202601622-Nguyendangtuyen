from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _supervisor() -> SupervisorAgent:
    return SupervisorAgent(settings=Settings(MAX_ITERATIONS=6))


def test_supervisor_routes_through_required_stages() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    _supervisor().run(state)
    assert state.route_history[-1] == "researcher"

    state.sources.append(
        {
            "title": "Source",
            "snippet": "Evidence",
            "metadata": {"citation_label": "source"},
        }
    )
    state.research_notes = "Notes [source]."
    _supervisor().run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "Analysis [source]."
    _supervisor().run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "Answer [source]."
    _supervisor().run(state)
    assert state.route_history == ["researcher", "analyst", "writer", "done"]


def test_supervisor_stops_at_iteration_limit_with_partial_fallback() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        iteration=1,
        research_notes="Partial grounded notes.",
    )
    supervisor = SupervisorAgent(settings=Settings(MAX_ITERATIONS=1))

    supervisor.run(state)

    assert state.route_history == ["done"]
    assert state.final_answer == "Partial grounded notes."
    assert "max_iterations=1" in state.errors[0]
