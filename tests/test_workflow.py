from unittest.mock import MagicMock

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_runs_all_required_agents_in_order() -> None:
    researcher = MagicMock(spec=ResearcherAgent)
    analyst = MagicMock(spec=AnalystAgent)
    writer = MagicMock(spec=WriterAgent)

    def research(state: ResearchState) -> ResearchState:
        state.sources = [
            SourceDocument(
                title="Source",
                snippet="Evidence",
                metadata={"citation_label": "source", "is_synthetic": False},
            )
        ]
        state.research_notes = "Research [source]."
        return state

    def analyze(state: ResearchState) -> ResearchState:
        state.analysis_notes = "Analysis [source]."
        return state

    def write(state: ResearchState) -> ResearchState:
        state.final_answer = "Final answer [source]."
        return state

    researcher.run.side_effect = research
    analyst.run.side_effect = analyze
    writer.run.side_effect = write
    settings = Settings(
        MAX_ITERATIONS=6,
        TIMEOUT_SECONDS=30,
        LANGSMITH_API_KEY=None,
    )
    state = ResearchState(request=ResearchQuery(query="Compare agent architectures"))

    result = MultiAgentWorkflow(
        settings=settings,
        researcher=researcher,
        analyst=analyst,
        writer=writer,
    ).run(state)

    assert result.final_answer == "Final answer [source]."
    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    researcher.run.assert_called_once()
    analyst.run.assert_called_once()
    writer.run.assert_called_once()
    assert result.trace[-1]["name"] == "external_trace"
