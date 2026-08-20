from unittest.mock import MagicMock

import pytest

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


def _source(*, synthetic: bool = False) -> SourceDocument:
    label = "synthetic-study" if synthetic else "public-source"
    return SourceDocument(
        title="Synthetic Study" if synthetic else "Public Reference",
        snippet="Evidence about specialized agents and coordination overhead.",
        metadata={
            "document_id": label,
            "citation_label": label,
            "document_class": "synthetic_benchmark" if synthetic else "public_reference_summary",
            "is_synthetic": synthetic,
        },
    )


def test_analyst_populates_analysis_result_and_trace() -> None:
    llm_client = MagicMock(spec=LLMClient)
    llm_client.complete.return_value = LLMResponse(
        content="The public evidence is scoped [public-source]. The other result is synthetic.",
        input_tokens=200,
        output_tokens=40,
    )
    state = ResearchState(
        request=ResearchQuery(query="Compare agent architectures"),
        sources=[_source(), _source(synthetic=True)],
        research_notes="Specialization may help [public-source] [synthetic-study].",
    )

    result = AnalystAgent(llm_client=llm_client).run(state)

    assert result is state
    assert state.analysis_notes is not None
    assert state.agent_results[0].agent == AgentName.ANALYST
    assert state.agent_results[0].metadata["public_source_count"] == 1
    assert state.agent_results[0].metadata["synthetic_source_count"] == 1
    assert state.trace[0]["payload"]["status"] == "completed"


@pytest.mark.parametrize(
    ("sources", "research_notes", "expected_reason"),
    [
        ([_source()], None, "research_notes are required"),
        ([], "Some notes.", "sources are required"),
    ],
)
def test_analyst_requires_research_inputs(
    sources: list[SourceDocument],
    research_notes: str | None,
    expected_reason: str,
) -> None:
    state = ResearchState(
        request=ResearchQuery(query="Compare agent architectures"),
        sources=sources,
        research_notes=research_notes,
    )

    with pytest.raises(AgentExecutionError, match=expected_reason):
        AnalystAgent(llm_client=MagicMock(spec=LLMClient)).run(state)

    assert state.errors
    assert state.trace[0]["payload"]["status"] == "error"
