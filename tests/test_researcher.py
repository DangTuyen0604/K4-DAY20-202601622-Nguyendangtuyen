from unittest.mock import MagicMock

import pytest

from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


def _source() -> SourceDocument:
    return SourceDocument(
        title="Grounded Agent Study",
        url="https://example.test/study",
        snippet="Specialized agents can improve decomposable research tasks.",
        metadata={
            "document_id": "study-1",
            "citation_label": "study-1",
            "document_class": "public_reference_summary",
            "is_synthetic": False,
        },
    )


def test_researcher_populates_sources_notes_result_and_trace() -> None:
    search_client = MagicMock(spec=SearchClient)
    search_client.search.return_value = [_source()]
    llm_client = MagicMock(spec=LLMClient)
    llm_client.complete.return_value = LLMResponse(
        content="Specialization helps decomposable work [study-1].",
        input_tokens=120,
        output_tokens=25,
    )
    state = ResearchState(
        request=ResearchQuery(query="Compare agent specialization", max_sources=3)
    )

    result = ResearcherAgent(search_client=search_client, llm_client=llm_client).run(state)

    assert result is state
    assert state.sources == [_source()]
    assert state.research_notes == "Specialization helps decomposable work [study-1]."
    assert state.agent_results[0].agent == AgentName.RESEARCHER
    assert state.agent_results[0].metadata["citation_labels"] == ["study-1"]
    assert state.trace[0]["name"] == "researcher"
    assert state.trace[0]["payload"]["status"] == "completed"
    search_client.search.assert_called_once_with("Compare agent specialization", max_results=3)


def test_researcher_records_failure_when_synthesis_fails() -> None:
    search_client = MagicMock(spec=SearchClient)
    search_client.search.return_value = [_source()]
    llm_client = MagicMock(spec=LLMClient)
    llm_client.complete.side_effect = AgentExecutionError("provider unavailable")
    state = ResearchState(request=ResearchQuery(query="Compare agent specialization"))

    with pytest.raises(AgentExecutionError, match="Researcher failed"):
        ResearcherAgent(search_client=search_client, llm_client=llm_client).run(state)

    assert state.sources == [_source()]
    assert state.research_notes is None
    assert state.errors == ["Researcher failed: provider unavailable"]
    assert state.trace[0]["payload"]["status"] == "error"
