from unittest.mock import MagicMock

import pytest

from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


def _ready_state() -> ResearchState:
    return ResearchState(
        request=ResearchQuery(query="Compare agent architectures"),
        sources=[
            SourceDocument(
                title="Public Reference",
                url="https://example.test/source",
                snippet="Evidence.",
                metadata={
                    "document_id": "source-1",
                    "citation_label": "source-1",
                    "is_synthetic": False,
                },
            )
        ],
        research_notes="Research notes [source-1].",
        analysis_notes="Analysis notes [source-1].",
    )


def test_writer_populates_validated_answer_result_and_trace() -> None:
    llm_client = MagicMock(spec=LLMClient)
    llm_client.complete.return_value = LLMResponse(
        content="Grounded final answer [source-1].\n\n## Sources\n[source-1] Public Reference",
        input_tokens=100,
        output_tokens=30,
    )
    state = _ready_state()

    result = WriterAgent(llm_client=llm_client).run(state)

    assert result is state
    assert state.final_answer is not None
    assert state.agent_results[0].agent == AgentName.WRITER
    assert state.agent_results[0].metadata["cited_labels"] == ["source-1"]
    assert state.trace[0]["payload"]["status"] == "completed"


def test_writer_rejects_unknown_citation() -> None:
    llm_client = MagicMock(spec=LLMClient)
    llm_client.complete.return_value = LLMResponse(content="Unsupported claim [invented].")
    state = _ready_state()

    with pytest.raises(AgentExecutionError, match="unknown citations"):
        WriterAgent(llm_client=llm_client).run(state)

    assert state.final_answer is None
    assert state.errors
