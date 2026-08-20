from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def test_benchmark_computes_cost_quality_citations_and_failure_rate() -> None:
    def runner(query: str) -> ResearchState:
        return ResearchState(
            request=ResearchQuery(query=query),
            sources=[
                SourceDocument(
                    title="Source",
                    snippet="Evidence",
                    metadata={"citation_label": "source"},
                )
            ],
            final_answer=(
                "# Answer\n"
                + "Grounded recommendation with limitations and practical trade-offs " * 30
                + "[source]."
            ),
            agent_results=[
                AgentResult(
                    agent=AgentName.WRITER,
                    content="answer",
                    metadata={"cost_usd": 0.001},
                )
            ],
        )

    _, metrics = run_benchmark("multi-agent-test", "Compare agents", runner)

    assert metrics.estimated_cost_usd == 0.001
    assert metrics.quality_score is not None and metrics.quality_score >= 8
    assert metrics.citation_coverage == 1.0
    assert metrics.failure_rate == 0.0
