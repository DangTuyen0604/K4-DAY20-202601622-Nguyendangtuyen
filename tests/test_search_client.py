from pathlib import Path

import pytest

from multi_agent_research_lab.services.search_client import SearchClient

CORPUS_DIR = Path(__file__).parents[1] / "ai_agent_offline_research_corpus_v2"


def test_search_returns_ranked_sources_with_citation_metadata() -> None:
    results = SearchClient(CORPUS_DIR).search(
        "role specialization in multi-agent systems",
        max_results=3,
    )

    assert len(results) == 3
    assert any(source.metadata["topic_id"] == "AIAGENT-02" for source in results)
    assert all(source.title and source.snippet for source in results)
    assert all(source.metadata["document_id"] for source in results)
    assert all(source.metadata["citation_label"] for source in results)
    assert all(source.metadata["retrieval_score"] > 0 for source in results)
    assert all(source.metadata["retrieval_fallback"] is False for source in results)


def test_search_uses_deterministic_fallback_for_unknown_query() -> None:
    first_run = SearchClient(CORPUS_DIR).search("quokka zephyr xylophone", max_results=2)
    second_run = SearchClient(CORPUS_DIR).search("quokka zephyr xylophone", max_results=2)

    assert len(first_run) == 2
    assert [source.metadata["document_id"] for source in first_run] == [
        source.metadata["document_id"] for source in second_run
    ]
    assert all(source.metadata["retrieval_fallback"] is True for source in first_run)


@pytest.mark.parametrize(("query", "max_results"), [("", 5), ("valid query", 0)])
def test_search_rejects_invalid_arguments(query: str, max_results: int) -> None:
    with pytest.raises(ValueError):
        SearchClient(CORPUS_DIR).search(query, max_results=max_results)
