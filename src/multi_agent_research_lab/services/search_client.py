"""Offline corpus search client used by :class:`ResearcherAgent`."""

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, cast

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "with",
}


@dataclass(frozen=True)
class _CorpusEntry:
    source: SourceDocument
    topic_text: str
    document_text: str


class SearchClient:
    """Retrieve evidence from the repository's self-contained JSON corpus."""

    def __init__(self, corpus_dir: Path | None = None) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        self.corpus_dir = corpus_dir or repository_root / "ai_agent_offline_research_corpus_v2"

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return the most relevant, distinct offline source documents.

        Ranking is deterministic and lexical: matches in a document title receive
        the most weight, followed by topic metadata and then the embedded body.
        The returned snippet is the complete embedded source text, so callers never
        need to open the provenance URL.
        """

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if max_results < 1:
            raise ValueError("max_results must be at least 1")

        query_tokens = _tokenize(normalized_query)
        ranked = sorted(
            (
                (self._score(normalized_query, query_tokens, entry), entry)
                for entry in self.entries
            ),
            key=lambda item: (-item[0], item[1].source.title),
        )
        has_lexical_match = bool(ranked and ranked[0][0] > 0)

        results: list[SourceDocument] = []
        seen_document_ids: set[str] = set()
        synthetic_count = 0
        for score, entry in ranked:
            document_id = str(entry.source.metadata["document_id"])
            if document_id in seen_document_ids:
                continue
            is_synthetic = bool(entry.source.metadata["is_synthetic"])
            # Keep synthetic benchmark evidence visible but prevent it from
            # crowding out the public-reference sources in a small result set.
            if is_synthetic and synthetic_count >= 1:
                continue

            metadata = {
                **entry.source.metadata,
                "retrieval_score": score,
                "retrieval_fallback": not has_lexical_match,
            }
            results.append(entry.source.model_copy(update={"metadata": metadata}))
            seen_document_ids.add(document_id)
            synthetic_count += int(is_synthetic)
            if len(results) == max_results:
                break

        return results

    @cached_property
    def entries(self) -> tuple[_CorpusEntry, ...]:
        """Load and validate corpus documents once per client instance."""

        topics_dir = self.corpus_dir / "topics"
        if not topics_dir.is_dir():
            raise AgentExecutionError(f"Offline corpus directory was not found: {topics_dir}")

        entries: list[_CorpusEntry] = []
        for topic_path in sorted(topics_dir.glob("*.json")):
            try:
                raw_text = topic_path.read_text(encoding="utf-8")
                raw: object = json.loads(raw_text)
            except (OSError, json.JSONDecodeError) as exc:
                raise AgentExecutionError(f"Could not read corpus topic: {topic_path}") from exc

            payload = _as_dict(raw)
            metadata = _as_dict(payload.get("benchmark_metadata"))
            topic = _as_dict(payload.get("topic"))
            knowledge_base = _as_dict(payload.get("knowledge_base"))
            topic_id = _as_string(metadata.get("topic_id"), default=topic_path.stem)
            topic_name = _as_string(topic.get("name"))
            topic_tags = " ".join(_as_string_list(topic.get("tags")))
            research_question = _as_string(topic.get("research_question"))
            # The topic file also contains facts, failure modes, patterns, and
            # questions. Indexing it at topic weight helps map broad queries to
            # the right evidence packet before individual sources are ranked.
            topic_text = " ".join((topic_name, topic_tags, research_question, raw_text))

            documents = knowledge_base.get("source_documents")
            if not isinstance(documents, list):
                continue
            for raw_document in documents:
                document = _as_dict(raw_document)
                document_id = _as_string(document.get("document_id"))
                title = _as_string(document.get("title"))
                full_text = _as_string(document.get("full_text"))
                if not document_id or not title or not full_text:
                    continue

                takeaways = _as_string_list(document.get("key_takeaways"))
                citation_label = _as_string(
                    document.get("citation_label"), default=document_id
                )
                url = _as_string(document.get("provenance_url")) or None
                source = SourceDocument(
                    title=title,
                    url=url,
                    snippet=full_text,
                    metadata={
                        "document_id": document_id,
                        "citation_label": citation_label,
                        "document_class": _as_string(document.get("document_class")),
                        "is_synthetic": bool(document.get("is_synthetic", False)),
                        "topic_id": topic_id,
                        "topic_name": topic_name,
                        "corpus_file": topic_path.name,
                    },
                )
                document_text = " ".join((title, citation_label, " ".join(takeaways), full_text))
                entries.append(
                    _CorpusEntry(
                        source=source,
                        topic_text=topic_text,
                        document_text=document_text,
                    )
                )

        if not entries:
            raise AgentExecutionError(f"No source documents were found in: {topics_dir}")
        return tuple(entries)

    @staticmethod
    def _score(query: str, query_tokens: set[str], entry: _CorpusEntry) -> int:
        title_tokens = _tokenize(entry.source.title)
        topic_tokens = _tokenize(entry.topic_text)
        document_tokens = _tokenize(entry.document_text)
        score = (
            12 * len(query_tokens & title_tokens)
            + 6 * len(query_tokens & topic_tokens)
            + 2 * len(query_tokens & document_tokens)
        )
        normalized_query = " ".join(_TOKEN_PATTERN.findall(query.lower()))
        if normalized_query and normalized_query in entry.source.title.lower():
            score += 20
        if normalized_query and normalized_query in entry.topic_text.lower():
            score += 10
        return score


def _tokenize(text: str) -> set[str]:
    tokens = set(_TOKEN_PATTERN.findall(text.lower()))
    meaningful_tokens = tokens - _STOPWORDS
    return meaningful_tokens or tokens


def _as_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, Any], value)


def _as_string(value: object, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
