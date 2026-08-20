"""Writer agent implementation."""

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)
_CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\]")


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize a final answer and validate its citation labels."""

        if not state.research_notes:
            return self._fail(state, "research_notes are required before writing")
        if not state.analysis_notes:
            return self._fail(state, "analysis_notes are required before writing")
        if not state.sources:
            return self._fail(state, "sources are required before writing")

        valid_labels = {
            str(source.metadata["citation_label"]) for source in state.sources
        }
        span: dict[str, object]
        try:
            with trace_span(
                "writer",
                {"query": state.request.query, "source_count": len(state.sources)},
            ) as raw_span:
                span = raw_span
                response = self.llm_client.complete(
                    system_prompt=(
                        "You are the Writer in a multi-agent research workflow. Produce a "
                        "clear, self-contained answer grounded only in the supplied research "
                        "and analysis. Cite substantive claims using only the exact labels in "
                        "the source catalog, formatted as [label]. Never invent citations. "
                        "Explicitly identify synthetic evidence and avoid presenting it as a "
                        "real-world study. Preserve uncertainty and important limitations."
                    ),
                    user_prompt=self._build_prompt(state),
                )
                cited_labels = set(_CITATION_PATTERN.findall(response.content))
                unknown_labels = cited_labels - valid_labels
                if unknown_labels:
                    unknown = ", ".join(sorted(unknown_labels))
                    raise AgentExecutionError(f"Writer produced unknown citations: {unknown}")
                if not cited_labels:
                    raise AgentExecutionError("Writer produced no valid citations.")

                state.final_answer = response.content
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.WRITER,
                        content=response.content,
                        metadata={
                            "cited_labels": sorted(cited_labels),
                            "citation_count": len(cited_labels),
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
        except AgentExecutionError as exc:
            return self._fail(state, str(exc), cause=exc)

        state.add_trace_event(
            "writer",
            {
                "status": "completed",
                "citation_count": len(cited_labels),
                "cited_labels": sorted(cited_labels),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "duration_seconds": span["duration_seconds"],
            },
        )
        return state

    @staticmethod
    def _build_prompt(state: ResearchState) -> str:
        source_lines = []
        for source in state.sources:
            label = str(source.metadata["citation_label"])
            synthetic = bool(source.metadata.get("is_synthetic", False))
            source_lines.append(
                f"[{label}] {source.title} | synthetic={synthetic} | "
                f"url={source.url or 'offline-only'}"
            )

        return (
            f"Question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"RESEARCH NOTES:\n{state.research_notes}\n\n"
            f"ANALYSIS NOTES:\n{state.analysis_notes}\n\n"
            f"SOURCE CATALOG:\n{chr(10).join(source_lines)}\n\n"
            "Write the final answer with:\n"
            "1. A direct answer or executive summary\n"
            "2. A structured evidence-based discussion\n"
            "3. Trade-offs, limitations, and practical recommendations\n"
            "4. A Sources section listing only sources actually cited"
        )

    @staticmethod
    def _fail(
        state: ResearchState,
        reason: str,
        cause: Exception | None = None,
    ) -> ResearchState:
        message = f"Writer failed: {reason}"
        state.errors.append(message)
        state.add_trace_event(
            "writer",
            {"status": "error", "error": reason, "source_count": len(state.sources)},
        )
        logger.error(message)
        error = AgentExecutionError(message)
        if cause is not None:
            raise error from cause
        raise error
