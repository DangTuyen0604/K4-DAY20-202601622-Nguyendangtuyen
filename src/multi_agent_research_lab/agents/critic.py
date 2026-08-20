"""Optional deterministic final-answer critic."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

_CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\]")


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Check citation validity and append structured findings."""

        if not state.final_answer:
            raise AgentExecutionError("Critic requires final_answer.")
        valid_labels = {
            str(source.metadata.get("citation_label", "")) for source in state.sources
        }
        cited_labels = set(_CITATION_PATTERN.findall(state.final_answer))
        unknown_labels = cited_labels - valid_labels
        uncited_labels = valid_labels - cited_labels
        findings = (
            f"Valid citations: {len(cited_labels & valid_labels)}; "
            f"unknown citations: {sorted(unknown_labels)}; "
            f"retrieved but uncited: {sorted(uncited_labels)}."
        )
        passed = not unknown_labels and bool(cited_labels)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=findings,
                metadata={
                    "unknown_labels": sorted(unknown_labels),
                    "uncited_labels": sorted(uncited_labels),
                    "passed": passed,
                },
            )
        )
        state.add_trace_event(
            "critic",
            {
                "status": "completed",
                "passed": passed,
                "unknown_labels": sorted(unknown_labels),
            },
        )
        return state
