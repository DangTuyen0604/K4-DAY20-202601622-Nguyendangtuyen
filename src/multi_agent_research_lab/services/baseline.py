"""Reusable single-agent baseline runner."""

from time import perf_counter

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


def run_single_agent(
    request: ResearchQuery,
    llm_client: LLMClient | None = None,
) -> ResearchState:
    """Answer the complete research query in one model call."""

    client = llm_client or LLMClient()
    state = ResearchState(request=request)
    started = perf_counter()
    response = client.complete(
        system_prompt=(
            "You are a careful research assistant. Answer the user's question clearly and "
            "accurately for the requested audience. Distinguish established facts from "
            "inferences, do not invent citations, and state important limitations."
        ),
        user_prompt=(
            f"Research question: {request.query}\n"
            f"Audience: {request.audience}\n"
            "Produce a self-contained answer."
        ),
    )
    latency_seconds = perf_counter() - started
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.BASELINE,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_seconds": latency_seconds,
            },
        )
    )
    state.add_trace_event(
        "baseline",
        {
            "status": "completed",
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "duration_seconds": latency_seconds,
        },
    )
    return state
