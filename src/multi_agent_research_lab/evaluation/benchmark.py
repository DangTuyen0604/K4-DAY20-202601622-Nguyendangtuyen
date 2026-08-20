"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run one system and compute deterministic, reproducible metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_total_cost(state),
        quality_score=_quality_score(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=float(bool(state.errors or not state.final_answer)),
        notes=(
            f"query={query}; words={len((state.final_answer or '').split())}; "
            f"errors={len(state.errors)}"
        ),
    )
    return state, metrics


def _total_cost(state: ResearchState) -> float | None:
    costs: list[float] = []
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, int | float):
            costs.append(float(cost))
    return sum(costs) if costs else None


def _citation_coverage(state: ResearchState) -> float | None:
    if not state.sources or not state.final_answer:
        return None
    labels = {str(source.metadata.get("citation_label", "")) for source in state.sources}
    cited = set(re.findall(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\]", state.final_answer))
    return len(labels & cited) / len(labels) if labels else None


def _quality_score(state: ResearchState) -> float:
    """Score answer completeness with a documented 0-10 heuristic rubric."""

    answer = state.final_answer or ""
    lower_answer = answer.lower()
    word_count = len(answer.split())
    score = min(word_count / 200, 1.0) * 2.0
    score += 2.0 if any(marker in answer for marker in ("#", "\n-", "\n1.")) else 0.5
    score += 1.5 if any(term in lower_answer for term in ("limit", "trade-off", "risk")) else 0.0
    score += (
        1.5
        if any(term in lower_answer for term in ("recommend", "should", "practical"))
        else 0.0
    )
    score += 1.0 if not state.errors else 0.0
    citation_coverage = _citation_coverage(state)
    if citation_coverage is None:
        score += 1.0
    else:
        score += 2.0 * citation_coverage
    return round(min(score, 10.0), 1)
