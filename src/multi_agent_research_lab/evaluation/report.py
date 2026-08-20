"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    trace_url: str | None = None,
) -> str:
    """Render metrics, method, comparison, trace, and failure analysis."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )
    baseline = [item for item in metrics if item.run_name.startswith("baseline")]
    multi_agent = [item for item in metrics if item.run_name.startswith("multi-agent")]
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "Both systems receive the same queries. Latency is wall-clock time and cost comes ",
            "from provider usage metadata. Quality uses a deterministic 0-10 rubric covering ",
            "completeness, structure, limitations, recommendations, errors, and citations. ",
            "Citation coverage is cited corpus labels divided by retrieved labels.",
            "",
            "## Aggregate comparison",
            "",
            _comparison_line("Baseline", baseline),
            _comparison_line("Multi-agent", multi_agent),
            "",
            "## Interpretation",
            "",
            _interpretation(baseline, multi_agent),
            "The quality score is an automated reproducible heuristic, not a substitute for ",
            "the peer-review rubric or a human factuality review.",
            "",
            "## Trace",
            "",
            trace_url or "No external trace URL was recorded.",
            "",
            "## Failure mode analysis",
            "",
            "The main observed risk is coordination overhead: multi-agent execution makes three ",
            "LLM calls and passes growing context between roles, increasing latency and cost. ",
            "Citation drift is another risk; Writer validates every bracketed citation against ",
            "the retrieved source catalog. Transient provider failures are retried centrally, ",
            "offline retrieval avoids search outages, and workflow timeout/max-iteration guards ",
            "return the best available partial result instead of looping indefinitely.",
        ]
    )
    return "\n".join(lines) + "\n"


def _comparison_line(label: str, items: list[BenchmarkMetrics]) -> str:
    if not items:
        return f"- {label}: no runs"
    latency = sum(item.latency_seconds for item in items) / len(items)
    costs = [item.estimated_cost_usd for item in items if item.estimated_cost_usd is not None]
    quality = [item.quality_score for item in items if item.quality_score is not None]
    cost_text = "n/a" if not costs else f"${sum(costs) / len(costs):.6f}"
    quality_text = "n/a" if not quality else f"{sum(quality) / len(quality):.1f}/10"
    return (
        f"- {label}: average latency {latency:.2f}s, average cost {cost_text}, "
        f"average quality {quality_text}."
    )


def _interpretation(
    baseline: list[BenchmarkMetrics], multi_agent: list[BenchmarkMetrics]
) -> str:
    if not baseline or not multi_agent:
        return "Insufficient runs for a relative comparison."
    baseline_latency = sum(item.latency_seconds for item in baseline) / len(baseline)
    multi_latency = sum(item.latency_seconds for item in multi_agent) / len(multi_agent)
    baseline_costs = [
        item.estimated_cost_usd for item in baseline if item.estimated_cost_usd is not None
    ]
    multi_costs = [
        item.estimated_cost_usd for item in multi_agent if item.estimated_cost_usd is not None
    ]
    latency_ratio = multi_latency / baseline_latency
    if not baseline_costs or not multi_costs:
        return f"Multi-agent latency was {latency_ratio:.2f}x baseline; cost ratio unavailable."
    baseline_cost = sum(baseline_costs) / len(baseline_costs)
    multi_cost = sum(multi_costs) / len(multi_costs)
    return (
        f"Multi-agent latency was {latency_ratio:.2f}x baseline and cost was "
        f"{multi_cost / baseline_cost:.2f}x baseline. The quality/citation gain therefore "
        "comes with measurable coordination overhead."
    )
