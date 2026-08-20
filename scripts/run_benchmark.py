"""Run the configured baseline/multi-agent comparison and write its report."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.baseline import run_single_agent
from multi_agent_research_lab.services.storage import LocalArtifactStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sources", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path("configs/lab_default.yaml").read_text(encoding="utf-8"))
    queries: list[str] = config["benchmark"]["queries"]
    if args.limit is not None:
        queries = queries[: args.limit]

    settings = get_settings()
    all_metrics = []
    trace_url: str | None = None

    def baseline_runner(query: str) -> ResearchState:
        return run_single_agent(ResearchQuery(query=query))

    def multi_agent_runner(query: str) -> ResearchState:
        request = ResearchQuery(query=query, max_sources=args.max_sources)
        return MultiAgentWorkflow(settings=settings).run(ResearchState(request=request))

    for index, query in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] baseline: {query}")
        _, baseline_metrics = run_benchmark(f"baseline-{index}", query, baseline_runner)
        all_metrics.append(baseline_metrics)

        print(f"[{index}/{len(queries)}] multi-agent: {query}")
        multi_state, multi_metrics = run_benchmark(
            f"multi-agent-{index}", query, multi_agent_runner
        )
        all_metrics.append(multi_metrics)
        current_url = _trace_url(multi_state)
        if current_url:
            trace_url = current_url

    store = LocalArtifactStore()
    report = render_markdown_report(all_metrics, trace_url=trace_url)
    report_path = store.write_text("benchmark_report.md", report)
    raw_metrics = json.dumps(
        [metric.model_dump(mode="json") for metric in all_metrics],
        ensure_ascii=False,
        indent=2,
    )
    store.write_text("benchmark_metrics.json", raw_metrics + "\n")
    print(f"Report written to {report_path}")
    if trace_url:
        print(f"Trace: {trace_url}")


def _trace_url(state: ResearchState) -> str | None:
    for event in reversed(state.trace):
        if event["name"] != "external_trace":
            continue
        payload: dict[str, Any] = event["payload"]
        url = payload.get("url")
        return url if isinstance(url, str) else None
    return None


if __name__ == "__main__":
    main()
