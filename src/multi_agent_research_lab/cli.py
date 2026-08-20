"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.baseline import run_single_agent
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the complete research task with one LLM call."""

    _init()
    request = _parse_query(query)
    try:
        state = run_single_agent(request, LLMClient())
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=2) from exc

    console.print(Panel.fit(state.final_answer or "No answer", title="Single-Agent Baseline"))

    usage = state.agent_results[-1].metadata
    metrics = Table(title="Run Metrics")
    metrics.add_column("Latency (s)", justify="right")
    metrics.add_column("Input tokens", justify="right")
    metrics.add_column("Output tokens", justify="right")
    metrics.add_column("Cost (USD)", justify="right")
    metrics.add_row(
        f"{float(usage['latency_seconds']):.2f}",
        "n/a" if usage["input_tokens"] is None else str(usage["input_tokens"]),
        "n/a" if usage["output_tokens"] is None else str(usage["output_tokens"]),
        "n/a" if usage["cost_usd"] is None else f"{float(usage['cost_usd']):.6f}",
    )
    console.print(metrics)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[
        int,
        typer.Option("--max-sources", min=1, max=20, help="Maximum offline sources"),
    ] = 3,
) -> None:
    """Run the Supervisor/Researcher/Analyst/Writer workflow."""

    _init()
    request = _parse_query(query)
    request.max_sources = max_sources
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    console.print(Panel.fit(result.final_answer or "No final answer", title="Multi-Agent Answer"))
    console.print(f"[bold]Route history:[/bold] {' -> '.join(result.route_history)}")
    external_trace = next(
        (event for event in reversed(result.trace) if event["name"] == "external_trace"),
        None,
    )
    if external_trace and external_trace["payload"].get("url"):
        console.print(f"[bold]LangSmith trace:[/bold] {external_trace['payload']['url']}")
    if result.errors:
        console.print(
            Panel.fit("\n".join(result.errors), title="Workflow Warnings", style="yellow")
        )


if __name__ == "__main__":
    app()
