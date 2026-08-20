"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ExternalTrace:
    """Mutable trace result populated after its context exits."""

    enabled: bool
    url: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Record a lightweight local span alongside external LangSmith traces."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


@contextmanager
def langsmith_trace(
    name: str,
    inputs: dict[str, Any],
    settings: Settings,
) -> Iterator[ExternalTrace]:
    """Create an optional LangSmith root trace without exported env vars.

    Local tracing remains available when no key is configured, and provider
    failures are logged without breaking the research workflow.
    """

    if not settings.langsmith_api_key:
        yield ExternalTrace(enabled=False)
        return

    from langsmith import Client, trace, tracing_context

    handle = ExternalTrace(enabled=True)
    yielded = False
    body_error: BaseException | None = None
    run: Any = None
    try:
        client = Client(api_key=settings.langsmith_api_key)
        with tracing_context(
            enabled=True,
            client=client,
            project_name=settings.langsmith_project,
            tags=["multi-agent-lab"],
        ), trace(
            name,
            run_type="chain",
            inputs=inputs,
            project_name=settings.langsmith_project,
            client=client,
            tags=["multi-agent-lab"],
        ) as run:
            try:
                yielded = True
                yield handle
            except BaseException as exc:
                body_error = exc
                raise
            finally:
                if handle.outputs:
                    run.end(outputs=handle.outputs)
        client.flush(timeout=10)
        handle.url = client.get_run_url(run=run, project_name=settings.langsmith_project)
    except BaseException as exc:
        if body_error is not None:
            raise
        logger.warning("LangSmith tracing unavailable: %s", exc)
        handle.enabled = False
        if not yielded:
            yield handle
