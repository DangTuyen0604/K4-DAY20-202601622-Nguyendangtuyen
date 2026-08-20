"""LangGraph orchestration for the research agents."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Literal, Protocol, cast

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import langsmith_trace

Route = Literal["researcher", "analyst", "writer", "done"]


class CompiledGraph(Protocol):
    def invoke(
        self,
        input: ResearchState,
        config: dict[str, Any] | None = None,
    ) -> ResearchState | dict[str, Any]: ...


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self._last_state: ResearchState | None = None

    def build(self) -> CompiledGraph:
        """Compile nodes and conditional routes into a LangGraph graph."""

        builder = StateGraph(ResearchState)
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("analyst", self._analyst_node)
        builder.add_node("writer", self._writer_node)
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")
        return cast(CompiledGraph, builder.compile())

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph with timeout, tracing, and partial-result fallback."""

        self._last_state = state
        graph = self.build()

        def invoke() -> tuple[ResearchState | dict[str, Any], str | None, bool]:
            with langsmith_trace(
                "multi-agent-research-workflow",
                {"query": state.request.query, "max_sources": state.request.max_sources},
                self.settings,
            ) as external_trace:
                raw_result = graph.invoke(
                    state,
                    config={
                        "recursion_limit": self.settings.max_iterations * 2 + 4,
                        "run_name": "multi-agent-research-graph",
                        "tags": ["multi-agent-lab"],
                    },
                )
                result_state = self._to_state(raw_result)
                external_trace.outputs = {
                    "route_history": result_state.route_history,
                    "has_final_answer": bool(result_state.final_answer),
                    "error_count": len(result_state.errors),
                }
            return raw_result, external_trace.url, external_trace.enabled

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-workflow")
        future = executor.submit(invoke)
        try:
            raw_result, trace_url, trace_enabled = future.result(
                timeout=self.settings.timeout_seconds
            )
        except FutureTimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return self._fallback(
                self._last_state or state,
                f"Workflow exceeded timeout_seconds={self.settings.timeout_seconds}.",
            )
        except (AgentExecutionError, GraphRecursionError) as exc:
            executor.shutdown(wait=True)
            return self._fallback(self._last_state or state, f"Workflow failed: {exc}")
        else:
            executor.shutdown(wait=True)

        result = self._to_state(raw_result)
        result.add_trace_event(
            "external_trace",
            {
                "provider": "langsmith",
                "enabled": trace_enabled,
                "url": trace_url,
                "project": self.settings.langsmith_project if trace_enabled else None,
            },
        )
        return result

    def _supervisor_node(self, state: ResearchState) -> ResearchState:
        return self._remember(self.supervisor.run(state))

    def _researcher_node(self, state: ResearchState) -> ResearchState:
        try:
            return self._remember(self.researcher.run(state))
        finally:
            self._last_state = state

    def _analyst_node(self, state: ResearchState) -> ResearchState:
        try:
            return self._remember(self.analyst.run(state))
        finally:
            self._last_state = state

    def _writer_node(self, state: ResearchState) -> ResearchState:
        try:
            return self._remember(self.writer.run(state))
        finally:
            self._last_state = state

    @staticmethod
    def _route_after_supervisor(state: ResearchState) -> Route:
        if not state.route_history:
            raise AgentExecutionError("Supervisor did not record a route.")
        route = state.route_history[-1]
        if route not in {"researcher", "analyst", "writer", "done"}:
            raise AgentExecutionError(f"Supervisor produced an invalid route: {route}")
        return cast(Route, route)

    def _remember(self, state: ResearchState) -> ResearchState:
        self._last_state = state
        return state

    @staticmethod
    def _to_state(result: ResearchState | dict[str, Any]) -> ResearchState:
        return result if isinstance(result, ResearchState) else ResearchState.model_validate(result)

    @staticmethod
    def _fallback(state: ResearchState, message: str) -> ResearchState:
        if message not in state.errors:
            state.errors.append(message)
        if not state.final_answer:
            state.final_answer = (
                state.analysis_notes
                or state.research_notes
                or "The workflow could not produce a grounded answer. See errors for details."
            )
        if not state.route_history or state.route_history[-1] != "done":
            state.record_route("done")
        state.add_trace_event("workflow_fallback", {"status": "fallback", "reason": message})
        return state
