"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Small OpenAI Responses API adapter used by all agents."""

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        if client is not None:
            self._client = client
            return

        if not self.settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not configured. Add it to .env before calling the LLM."
            )
        self._client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=float(self.settings.timeout_seconds),
            # Retry is centralized below so agents do not each implement it.
            max_retries=0,
        )

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _create_response(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        # OpenAI-compatible providers commonly implement Chat Completions but not
        # the newer Responses endpoint. Native OpenAI uses Responses by default.
        if self.settings.openai_base_url:
            completion = self._client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = (completion.choices[0].message.content or "").strip()
            usage = completion.usage
            return self._build_result(
                content=content,
                input_tokens=None if usage is None else usage.prompt_tokens,
                output_tokens=None if usage is None else usage.completion_tokens,
                cost_usd=_usage_cost(usage),
            )

        response = self._client.responses.create(
            model=self.settings.openai_model,
            instructions=system_prompt,
            input=user_prompt,
            store=False,
        )
        return self._build_result(
            content=response.output_text.strip(),
            input_tokens=None if response.usage is None else response.usage.input_tokens,
            output_tokens=None if response.usage is None else response.usage.output_tokens,
            cost_usd=_usage_cost(response.usage),
        )

    def _build_result(
        self,
        content: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
    ) -> LLMResponse:
        if not content:
            raise AgentExecutionError("The LLM returned an empty response.")

        result = LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        logger.info(
            "LLM completion model=%s input_tokens=%s output_tokens=%s",
            self.settings.openai_model,
            result.input_tokens,
            result.output_tokens,
        )
        return result

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with token usage metadata."""

        try:
            return self._create_response(system_prompt, user_prompt)
        except AgentExecutionError:
            raise
        except OpenAIError as exc:
            logger.exception("OpenAI request failed after retries")
            raise AgentExecutionError(f"OpenAI request failed: {exc}") from exc


def _usage_cost(usage: object | None) -> float | None:
    if usage is None:
        return None
    model_extra = getattr(usage, "model_extra", None)
    if not isinstance(model_extra, dict):
        return None
    cost = model_extra.get("cost")
    return float(cost) if isinstance(cost, int | float) else None
