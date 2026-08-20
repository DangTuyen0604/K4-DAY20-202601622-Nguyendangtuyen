from unittest.mock import MagicMock

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.llm_client import LLMClient


def test_complete_returns_content_and_token_usage() -> None:
    response = MagicMock()
    response.output_text = "A grounded answer."
    response.usage.input_tokens = 12
    response.usage.output_tokens = 7

    openai_client = MagicMock()
    openai_client.responses.create.return_value = response
    settings = Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4o-mini",
        OPENAI_BASE_URL=None,
        TIMEOUT_SECONDS=30,
    )

    result = LLMClient(settings=settings, client=openai_client).complete(
        system_prompt="Be accurate.",
        user_prompt="Explain agents.",
    )

    assert result.content == "A grounded answer."
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    openai_client.responses.create.assert_called_once_with(
        model="gpt-4o-mini",
        instructions="Be accurate.",
        input="Explain agents.",
        store=False,
    )


def test_complete_supports_openai_compatible_provider() -> None:
    completion = MagicMock()
    completion.choices[0].message.content = "Provider answer."
    completion.usage.prompt_tokens = 9
    completion.usage.completion_tokens = 4
    completion.usage.model_extra = {"cost": 0.00012}

    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = completion
    settings = Settings(
        OPENAI_API_KEY="test-provider-key",
        OPENAI_MODEL="openai/gpt-4o-mini",
        OPENAI_BASE_URL="https://openrouter.ai/api/v1",
    )

    result = LLMClient(settings=settings, client=openai_client).complete(
        system_prompt="Be accurate.",
        user_prompt="Explain agents.",
    )

    assert result.content == "Provider answer."
    assert result.input_tokens == 9
    assert result.output_tokens == 4
    assert result.cost_usd == 0.00012
    openai_client.chat.completions.create.assert_called_once_with(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Be accurate."},
            {"role": "user", "content": "Explain agents."},
        ],
    )
