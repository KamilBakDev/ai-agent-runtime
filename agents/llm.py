"""LLM provider factory.

Providers are lazily imported so an unconfigured/uninstalled provider never breaks
startup for someone only using a different one. ``fake`` is a scripted, offline
chat model used by the test suite so unit tests never make network calls.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agents.config import Settings, get_settings


class ScriptedFakeChatModel(GenericFakeChatModel):
    """A fake chat model whose replies can be scripted per-call by node name.

    Falls back to echoing a short canned response derived from the last human
    message when no explicit script entry matches, so it never needs network access.
    """

    responses: dict[str, str] | None = None

    def _next_response(self, messages: list[Any]) -> str:
        if self.responses:
            # very small heuristic: match on any system message content substring
            for key, value in self.responses.items():
                for m in messages:
                    if key in getattr(m, "content", ""):
                        return value
        return "Mocked response."

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:  # type: ignore[override]
        messages = input if isinstance(input, list) else getattr(input, "messages", [input])
        return AIMessage(content=self._next_response(messages))

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:  # type: ignore[override]
        return self.invoke(input, config, **kwargs)


def get_chat_model(settings: Settings | None = None, **overrides: Any) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider.

    ``overrides`` are passed through to the underlying provider constructor
    (e.g. ``temperature=0``).
    """
    settings = settings or get_settings()
    provider = overrides.pop("provider", settings.llm_provider)

    if provider == "fake":
        return ScriptedFakeChatModel(messages=iter([]))

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model, base_url=settings.ollama_base_url, **overrides
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set")
        return ChatOpenAI(
            model=settings.llm_model or "gpt-4o-mini",
            api_key=settings.openai_api_key,
            **overrides,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini requires GOOGLE_API_KEY to be set")
        return ChatGoogleGenerativeAI(
            model=settings.llm_model or "gemini-1.5-flash",
            google_api_key=settings.google_api_key,
            **overrides,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
