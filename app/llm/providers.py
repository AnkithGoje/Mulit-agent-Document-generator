import asyncio
import json
import logging
import time
from typing import Any, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from app.config import Settings


logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)

MAX_RATE_LIMIT_WAIT = 90.0


class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class StructuredLLM(Protocol):
    async def generate(
        self,
        schema: type[ModelT],
        system_prompt: str,
        user_payload: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> ModelT: ...


TOKEN_BUDGET = {
    "TaskResult": 2000,
    "DocumentPlan": 2000,
    "ReviewResult": 2000,
    "DocumentContent": 5000,
}


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: dict[int, BaseChatModel] = {}
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    def _get_model(self, max_tokens: int) -> BaseChatModel:
        if max_tokens in self._models:
            return self._models[max_tokens]

        if self.settings.llm_provider == "poolside":
            if not self.settings.poolside_api_key.get_secret_value():
                raise LLMConfigurationError(
                    "Set POOLSIDE_API_KEY in .env to use LLM_PROVIDER=poolside."
                )
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                api_key=self.settings.poolside_api_key.get_secret_value(),
                base_url=self.settings.poolside_base_url,
                model=self.settings.poolside_model,
                temperature=self.settings.llm_temperature,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,
                use_responses_api=False,
                # ChatOpenAI otherwise rewrites max_tokens to max_completion_tokens.
                extra_body={
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        elif self.settings.llm_provider == "groq":
            if not self.settings.groq_api_key.get_secret_value():
                raise LLMConfigurationError(
                    "Set GROQ_API_KEY in .env, or select LLM_PROVIDER=ollama "
                    "and start a local Ollama model."
                )
            from langchain_groq import ChatGroq

            kwargs: dict[str, Any] = dict(
                api_key=self.settings.groq_api_key.get_secret_value(),
                model=self.settings.groq_model,
                temperature=self.settings.llm_temperature,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,
                max_tokens=max_tokens,
            )
            if self.settings.groq_reasoning_effort:
                kwargs["reasoning_effort"] = self.settings.groq_reasoning_effort
            model = ChatGroq(**kwargs)
        else:
            from langchain_ollama import ChatOllama

            model = ChatOllama(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=self.settings.llm_temperature,
                num_predict=max_tokens,
                client_kwargs={"timeout": self.settings.llm_timeout_seconds},
            )
        self._models[max_tokens] = model
        return model

    async def _throttle(self) -> None:
        gap = self.settings.llm_min_interval_seconds
        if gap <= 0:
            return
        async with self._lock:
            wait = gap - (time.monotonic() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    @staticmethod
    def _status_code(exc: Exception | None) -> int | None:
        for _ in range(4):
            if exc is None:
                break
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                return int(status)
            exc = exc.__cause__
        return None

    @staticmethod
    def _retry_after(exc: Exception | None) -> float:
        for _ in range(4):
            if exc is None:
                break
            response = getattr(exc, "response", None)
            try:
                delay = float(response.headers.get("retry-after", "0"))
            except (AttributeError, TypeError, ValueError):
                delay = 0.0
            if delay > 0:
                return delay
            exc = exc.__cause__
        return 0.0

    async def generate(
        self,
        schema: type[ModelT],
        system_prompt: str,
        user_payload: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> ModelT:
        parser = None
        if self.settings.llm_provider == "poolside":
            parser = PydanticOutputParser(pydantic_object=schema)
            system_prompt = f"{system_prompt}\n\n{parser.get_format_instructions()}"
        base_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_payload, ensure_ascii=True)),
        ]
        messages = base_messages
        max_tokens = TOKEN_BUDGET.get(schema.__name__, self.settings.llm_max_tokens)
        model = self._get_model(max_tokens)
        structured = (
            model | parser if parser is not None
            else model.with_structured_output(schema, method="json_schema")
        )
        last_exc: Exception | None = None

        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                await self._throttle()
                async with asyncio.timeout(self.settings.llm_timeout_seconds):
                    result = await structured.ainvoke(messages, config=config)
                return result if isinstance(result, schema) else schema.model_validate(result)
            except Exception as exc:
                last_exc = exc
                status = self._status_code(exc)
                error_text = str(exc)
                for secret in (
                    self.settings.poolside_api_key,
                    self.settings.groq_api_key, self.settings.langsmith_api_key,
                ):
                    value = secret.get_secret_value()
                    if value:
                        error_text = error_text.replace(value, "[redacted]")
                logger.warning(
                    "LLM call failed schema=%s attempt=%d status=%s error_type=%s detail=%s",
                    schema.__name__, attempt + 1, status, type(exc).__name__, error_text[:300],
                )
                if attempt == self.settings.llm_max_retries:
                    break

                delay = float(2 ** (attempt + 1))
                if status == 429:
                    delay = max(delay, self._retry_after(exc))
                    if delay > MAX_RATE_LIMIT_WAIT:
                        raise LLMRateLimitError(
                            "The provider's rate limit needs a long wait "
                            f"(about {int(delay)} seconds). Try again later or switch model.",
                            retry_after=delay,
                        ) from exc
                elif not isinstance(exc, TimeoutError):
                    messages = base_messages + [HumanMessage(
                        content="Your previous reply was rejected: "
                                f"{error_text[:600]}. Return a corrected, complete result "
                                "that satisfies the schema."
                    )]
                logger.info("Retrying schema=%s delay_seconds=%.2f", schema.__name__, delay)
                await asyncio.sleep(delay)

        if self._status_code(last_exc) == 429:
            raise LLMRateLimitError(
                "The model provider is rate limiting requests. Wait a minute and retry, "
                "or reduce MAX_REVISIONS.",
                retry_after=self._retry_after(last_exc) or None,
            ) from last_exc
        raise LLMError(
            "The language model could not produce a valid result. "
            "Check the configured model, provider availability, and rate limits."
        ) from last_exc
