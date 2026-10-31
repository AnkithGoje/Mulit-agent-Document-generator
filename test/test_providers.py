import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from groq import BadRequestError, RateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import ValidationError

import app.llm.providers as providers
from app.config import Settings
from app.llm.providers import LLMClient, LLMConfigurationError, LLMError, LLMRateLimitError, TOKEN_BUDGET
from app.schemas import DocumentContent, DocumentPlan, DocumentSection, PlannedTask, ReviewResult, TaskResult


RESULT = {"summary": "Analyzed the illustrative budget.", "findings": ["Mock costs total USD 30000."]}


class CustomTaskResult(TaskResult):
    pass


OUTPUT_CASES = [
    (TaskResult.model_validate(RESULT), 2000),
    (
        DocumentPlan(
            title="Mock plan", document_type="Brief", audience="Sponsors",
            tasks=[PlannedTask(title="Analyze scope", instructions="Use mock data.")],
        ),
        2000,
    ),
    (ReviewResult(approved=True, score=9), 2000),
    (
        DocumentContent(
            title="Mock brief", summary="An illustrative recommendation.",
            sections=[DocumentSection(heading="Scope", paragraphs=["Use a small mock pilot."])],
        ),
        5000,
    ),
]


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for name in [
        *(field.upper() for field in Settings.model_fields),
        "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID", "OPENAI_PROXY",
        "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})

    def reject_network(*args, **kwargs):
        raise AssertionError("Offline tests must not make live HTTP requests.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", reject_network)
    with tracing_context(enabled=False):
        yield


@pytest.fixture
def poolside_http(monkeypatch):
    import langchain_openai

    requests = []
    replies = []

    def respond(request):
        requests.append(request)
        assert replies, "An unexpected SDK request exhausted the offline responses."
        reply = replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return httpx.Response(200, json={
            "id": "chatcmpl-poolside-test",
            "object": "chat.completion",
            "created": 0,
            "model": "poolside/laguna-s-2.1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    transport = httpx.MockTransport(respond)
    sync_client = httpx.Client(transport=transport, trust_env=False)
    async_client = httpx.AsyncClient(transport=transport, trust_env=False)

    def build_model(**kwargs):
        return ChatOpenAI(**kwargs, http_client=sync_client, http_async_client=async_client)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", build_model)
    client = LLMClient(Settings(
        _env_file=None,
        poolside_api_key="poolside-test-key",
        groq_api_key="groq-test-key",
        langsmith_api_key="langsmith-test-key",
        llm_min_interval_seconds=0,
        llm_max_retries=1,
        llm_max_tokens=3072,
    ))
    try:
        yield SimpleNamespace(client=client, requests=requests, replies=replies)
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())
        assert sync_client.is_closed and async_client.is_closed


class StubModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.schemas = []
        self.token_budgets = []

    def with_structured_output(self, schema, method):
        self.schemas.append((schema, method))
        return self

    async def ainvoke(self, messages, config=None):
        self.calls.append((messages, config))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return await response()
        return response


@pytest.fixture
def backoffs(monkeypatch):
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("app.llm.providers.asyncio.sleep", record_sleep)
    return delays


def stub_client(monkeypatch, responses, retries=1, timeout=90):
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key="test-only-key",
        langsmith_tracing=False,
        llm_max_retries=retries,
        llm_min_interval_seconds=0,
    ).model_copy(update={"llm_timeout_seconds": timeout})
    client = LLMClient(settings)
    model = StubModel(responses)
    def get_model(max_tokens):
        model.token_budgets.append(max_tokens)
        return model

    monkeypatch.setattr(client, "_get_model", get_model)
    return client, model


@pytest.mark.parametrize("response", [RESULT, TaskResult.model_validate(RESULT)])
def test_generate_validates_results_and_forwards_messages_and_config(monkeypatch, response):
    client, model = stub_client(monkeypatch, [response])
    payload = {"request": "Analyze an illustrative business budget."}
    config = {"tags": ["provider-test"]}
    result = asyncio.run(client.generate(TaskResult, "Use mock data.", payload, config))

    assert result.model_dump() == RESULT
    assert model.schemas == [(TaskResult, "json_schema")]
    messages, received_config = model.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "Use mock data."
    assert isinstance(messages[1], HumanMessage)
    assert json.loads(messages[1].content) == payload
    assert received_config == config


def test_generate_retries_invalid_xml_output_then_recovers(monkeypatch, backoffs):
    invalid = {"summary": "Bad\x00summary", "findings": ["Illustrative finding"]}
    client, model = stub_client(monkeypatch, [invalid, RESULT])
    result = asyncio.run(client.generate(TaskResult, "Use mock data.", {}))

    assert result.model_dump() == RESULT
    assert len(model.calls) == 2
    assert backoffs == [2]
    assert len(model.calls[1][0]) == 3
    assert "Word/XML" in model.calls[1][0][-1].content


def test_generate_exhausts_retries_without_exposing_upstream_details(monkeypatch, backoffs):
    client, model = stub_client(
        monkeypatch, [ConnectionError("private-provider-secret")] * 3, retries=2
    )
    with pytest.raises(LLMError) as error:
        asyncio.run(client.generate(TaskResult, "Use mock data.", {}))

    assert len(model.calls) == 3
    assert backoffs == [2, 4]
    assert "private-provider-secret" not in str(error.value)


@pytest.mark.parametrize(
    "retry_after,expected_delay", [("4", 4), ("3.945", 3.945), ("0", 2), ("invalid", 2), ("90", 90)]
)
def test_rate_limit_retry_honors_provider_wait_time(monkeypatch, backoffs, retry_after, expected_delay):
    response = httpx.Response(
        429,
        headers={"Retry-After": retry_after},
        request=httpx.Request("POST", "https://api.groq.com/"),
    )
    error = RateLimitError("Test rate limit", response=response, body={})
    client, model = stub_client(monkeypatch, [error, RESULT])
    result = asyncio.run(client.generate(TaskResult, "Use mock data.", {}))

    assert result.model_dump() == RESULT
    assert len(model.calls) == 2
    assert backoffs == [expected_delay]
    assert all(len(messages) == 2 for messages, _ in model.calls)


def test_attempt_timeout_cancels_slow_call_and_retries(monkeypatch, backoffs):
    cancelled = []

    async def never_finish():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    client, model = stub_client(monkeypatch, [never_finish, RESULT], timeout=0.01)
    result = asyncio.run(client.generate(TaskResult, "Use mock data.", {}))

    assert result.model_dump() == RESULT
    assert cancelled == [True]
    assert len(model.calls) == 2
    assert backoffs == [2]
    assert all(len(messages) == 2 for messages, _ in model.calls)


def test_caller_cancellation_propagates_without_retry(monkeypatch, backoffs):
    async def cancel_request():
        started = asyncio.Event()

        async def never_finish():
            started.set()
            await asyncio.Event().wait()

        client, model = stub_client(monkeypatch, [never_finish])
        request = asyncio.create_task(client.generate(TaskResult, "Use mock data.", {}))
        async with asyncio.timeout(1):
            await started.wait()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        return model

    model = asyncio.run(cancel_request())
    assert len(model.calls) == 1
    assert backoffs == []


@pytest.mark.parametrize("provider", ["poolside", "groq", "ollama"])
@pytest.mark.parametrize("schema", [type(result) for result, _ in OUTPUT_CASES])
def test_real_provider_classes_accept_all_output_schemas_without_network(poolside_http, provider, schema):
    settings = Settings(
        _env_file=None,
        llm_provider=provider,
        poolside_api_key="test-only-key",
        groq_api_key="test-only-key",
        langsmith_tracing=False,
    )
    client = LLMClient(settings)
    budget = TOKEN_BUDGET.get(schema.__name__, settings.llm_max_tokens)
    model = client._get_model(budget)
    structured = (
        model | PydanticOutputParser(pydantic_object=schema)
        if provider == "poolside"
        else model.with_structured_output(schema, method="json_schema")
    )

    assert client._get_model(budget) is model
    assert callable(structured.ainvoke)
    assert poolside_http.requests == []
    if provider == "poolside":
        assert isinstance(model, ChatOpenAI)
        assert model.max_tokens is None
        assert model.use_responses_api is False
        assert model.extra_body == {
            "max_tokens": budget, "chat_template_kwargs": {"enable_thinking": False},
        }
    elif provider == "groq":
        assert model.max_tokens == budget
    else:
        assert model.num_predict == budget


def test_long_rate_limit_wait_fails_without_sleeping(monkeypatch, backoffs):
    response = httpx.Response(
        429, headers={"Retry-After": "91"}, request=httpx.Request("POST", "https://api.groq.com/")
    )
    error = RateLimitError("Test rate limit", response=response, body={})
    client, model = stub_client(monkeypatch, [error], retries=3)
    with pytest.raises(LLMRateLimitError) as caught:
        asyncio.run(client.generate(TaskResult, "Use mock data.", {}))
    assert caught.value.retry_after == 91
    assert len(model.calls) == 1
    assert backoffs == []


def test_exhausted_rate_limits_preserve_retry_after(monkeypatch, backoffs):
    response = httpx.Response(
        429, headers={"Retry-After": "3"}, request=httpx.Request("POST", "https://api.groq.com/")
    )
    error = RateLimitError("Test rate limit", response=response, body={})
    client, model = stub_client(monkeypatch, [error] * 4, retries=3)
    with pytest.raises(LLMRateLimitError) as caught:
        asyncio.run(client.generate(TaskResult, "Use mock data.", {}))
    assert caught.value.retry_after == 3
    assert len(model.calls) == 4
    assert backoffs == [3, 4, 8]


def test_throttle_spaces_concurrent_call_starts(monkeypatch):
    clock = [100.0]
    delays = []
    real_sleep = asyncio.sleep

    async def advance_clock(delay):
        delays.append(delay)
        clock[0] += delay
        await real_sleep(0)

    monkeypatch.setattr(providers, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(providers.asyncio, "sleep", advance_clock)
    client = LLMClient(Settings(_env_file=None, llm_min_interval_seconds=3))

    async def exercise():
        starts = []

        async def reserve():
            await client._throttle()
            starts.append(clock[0])

        await asyncio.gather(*(reserve() for _ in range(4)))
        return starts

    assert asyncio.run(exercise()) == [100, 103, 106, 109]
    assert delays == [3, 3, 3]


def test_disabled_throttle_does_not_sleep(monkeypatch, backoffs):
    client = LLMClient(Settings(_env_file=None, llm_min_interval_seconds=0))
    asyncio.run(client._throttle())
    assert backoffs == []


@pytest.mark.parametrize("effort", ["low", ""])
def test_groq_reasoning_effort_is_optional(monkeypatch, effort):
    import langchain_groq

    captured = {}
    model = StubModel([])

    def build_model(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(langchain_groq, "ChatGroq", build_model)
    settings = Settings(
        _env_file=None, llm_provider="groq", groq_api_key="test-only-key", groq_reasoning_effort=effort
    )
    assert LLMClient(settings)._get_model(2000) is model
    assert captured["max_tokens"] == 2000
    if effort:
        assert captured["reasoning_effort"] == effort
    else:
        assert "reasoning_effort" not in captured


def test_retry_feedback_redacts_configured_keys(monkeypatch, backoffs):
    client, model = stub_client(monkeypatch, [ValueError("Rejected test-only-key"), RESULT])
    asyncio.run(client.generate(TaskResult, "Use mock data.", {}))
    correction = model.calls[1][0][-1].content
    assert "test-only-key" not in correction
    assert "[redacted]" in correction


@pytest.mark.parametrize("provider", ["poolside", "groq", "ollama"])
def test_models_are_cached_by_token_budget(poolside_http, provider):
    settings = Settings(
        _env_file=None, llm_provider=provider,
        poolside_api_key="test-only-key", groq_api_key="test-only-key",
    )
    client = LLMClient(settings)
    small = client._get_model(2000)
    large = client._get_model(5000)
    assert client._get_model(2000) is small
    assert client._get_model(5000) is large
    assert small is not large
    assert set(client._models) == {2000, 5000}
    assert poolside_http.requests == []


@pytest.mark.parametrize(
    "result,budget", OUTPUT_CASES + [(CustomTaskResult.model_validate(RESULT), 4096)],
)
def test_generate_selects_schema_specific_budget(monkeypatch, result, budget):
    client, model = stub_client(monkeypatch, [result])
    actual = asyncio.run(client.generate(type(result), "Use mock data.", {}))
    assert actual is result
    assert model.token_budgets == [budget]


def test_error_details_are_logged_truncated_and_redacted(monkeypatch, backoffs, caplog):
    caplog.set_level(logging.WARNING, logger="app.llm.providers")
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/"))
    error = BadRequestError("Malformed tool call test-only-key " + "x" * 400, response=response, body={})
    client, _ = stub_client(monkeypatch, [error, RESULT])
    asyncio.run(client.generate(TaskResult, "Use mock data.", {}))
    record = next(item for item in caplog.records if item.name == "app.llm.providers")
    text = record.getMessage()
    assert "status=400" in text and "detail=" in text
    assert "[redacted]" in text and "test-only-key" not in text
    assert len(record.args[-1]) <= 300


def test_poolside_defaults_are_isolated_from_ambient_configuration():
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "poolside"
    assert settings.poolside_model == settings.model_name == "poolside/laguna-s-2.1"
    assert settings.poolside_base_url == "https://inference.poolside.ai/v1"
    assert settings.poolside_api_key.get_secret_value() == ""
    assert settings.llm_temperature == 0.2


@pytest.mark.parametrize("provider,model_field,model", [
    ("poolside", "poolside_model", "poolside/test-model"),
    ("groq", "groq_model", "test/groq-model"),
    ("ollama", "ollama_model", "test-ollama-model"),
])
def test_explicit_provider_environment_selects_model(monkeypatch, provider, model_field, model):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(model_field.upper(), model)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == provider
    assert settings.model_name == model


@pytest.mark.parametrize("selector", ["gemini", "unknown"])
def test_settings_reject_removed_or_unknown_provider(selector):
    with pytest.raises(ValidationError, match="llm_provider"):
        Settings(_env_file=None, llm_provider=selector)


@pytest.mark.parametrize("explicit", [False, True])
def test_poolside_sdk_construction_uses_only_confirmed_options(monkeypatch, explicit):
    import langchain_openai

    captured = {}
    model = StubModel([])

    def build_model(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", build_model)
    if explicit:
        for name, value in {
            "POOLSIDE_API_KEY": "explicit-poolside-test-key",
            "POOLSIDE_BASE_URL": "https://poolside-test.invalid/v1",
            "POOLSIDE_MODEL": "poolside/test-model",
            "LLM_TEMPERATURE": "0.65",
            "LLM_TIMEOUT_SECONDS": "12",
        }.items():
            monkeypatch.setenv(name, value)
        settings = Settings(_env_file=None)
    else:
        settings = Settings(_env_file=None, poolside_api_key="poolside-test-key")
    assert LLMClient(settings)._get_model(2000) is model
    assert captured == {
        "api_key": "explicit-poolside-test-key" if explicit else "poolside-test-key",
        "base_url": "https://poolside-test.invalid/v1" if explicit else "https://inference.poolside.ai/v1",
        "model": "poolside/test-model" if explicit else "poolside/laguna-s-2.1",
        "temperature": 0.65 if explicit else 0.2,
        "timeout": 12 if explicit else 90,
        "max_retries": 0,
        "use_responses_api": False,
        "extra_body": {"max_tokens": 2000, "chat_template_kwargs": {"enable_thinking": False}},
    }


def test_poolside_requires_its_own_key_before_sdk_construction(monkeypatch):
    import langchain_openai

    def unexpected_model(**kwargs):
        pytest.fail("An empty Poolside key must be rejected before SDK construction.")

    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-test-key")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", unexpected_model)
    client = LLMClient(Settings(
        _env_file=None, poolside_api_key="", groq_api_key="unrelated-groq-test-key",
    ))
    with pytest.raises(LLMConfigurationError, match="POOLSIDE_API_KEY"):
        client._get_model(2000)
    assert client._models == {}


@pytest.mark.parametrize(
    "expected,budget", OUTPUT_CASES + [(CustomTaskResult.model_validate(RESULT), 3072)],
)
def test_poolside_wire_contract_and_pydantic_parsing_for_each_schema(poolside_http, expected, budget):
    schema = type(expected)
    poolside_http.replies.append(expected.model_dump(mode="json"))
    payload = {"request": "Analyze an illustrative business budget."}
    result = asyncio.run(poolside_http.client.generate(schema, "Use mock data.", payload))

    assert isinstance(result, schema)
    assert result == expected
    assert len(poolside_http.requests) == 1
    request = poolside_http.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://inference.poolside.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer poolside-test-key"
    body = json.loads(request.content)
    assert body["model"] == "poolside/laguna-s-2.1"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == budget
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert not {
        "max_completion_tokens", "max_output_tokens", "response_format", "tools",
        "tool_choice", "reasoning_effort", "extra_body",
    } & body.keys()
    instructions = PydanticOutputParser(pydantic_object=schema).get_format_instructions()
    assert body["messages"] == [
        {"role": "system", "content": f"Use mock data.\n\n{instructions}"},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    wire_schema = json.loads(body["messages"][0]["content"].split("```")[1])
    assert wire_schema["properties"] == schema.model_json_schema()["properties"]
    assert wire_schema["required"] == schema.model_json_schema()["required"]


@pytest.mark.parametrize("invalid,detail", [
    ("not valid JSON", "Invalid json output"),
    ({"summary": "Missing required findings"}, "findings"),
    ({**RESULT, "unexpected": "not allowed"}, "Extra inputs are not permitted"),
    ({"summary": "Bad\x00summary", "findings": ["Illustrative finding"]}, "Word/XML"),
])
def test_poolside_parser_retries_json_schema_and_xml_failures(poolside_http, backoffs, invalid, detail):
    poolside_http.replies.extend([invalid, RESULT])
    result = asyncio.run(poolside_http.client.generate(TaskResult, "Use mock data.", {}))
    assert result.model_dump() == RESULT
    assert len(poolside_http.requests) == 2
    assert backoffs == [2]
    first, retry = [json.loads(request.content)["messages"] for request in poolside_http.requests]
    assert retry[:2] == first
    assert len(retry) == 3
    assert retry[-1]["role"] == "user"
    assert detail in retry[-1]["content"]
    assert "Return a corrected, complete result" in retry[-1]["content"]


def test_poolside_exhausted_errors_redact_logs_feedback_and_public_message(poolside_http, backoffs, caplog):
    caplog.set_level(logging.WARNING, logger="app.llm.providers")
    secrets = ("poolside-test-key", "groq-test-key", "langsmith-test-key")
    private_detail = "Private provider failure " + " ".join(secrets) + " " + "x" * 400
    poolside_http.replies.extend([
        httpx.Response(400, json={"error": {"message": private_detail, "type": "invalid_request_error"}})
        for _ in range(3)
    ])
    poolside_http.client.settings.llm_max_retries = 2
    with pytest.raises(LLMError) as caught:
        asyncio.run(poolside_http.client.generate(TaskResult, "Use mock data.", {}))
    assert not isinstance(caught.value, LLMRateLimitError)
    assert len(poolside_http.requests) == 3
    assert backoffs == [2, 4]
    assert private_detail not in str(caught.value)
    assert "Private provider failure" not in str(caught.value)
    feedback = [json.loads(request.content)["messages"][-1]["content"] for request in poolside_http.requests[1:]]
    assert all("[redacted]" in text for text in feedback)
    assert "[redacted]" in caplog.text
    assert all(secret not in caplog.text and all(secret not in text for text in feedback) for secret in secrets)
    records = [record for record in caplog.records if record.name == "app.llm.providers"]
    assert len(records) == 3
    assert all("status=400" in record.getMessage() and len(record.args[-1]) <= 300 for record in records)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("retry_after,delay", [("4.5", 4.5), ("invalid", 2), ("0", 2)])
def test_poolside_sdk_rate_limit_retry_after_including_wrapped_cause(
    monkeypatch, poolside_http, backoffs, wrapped, retry_after, delay
):
    if wrapped:
        real_invoke = ChatOpenAI.ainvoke

        async def wrap_rate_limit(self, *args, **kwargs):
            try:
                return await real_invoke(self, *args, **kwargs)
            except OpenAIRateLimitError as exc:
                raise RuntimeError("Wrapped Poolside rate limit") from exc

        monkeypatch.setattr(ChatOpenAI, "ainvoke", wrap_rate_limit)
    poolside_http.replies.extend([
        httpx.Response(429, headers={"Retry-After": retry_after}, json={
            "error": {"message": "Test rate limit", "type": "rate_limit_error", "code": "rate_limit_exceeded"},
        }),
        RESULT,
    ])
    result = asyncio.run(poolside_http.client.generate(TaskResult, "Use mock data.", {}))
    assert result.model_dump() == RESULT
    assert len(poolside_http.requests) == 2
    assert backoffs == [delay]
    messages = [json.loads(request.content)["messages"] for request in poolside_http.requests]
    assert len(messages[0]) == 2 and messages[0] == messages[1]


@pytest.mark.parametrize("retry_after,retries,requests,delays", [
    ("91", 3, 1, []),
    ("3", 3, 4, [3, 4, 8]),
    ("invalid", 0, 1, []),
])
def test_poolside_long_or_exhausted_rate_limits_preserve_safe_errors(
    poolside_http, backoffs, retry_after, retries, requests, delays
):
    poolside_http.client.settings.llm_max_retries = retries
    poolside_http.replies.extend([
        httpx.Response(429, headers={"Retry-After": retry_after}, json={
            "error": {"message": "Private poolside-test-key quota detail", "type": "rate_limit_error"},
        }) for _ in range(requests)
    ])
    with pytest.raises(LLMRateLimitError) as caught:
        asyncio.run(poolside_http.client.generate(TaskResult, "Use mock data.", {}))
    assert isinstance(caught.value.__cause__, OpenAIRateLimitError)
    assert caught.value.retry_after == (None if retry_after == "invalid" else float(retry_after))
    assert "Private" not in str(caught.value) and "poolside-test-key" not in str(caught.value)
    assert len(poolside_http.requests) == requests
    assert backoffs == delays
