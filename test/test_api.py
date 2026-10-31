import base64
import threading
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import app.graph.nodes as graph_nodes
from app.config import Settings
from app.llm.providers import LLMRateLimitError
from app.schemas import AgentResponse, DocumentContent
from test.test_graph import FakeLLM, REVIEW_FEEDBACK, assert_valid_docx, example_payload, offline_settings
from test.test_providers import offline_environment


PROVIDER_MODELS = [
    ("poolside", "poolside/laguna-s-2.1"),
    ("groq", "openai/gpt-oss-20b"),
    ("ollama", "llama3.1:8b"),
]


def create_app(settings, llm=None):
    # Defer the default app's Settings() until dotenv and tracing are isolated.
    from app.main import create_app as factory

    return factory(settings, llm)


def provider_settings(output_dir, provider):
    return Settings(_env_file=None, **{
        **offline_settings(output_dir).model_dump(),
        "llm_provider": provider,
        "poolside_api_key": "",
    })


@pytest.mark.parametrize("provider,model", PROVIDER_MODELS)
@pytest.mark.parametrize("example,task_count,reviews,revisions", [
    ("standard", 3, (True,), 0),
    ("complex", 4, (True,), 0),
    ("standard", 3, (False, True), 1),
])
def test_example_request_returns_typed_response_and_downloadable_docx(
    tmp_path, provider, model, example, task_count, reviews, revisions
):
    llm = FakeLLM(task_count=task_count, reviews=reviews)
    with TestClient(create_app(provider_settings(tmp_path, provider), llm)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "provider": provider, "model": model}
        assert llm.calls == []
        response = client.post("/agent", json=example_payload(example))
        assert response.status_code == 200, response.text
        result = AgentResponse.model_validate(response.json())
        run_id = UUID(result.run_id)
        assert result.status == "completed"
        assert result.message and result.summary == llm.documents[-1].summary
        assert len(result.tasks) == len(result.plan.tasks) == task_count
        assert all(task.status == "completed" for task in result.tasks)
        assert [task.title for task in result.tasks] == [task.title for task in result.plan.tasks]
        assert result.review.approved
        assert result.revisions == revisions
        assert result.warnings == []
        assert not result.trace.enabled
        assert result.trace.url is None
        assert result.elapsed_seconds >= 0
        assert result.document.filename == f"{run_id.hex}.docx"
        assert result.document.base64 is None
        assert urlsplit(result.document.download_url).path == f"/documents/{run_id.hex}"
        assert str(tmp_path) not in response.text
        assert llm.review_calls == len(llm.documents) == revisions + 1
        assert llm.configs and all(config is not None for config in llm.configs)
        assert all(config["metadata"]["provider"] == provider for config in llm.configs)
        assert all(config["metadata"]["model"] == model for config in llm.configs)
        if revisions:
            revision = [payload for schema, payload in llm.calls if schema is DocumentContent][-1]
            assert revision["review_feedback"] == [REVIEW_FEEDBACK]
            assert revision["current_draft"] == llm.documents[0].model_dump(mode="json")
            assert len(revision["completed_tasks"]) == task_count
            assert result.summary != llm.documents[0].summary

        download = client.get(result.document.download_url)
        assert download.status_code == 200
        assert download.headers["content-type"] == result.document.media_type
        assert result.document.filename in download.headers["content-disposition"]
        assert "attachment" in download.headers["content-disposition"]
        assert str(tmp_path) not in str(download.headers)
        assert download.content == (tmp_path / result.document.filename).read_bytes()
        assert_valid_docx(download.content, llm.documents[-1], result.plan)
        # UUID input is canonicalized to the same safe, server-generated filename.
        assert client.get(f"/documents/{run_id}").content == download.content


@pytest.mark.parametrize("provider,model", PROVIDER_MODELS)
def test_health_reports_configuration_without_calling_provider(tmp_path, provider, model):
    llm = FakeLLM()
    with TestClient(create_app(provider_settings(tmp_path, provider), llm)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "provider": provider,
        "model": model,
    }
    assert llm.calls == []
    assert llm.configs == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"request": "x" * 14},
        {"request": "x" * 6001},
        {"request": " " * 20},
        {"request": "Write a useful business proposal.", "output_path": "private.docx"},
        {},
    ],
)
def test_invalid_requests_are_rejected_before_generation(tmp_path, payload):
    llm = FakeLLM()
    with TestClient(create_app(offline_settings(tmp_path), llm)) as client:
        response = client.post("/agent", json=payload)
    assert response.status_code == 422
    assert llm.calls == []
    assert list(tmp_path.iterdir()) == []


def test_exhausted_review_returns_completed_with_warnings(tmp_path):
    llm = FakeLLM(reviews=(False,))
    with TestClient(create_app(offline_settings(tmp_path, max_revisions=1), llm)) as client:
        response = client.post("/agent", json=example_payload("standard"))
    assert response.status_code == 200
    result = AgentResponse.model_validate(response.json())
    assert result.status == "completed_with_warnings"
    assert not result.review.approved
    assert result.revisions == 1
    assert result.warnings and result.review.feedback
    assert (tmp_path / result.document.filename).is_file()


def test_provider_failure_returns_502_without_artifact_or_private_details(tmp_path):
    llm = FakeLLM(fail_task=2)
    with TestClient(create_app(offline_settings(tmp_path), llm)) as client:
        response = client.post("/agent", json=example_payload("standard"))
    assert response.status_code == 502
    assert response.json() == {"detail": "The test provider could not produce a valid result."}
    assert llm.task_calls == 2
    assert llm.documents == []
    assert list(tmp_path.iterdir()) == []


def test_missing_groq_configuration_returns_503_without_network_or_artifact(tmp_path):
    # An explicit empty key overrides both .env and any ambient developer credentials.
    with TestClient(create_app(offline_settings(tmp_path))) as client:
        response = client.post("/agent", json=example_payload("standard"))
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_missing_poolside_configuration_returns_503_without_network_or_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-test-key")
    settings = provider_settings(tmp_path, "poolside")
    assert settings.poolside_api_key.get_secret_value() == ""
    with TestClient(create_app(settings)) as client:
        response = client.post("/agent", json=example_payload("standard"))
    assert response.status_code == 503
    assert "POOLSIDE_API_KEY" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "target,error,detail",
    [
        (
            "save_docx",
            OSError("private filesystem path and secret"),
            "The document could not be saved. Check OUTPUT_DIR write permissions.",
        ),
        ("render_docx", RuntimeError("private internal error and secret"), "Agent execution failed."),
    ],
)
def test_render_errors_return_safe_500_without_artifact(tmp_path, monkeypatch, target, error, detail):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(graph_nodes, target, fail)
    with TestClient(create_app(offline_settings(tmp_path), FakeLLM())) as client:
        response = client.post("/agent", json=example_payload("standard"))
    assert response.status_code == 500
    assert response.json() == {"detail": detail}
    assert list(tmp_path.iterdir()) == []


def test_render_timeout_returns_504_and_cannot_publish_late_artifact(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    real_render = graph_nodes.render_docx

    def delayed_render(*args, **kwargs):
        started.set()
        try:
            if not release.wait(timeout=5):
                raise RuntimeError("Test render was not released.")
            return real_render(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(graph_nodes, "render_docx", delayed_render)
    # Shorten only this test's deadline without changing production Settings validation.
    settings = offline_settings(tmp_path).model_copy(update={"agent_timeout_seconds": 1.0})
    with TestClient(create_app(settings, FakeLLM())) as client:
        try:
            response = client.post("/agent", json=example_payload("standard"))
            assert response.status_code == 504
            assert "timed out" in response.json()["detail"]
            assert started.is_set()
            assert list(tmp_path.iterdir()) == []
        finally:
            release.set()
        assert finished.wait(timeout=5)
        assert list(tmp_path.iterdir()) == []


def test_download_rejects_missing_invalid_and_traversal_paths(tmp_path):
    (tmp_path / "secret.txt").write_text("private-secret", encoding="utf-8")
    settings = offline_settings(tmp_path / "documents")
    with TestClient(create_app(settings, FakeLLM())) as client:
        cases = [
            (uuid4().hex, 404),
            ("not-a-uuid", 422),
            ("..%5Csecret.txt", 422),
            ("..%2Fsecret.txt", 404),
        ]
        for identifier, status in cases:
            response = client.get(f"/documents/{identifier}")
            assert response.status_code == status, identifier
            assert "private-secret" not in response.text
            assert str(tmp_path) not in response.text


@pytest.mark.parametrize("include_base64", [False, True])
def test_optional_base64_matches_downloaded_document(tmp_path, include_base64):
    llm = FakeLLM()
    payload = {**example_payload("standard"), "include_base64": include_base64}
    with TestClient(create_app(offline_settings(tmp_path), llm)) as client:
        response = client.post("/agent", json=payload)
        assert response.status_code == 200, response.text
        result = AgentResponse.model_validate(response.json())
        if include_base64:
            decoded = base64.b64decode(result.document.base64, validate=True)
            assert decoded == client.get(result.document.download_url).content
            assert_valid_docx(decoded, llm.documents[-1], result.plan)
        else:
            assert result.document.base64 is None


@pytest.mark.parametrize("retry_after", [12.0, None])
def test_rate_limits_return_429_with_optional_retry_header(tmp_path, retry_after):
    class RateLimitedLLM(FakeLLM):
        async def generate(self, *args, **kwargs):
            raise LLMRateLimitError("Test provider rate limit.", retry_after=retry_after)

    with TestClient(create_app(offline_settings(tmp_path), RateLimitedLLM())) as client:
        response = client.post("/agent", json=example_payload("standard"))
    assert response.status_code == 429
    assert response.json() == {"detail": "Test provider rate limit."}
    if retry_after is not None:
        assert response.headers["retry-after"] == "12"
    else:
        assert "retry-after" not in response.headers
    assert list(tmp_path.iterdir()) == []
