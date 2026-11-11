import json

import pytest
import yaml

from matheasy.llm import (
    BudgetExceeded,
    FakeBackend,
    LLMCallError,
    LLMClient,
    LLMOutputError,
    LLMUnavailable,
    load_config,
)
from matheasy.llm import client as client_module
from matheasy.llm.backends import BackendError, Message, rate_limit, to_gemini_contents
from matheasy.llm.config import DEFAULT_CONFIG_PATH


@pytest.fixture
def config_path(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["provider"] = "fake"
    raw["budget"]["session_tokens"] = 50
    raw["cache"]["dir"] = str(tmp_path / "cache")
    raw["logging"]["dir"] = str(tmp_path / "logs")
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def make_client(config_path, reply=None, sleep=None):
    backend = FakeBackend(reply)
    waited = [] if sleep is None else sleep
    client = LLMClient(load_config(config_path), backend=backend, run_id="test", sleep=waited.append)
    client.waited = waited
    return client, backend


def read_log(client):
    return [json.loads(line) for line in client.log_path.read_text().splitlines()]


def user(text):
    return [{"role": "user", "content": text}]


def test_repo_config_loads():
    cfg = load_config()
    assert cfg.provider == "gemini"
    assert cfg.model() == cfg.models["backbone"]
    assert len(cfg.hash) == 12


def test_rejects_unpinned_model(config_path):
    raw = yaml.safe_load(config_path.read_text())
    raw["models"]["backbone"] = "gemini-flash-latest"
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="pin a stable model"):
        load_config(config_path)


def test_call_charges_input_output_and_logs(config_path):
    client, _ = make_client(config_path, reply=lambda r: "two words")
    session = client.session("s1", condition="full")
    result = session.call("hinter", user("one two three"), system="be kind")

    assert result.text == "two words"
    assert (result.input_tokens, result.output_tokens) == (5, 2)
    assert session.used_tokens == 7
    assert session.usage_by_role["hinter"] == 7

    [record] = read_log(client)
    assert record["event"] == "call"
    assert record["role"] == "hinter"
    assert record["condition"] == "full"
    assert record["config_hash"] == client.config.hash
    assert record["session_used_tokens"] == 7
    assert record["response"] == "two words"


def test_budget_caps_output_then_refuses(config_path):
    client, backend = make_client(config_path, reply=lambda r: "x " * 20)
    session = client.session("s1", condition="full")  # budget 50

    session.call("hinter", user("a " * 10))  # 10 in + 20 out = 30
    assert session.remaining_tokens == 20
    session.call("hinter", user("b"))
    assert backend.requests[-1].max_output_tokens == 19  # capped to what is left after the prompt

    assert session.remaining_tokens == 0
    with pytest.raises(BudgetExceeded):
        session.call("hinter", user("c"))
    assert read_log(client)[-1]["event"] == "budget_exceeded"


def test_unlimited_session(config_path):
    client, _ = make_client(config_path, reply=lambda r: "x " * 100)
    session = client.session("student-1", condition="sim_student", budget_tokens=None)
    for _ in range(3):
        session.call("sim_student", user("hello"))
    assert session.remaining_tokens is None
    assert session.used_tokens > 50


def test_cache_hit_skips_backend_but_still_charges(config_path):
    client, backend = make_client(config_path, reply=lambda r: "same answer")
    first = client.session("s1", condition="full").call("critic", user("check this"))
    second_session = client.session("s2", condition="full")
    second = second_session.call("critic", user("check this"))

    assert len(backend.requests) == 1
    assert second.cached and not first.cached
    assert second_session.used_tokens == first.charged_tokens
    assert read_log(client)[-1]["cached"] is True


def test_no_cache_when_temperature_above_zero(config_path):
    client, backend = make_client(config_path)
    session = client.session("s1", condition="full", budget_tokens=None)
    session.call("sim_student", user("hi"), temperature=0.7)
    session.call("sim_student", user("hi"), temperature=0.7)
    assert len(backend.requests) == 2


def test_json_mode_parses_and_rejects_bad_json(config_path):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    client, _ = make_client(config_path, reply=lambda r: '{"ok": true}')
    result = client.session("s1", "full").call("critic", user("verdict?"), json_schema=schema)
    assert result.parsed == {"ok": True}

    client, _ = make_client(config_path, reply=lambda r: "not json")
    with pytest.raises(LLMOutputError):
        client.session("s1", "full").call("critic", user("other verdict?"), json_schema=schema)


def test_bad_replies_are_not_cached(config_path):
    replies = iter(["not json", '{"ok": true}'])
    client, backend = make_client(config_path, reply=lambda r: next(replies))
    session = client.session("s1", "full", budget_tokens=None)
    with pytest.raises(LLMOutputError):
        session.call("critic", user("verdict?"), json_schema={"type": "object"})
    assert session.call("critic", user("verdict?"), json_schema={"type": "object"}).parsed == {"ok": True}
    assert len(backend.requests) == 2


def test_empty_reply_raises_after_charging(config_path):
    client, _ = make_client(config_path, reply=lambda r: "")
    session = client.session("s1", "full")
    with pytest.raises(LLMOutputError):
        session.call("hinter", user("one two"))
    assert session.used_tokens == 2


def test_backend_error_is_logged_and_wrapped(config_path):
    def fail(request):
        raise BackendError("bad request", code=400)

    client, _ = make_client(config_path, reply=fail)
    with pytest.raises(LLMCallError, match="bad request"):
        client.session("s1", "full").call("hinter", user("hi"))
    [record] = read_log(client)
    assert record["event"] == "error"
    assert "bad request" in record["error"]
    assert client.waited == []  # a rejected request is not tried again


def failing(errors, then="ok"):
    """A reply that raises each error in turn, then answers."""
    queue = list(errors)

    def reply(request):
        if queue:
            raise queue.pop(0)
        return then
    return reply


def test_rate_limit_waits_as_asked_then_succeeds(config_path):
    client, backend = make_client(config_path, reply=failing([
        BackendError("slow down", code=429, retry_after=31.0),
        BackendError("slow down", code=429),  # no delay given
        BackendError("connection reset"),  # no answer at all
        BackendError("unavailable", code=503),
    ]))
    result = client.session("s1", "full", budget_tokens=None).call("hinter", user("hi"))
    assert result.text == "ok" and len(backend.requests) == 5
    assert client.waited == [32.0, client_module.RATE_LIMIT_WAIT_S + 1, 40.0, 80.0]  # back-off doubles each wait
    events = [r["event"] for r in read_log(client)]
    assert events == ["error"] * 4 + ["call"]
    assert "waiting 32s" in read_log(client)[0]["error"]


@pytest.mark.parametrize("error, waits", [
    (BackendError("daily quota", code=429, daily=True), 0),  # waiting minutes won't help
    (BackendError("slow down", code=429, retry_after=5.0), client_module.MAX_WAITS),
    (BackendError("down", code=503), client_module.MAX_WAITS),
])
def test_a_quota_or_outage_that_does_not_clear_is_unavailable(config_path, error, waits):
    client, _ = make_client(config_path, reply=failing([error] * 10))
    with pytest.raises(LLMUnavailable):
        client.session("s1", "full", budget_tokens=None).call("hinter", user("hi"))
    assert len(client.waited) == waits
    assert not issubclass(LLMUnavailable, LLMCallError)  # the tutor's fallbacks must not catch it


def test_rate_limit_reads_gemini_error_bodies():
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "31s"}]}}
    assert rate_limit(body) == (31.0, False)
    body["error"]["details"][0]["violations"][0]["quotaId"] = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    assert rate_limit(body) == (31.0, True)
    assert rate_limit({}, "Quota exceeded. Please retry in 12.5s.") == (12.5, False)
    assert rate_limit(None, "server error") == (None, False)


def test_a_call_the_budget_cannot_cover_is_not_started(config_path):
    client, backend = make_client(config_path, reply=lambda r: "x")
    session = client.session("s1", condition="full")  # budget 50
    with pytest.raises(BudgetExceeded):
        session.call("hinter", user("word " * 40))  # about 67 tokens by estimate: over the whole budget
    assert backend.requests == [] and session.used_tokens == 0
    assert read_log(client)[-1]["event"] == "budget_exceeded"


@pytest.mark.parametrize("messages, match", [
    ([], "empty"),
    ([{"role": "system", "content": "x"}], "role"),
    ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "last message"),
])
def test_message_validation(config_path, messages, match):
    client, _ = make_client(config_path)
    with pytest.raises(ValueError, match=match):
        client.session("s1", "full").call("hinter", messages)


def test_gemini_contents_map_assistant_to_model():
    contents = to_gemini_contents([Message("user", "q"), Message("assistant", "a"), Message("user", "q2")])
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].text == "a"


def test_gemini_backend_builds_request_and_reads_usage(config_path):
    from types import SimpleNamespace

    from google.genai import types

    from matheasy.llm.backends import GeminiBackend, Request

    seen = {}

    def generate_content(*, model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return types.GenerateContentResponse(
            candidates=[types.Candidate(
                content=types.Content(role="model", parts=[types.Part.from_text(text='{"ok": true}')]),
                finish_reason="STOP",
            )],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=11, candidates_token_count=4, thoughts_token_count=3,
            ),
        )

    backend = GeminiBackend(load_config(config_path))
    backend._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    response = backend.generate(Request(
        model="gemini-x", system="sys", messages=(Message("user", "hi"),), temperature=0.0,
        max_output_tokens=64, seed=1, thinking_budget=0, json_schema={"type": "object"},
    ))

    assert (response.text, response.finish_reason) == ('{"ok": true}', "STOP")
    assert (response.input_tokens, response.output_tokens, response.thought_tokens) == (11, 4, 3)
    config = seen["config"]
    assert seen["model"] == "gemini-x"
    assert config.system_instruction == "sys"
    assert config.max_output_tokens == 64
    assert config.thinking_config.thinking_budget == 0
    assert config.response_mime_type == "application/json"


def test_cache_is_separate_per_provider(config_path):
    class OtherBackend(FakeBackend):
        provider = "other"

    cfg = load_config(config_path)
    LLMClient(cfg, backend=FakeBackend(lambda r: "fake"), run_id="a").session("s", "full").call("hinter", user("q"))
    other = OtherBackend(lambda r: "real")
    result = LLMClient(cfg, backend=other, run_id="b").session("s", "full").call("hinter", user("q"))
    assert result.text == "real" and not result.cached
