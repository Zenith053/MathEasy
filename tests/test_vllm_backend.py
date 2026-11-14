"""The vLLM backend (an open model on our own GPU) and the switch between config files."""

import json

import httpx
import pytest
import yaml

from matheasy.llm import LLMClient
from matheasy.llm.backends import BackendError, Message, Request, VLLMBackend, make_backend, to_openai_body
from matheasy.llm.config import CONFIG_ENV, DEFAULT_CONFIG_PATH, REPO_ROOT, load_config

VLLM_CONFIG = REPO_ROOT / "config" / "llm_vllm.yaml"


def request(**kw):
    base = dict(model="Qwen/Qwen2.5-7B-Instruct-AWQ", system="Be brief.",
                messages=(Message("user", "hi"), Message("assistant", "hello"), Message("user", "2+2?")),
                temperature=0.0, max_output_tokens=50, seed=7, thinking_budget=None, json_schema=None)
    return Request(**{**base, **kw})


def backend_answering(handler):
    backend = VLLMBackend(load_config(VLLM_CONFIG))
    backend._client = httpx.Client(base_url="http://test/v1", transport=httpx.MockTransport(handler))
    return backend


def ok(content, finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": 12, "completion_tokens": 3}})


def test_request_becomes_an_openai_chat_body():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    body = to_openai_body(request(json_schema=schema))
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "hello"}, {"role": "user", "content": "2+2?"}]
    assert (body["temperature"], body["max_tokens"], body["seed"]) == (0.0, 50, 7)
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
    plain = to_openai_body(request(system=None, seed=None))
    assert plain["messages"][0]["role"] == "user" and "seed" not in plain and "response_format" not in plain


def test_reply_text_tokens_and_finish_reason_are_read():
    sent = []
    backend = backend_answering(lambda r: sent.append(json.loads(r.content)) or ok("4"))
    reply = backend.generate(request())
    assert (reply.text, reply.finish_reason, reply.input_tokens, reply.output_tokens) == ("4", "STOP", 12, 3)
    assert sent[0]["model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    cut = backend_answering(lambda r: ok("4 because", "length")).generate(request())
    assert cut.finish_reason == "MAX_TOKENS"  # the client's word for a truncated reply


def test_json_wrapped_in_fences_or_words_is_unwrapped():
    schema = {"type": "object"}
    for wrapped in ['```json\n{"x": 4}\n```', 'Here it is: {"x": 4} Hope that helps.']:
        reply = backend_answering(lambda r, w=wrapped: ok(w)).generate(request(json_schema=schema))
        assert reply.text == '{"x": 4}'
    broken = backend_answering(lambda r: ok("no json {here")).generate(request(json_schema=schema))
    assert broken.text == "no json {here"  # left as it is: the client rejects it
    prose = backend_answering(lambda r: ok("a {b} c")).generate(request())
    assert prose.text == "a {b} c"  # no schema asked: never touched


def test_server_errors_and_a_server_that_is_down_raise_backend_errors():
    with pytest.raises(BackendError) as e:
        backend_answering(lambda r: httpx.Response(400, text="bad schema")).generate(request())
    assert e.value.code == 400

    def down(r):
        raise httpx.ConnectError("refused")

    with pytest.raises(BackendError) as e:
        backend_answering(down).generate(request())
    assert e.value.code is None  # the client backs off and tries again, as for Gemini


def test_the_client_parses_json_from_vllm(tmp_path):
    cfg = load_config(VLLM_CONFIG)
    import dataclasses
    cfg = dataclasses.replace(cfg, log_dir=tmp_path / "llm", cache_dir=tmp_path / "cache")
    backend = backend_answering(lambda r: ok('{"x": 4}'))
    result = LLMClient(cfg, backend=backend, run_id="t").session("s", "eval", budget_tokens=None).call(
        "test", [{"role": "user", "content": "2+2?"}], json_schema={"type": "object"})
    assert result.parsed == {"x": 4}
    logged = json.loads((tmp_path / "llm" / "t.jsonl").read_text().splitlines()[0])
    assert logged["provider"] == "vllm" and logged["model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"


def test_the_vllm_config_differs_from_the_teams_only_in_the_model_and_server():
    team, ours = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()), yaml.safe_load(VLLM_CONFIG.read_text())
    for key in ("generation", "budget", "retry", "cache", "logging"):
        same = {k: v for k, v in ours[key].items() if k != "thinking_budget"}
        assert same == {k: v for k, v in team[key].items() if k != "thinking_budget"}, key
    cfg = load_config(VLLM_CONFIG)
    assert cfg.provider == "vllm" and cfg.base_url == "http://localhost:8000/v1"
    assert isinstance(make_backend(cfg), VLLMBackend)


def test_the_environment_variable_picks_the_config_file(monkeypatch):
    team_hash = load_config().hash
    monkeypatch.setenv(CONFIG_ENV, "config/llm_vllm.yaml")
    assert load_config().provider == "vllm"
    monkeypatch.delenv(CONFIG_ENV)
    assert load_config().provider == "gemini" and load_config().hash == team_hash


def test_vllm_without_a_server_address_is_refused(tmp_path):
    raw = yaml.safe_load(VLLM_CONFIG.read_text())
    del raw["base_url"]
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="base_url"):
        load_config(path)
