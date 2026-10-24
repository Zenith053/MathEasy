"""The shared LLM client. Every model call in the project goes through here.

    from matheasy.llm import LLMClient

    llm = LLMClient()                                   # reads config/llm.yaml
    session = llm.session("student-017", condition="full")
    result = session.call(
        "hinter",
        [{"role": "user", "content": "..."}],
        system="You are a maths tutor...",
    )
    print(result.text, session.used_tokens)

One session = one tutor session with one student. The session enforces the
token budget, so every condition gets the same budget in code. Calls that are
not part of the tutor (simulated students, LLM judges) use a session with
budget_tokens=None.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from .backends import Backend, BackendError, BackendResponse, Message, Request, make_backend
from .config import LLMConfig, load_config

_CONFIG_BUDGET = object()  # sentinel: use the budget from the config


class BudgetExceeded(Exception):
    """The session has no tokens left. The caller should fall back (e.g. safe template)."""


class LLMCallError(Exception):
    """The model call failed after retries."""


class LLMOutputError(Exception):
    """The model answered, but with no text or with JSON that does not parse."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    parsed: Any  # the decoded JSON when json_schema was given, else None
    finish_reason: str
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    cached: bool

    @property
    def charged_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thought_tokens

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "MAX_TOKENS"


class LLMClient:
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        backend: Optional[Backend] = None,
        run_id: Optional[str] = None,
    ):
        self.config = config or load_config()
        self.backend = backend or make_backend(self.config)
        self.run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.log_path = self.config.log_dir / f"{self.run_id}.jsonl"
        self._log_lock = threading.Lock()

    def session(self, session_id: str, condition: str, budget_tokens=_CONFIG_BUDGET) -> "Session":
        """Start a session. budget_tokens=None means unlimited (not for tutor conditions)."""
        budget = self.config.session_tokens if budget_tokens is _CONFIG_BUDGET else budget_tokens
        return Session(self, session_id, condition, budget)

    # --- cache ---------------------------------------------------------------

    def _cache_path(self, request: Request):
        # The provider is part of the key so fake replies never stand in for real ones.
        blob = {"provider": self.backend.provider, **asdict(request)}
        key = hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()
        return self.config.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, request: Request) -> Optional[BackendResponse]:
        path = self._cache_path(request)
        if not path.exists():
            return None
        return BackendResponse(**json.loads(path.read_text()))

    def _cache_put(self, request: Request, response: BackendResponse) -> None:
        path = self._cache_path(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(asdict(response)))
        tmp.replace(path)

    # --- logging -------------------------------------------------------------

    def _log(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class Session:
    def __init__(self, client: LLMClient, session_id: str, condition: str, budget_tokens: Optional[int]):
        self.client = client
        self.session_id = session_id
        self.condition = condition
        self.budget_tokens = budget_tokens
        self.used_tokens = 0
        self.usage_by_role: dict[str, int] = defaultdict(int)

    @property
    def remaining_tokens(self) -> Optional[int]:
        if self.budget_tokens is None:
            return None
        return max(self.budget_tokens - self.used_tokens, 0)

    def call(
        self,
        role: str,
        messages: Sequence[dict | Message],
        *,
        system: Optional[str] = None,
        model: str = "backbone",
        json_schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> LLMResult:
        """Make one model call as `role` (e.g. "hinter", "critic", "b0_tutor").

        `model` names an entry under `models:` in the config, not a model ID.
        The budget is checked before the call and charged with the real usage
        after it, so a session can overshoot by at most one call's input.
        """
        cfg = self.client.config
        msgs = _to_messages(messages)
        max_out = max_output_tokens or cfg.max_output_tokens
        remaining = self.remaining_tokens
        if remaining is not None:
            if remaining == 0:
                self._log_event(role, "budget_exceeded", cfg.model(model))
                raise BudgetExceeded(
                    f"session {self.session_id} used {self.used_tokens}/{self.budget_tokens} tokens"
                )
            max_out = min(max_out, remaining)

        request = Request(
            model=cfg.model(model),
            system=system,
            messages=tuple(msgs),
            temperature=cfg.temperature if temperature is None else temperature,
            max_output_tokens=max_out,
            seed=cfg.seed,
            thinking_budget=cfg.thinking_budget,
            json_schema=json_schema,
        )

        use_cache = cfg.cache_enabled and request.temperature == 0
        response = self.client._cache_get(request) if use_cache else None
        cached = response is not None
        start = time.monotonic()
        if response is None:
            try:
                response = self.client.backend.generate(request)
            except BackendError as e:
                self._log_call(role, request, None, cached=False, latency_s=time.monotonic() - start, error=str(e))
                raise LLMCallError(str(e)) from e
        latency_s = time.monotonic() - start

        # Cached replies are charged like fresh ones, so results don't depend on the cache.
        charged = response.input_tokens + response.output_tokens + response.thought_tokens
        self.used_tokens += charged
        self.usage_by_role[role] += charged
        self._log_call(role, request, response, cached=cached, latency_s=latency_s)

        if not response.text:
            raise LLMOutputError(f"no text from model (finish_reason={response.finish_reason})")
        parsed = None
        if json_schema is not None:
            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError as e:
                raise LLMOutputError(f"model returned invalid JSON: {e}") from e
        # Only replies that passed the checks above are cached.
        if use_cache and not cached:
            self.client._cache_put(request, response)

        return LLMResult(
            text=response.text,
            parsed=parsed,
            finish_reason=response.finish_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            thought_tokens=response.thought_tokens,
            cached=cached,
        )

    # --- logging -------------------------------------------------------------

    def _base_record(self, role: str, event: str, model_id: str) -> dict:
        cfg = self.client.config
        return {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.client.run_id,
            "session_id": self.session_id,
            "condition": self.condition,
            "role": role,
            "event": event,
            "provider": self.client.backend.provider,
            "model": model_id,
            "config_hash": cfg.hash,
            "session_used_tokens": self.used_tokens,
            "session_budget": self.budget_tokens,
        }

    def _log_event(self, role: str, event: str, model_id: str) -> None:
        self.client._log(self._base_record(role, event, model_id))

    def _log_call(
        self,
        role: str,
        request: Request,
        response: Optional[BackendResponse],
        *,
        cached: bool,
        latency_s: float,
        error: Optional[str] = None,
    ) -> None:
        record = self._base_record(role, "error" if error else "call", request.model)
        record.update({
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "json_mode": request.json_schema is not None,
            "input_tokens": response.input_tokens if response else 0,
            "output_tokens": response.output_tokens if response else 0,
            "thought_tokens": response.thought_tokens if response else 0,
            "finish_reason": response.finish_reason if response else None,
            "cached": cached,
            "latency_ms": round(latency_s * 1000),
            "error": error,
        })
        if self.client.config.log_include_text:
            record["system"] = request.system
            record["messages"] = [asdict(m) for m in request.messages]
            record["response"] = response.text if response else None
        self.client._log(record)


def _to_messages(messages: Sequence[dict | Message]) -> list[Message]:
    if not messages:
        raise ValueError("messages must not be empty")
    out = [m if isinstance(m, Message) else Message(role=m["role"], content=m["content"]) for m in messages]
    for m in out:
        if m.role not in ("user", "assistant"):
            raise ValueError(f"message role must be 'user' or 'assistant', got {m.role!r}")
    if out[-1].role != "user":
        raise ValueError("the last message must come from the user")
    return out
