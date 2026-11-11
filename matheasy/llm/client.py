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

When the provider says to slow down (429) or cannot be reached, the call waits
and tries again, as long as the provider asked or up to MAX_WAIT_S, at most
MAX_WAITS times. If that does not help, or a daily quota is used up, it raises
LLMUnavailable. That is not a tutor failure, so no part of the tutor catches it:
a run stops instead of filling sessions with fallback messages, and the next run
redoes the unfinished session.
"""

from __future__ import annotations

import hashlib
import json
import math
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

MAX_WAITS = 5  # waits for a rate limit or an outage, per call
MAX_WAIT_S = 120.0  # longest single wait
RATE_LIMIT_WAIT_S = 60.0  # when a 429 does not say how long
CHARS_PER_TOKEN = 3  # for estimating a prompt before sending it: English is about 4, maths fewer
MIN_OUTPUT_TOKENS = 16  # a call with less room than this for its reply is not started


class BudgetExceeded(Exception):
    """The session has no tokens left. The caller should fall back (e.g. safe template)."""


class LLMCallError(Exception):
    """The model call failed after retries."""


class LLMUnavailable(Exception):
    """The model can't be reached: a quota is used up, or the service is down even after
    waiting. Not an LLMCallError, so the tutor's fallbacks don't catch it (see above)."""


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
        sleep=time.sleep,  # tests pass a stub, so waits take no time
    ):
        self.config = config or load_config()
        self.sleep = sleep
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
        The prompt's size is estimated before the call (CHARS_PER_TOKEN, on the
        safe side), the reply is capped to what is left of the budget after it,
        and a call with too little room left is not started. The real usage is
        charged after the call, so a session goes over only if the estimate does.
        """
        cfg = self.client.config
        msgs = _to_messages(messages)
        max_out = max_output_tokens or cfg.max_output_tokens
        remaining = self.remaining_tokens
        if remaining is not None:
            room = remaining - estimate_tokens(system, msgs, json_schema)
            if room < MIN_OUTPUT_TOKENS:
                self._log_event(role, "budget_exceeded", cfg.model(model))
                raise BudgetExceeded(
                    f"session {self.session_id} used {self.used_tokens}/{self.budget_tokens} tokens"
                )
            max_out = min(max_out, room)

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
        waits = 0
        while response is None:
            try:
                response = self.client.backend.generate(request)
            except BackendError as e:
                wait = _wait_for(e, waits)
                note = f" (waiting {wait:.0f}s, then trying again)" if wait is not None else ""
                self._log_call(role, request, None, cached=False, latency_s=time.monotonic() - start,
                               error=f"{e}{note}")
                if wait is None:
                    if e.code == 429 or e.code is None or e.code >= 500:
                        raise LLMUnavailable(str(e)) from e
                    raise LLMCallError(str(e)) from e
                waits += 1
                self.client.sleep(wait)
                start = time.monotonic()
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


def estimate_tokens(system: Optional[str], messages: Sequence[Message], json_schema: Optional[dict]) -> int:
    """An upper estimate of a prompt's input tokens, from its length."""
    chars = len(system or "") + sum(len(m.content) for m in messages)
    chars += len(json.dumps(json_schema)) if json_schema else 0
    return math.ceil(chars / CHARS_PER_TOKEN)


def _wait_for(error: BackendError, waits: int) -> Optional[float]:
    """Seconds to wait before trying again, or None to give up: a daily quota, a request
    the provider rejected (4xx other than 429), or MAX_WAITS waits already."""
    if waits >= MAX_WAITS or error.daily:
        return None
    if error.code == 429:
        return min(error.retry_after or RATE_LIMIT_WAIT_S, MAX_WAIT_S) + 1
    if error.code is None or error.code >= 500:  # no answer, or a server error: back off
        return min(10.0 * 2 ** waits, MAX_WAIT_S)
    return None


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
