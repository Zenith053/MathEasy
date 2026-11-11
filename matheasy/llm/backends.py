"""Provider backends. The rest of the project never imports a provider SDK.

A backend makes exactly one model call and reports the tokens it used.
Budgets, caching and logging live in client.py, so they behave the same
whichever backend is in use.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

from .config import LLMConfig


@dataclass(frozen=True)
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class Request:
    model: str
    system: Optional[str]
    messages: tuple[Message, ...]
    temperature: float
    max_output_tokens: int
    seed: Optional[int]
    thinking_budget: Optional[int]
    json_schema: Optional[dict]


@dataclass(frozen=True)
class BackendResponse:
    text: Optional[str]  # None when the model returned no text (e.g. blocked)
    finish_reason: str
    input_tokens: int
    output_tokens: int
    thought_tokens: int


class BackendError(Exception):
    """A call failed after the backend's own retries.

    code         the HTTP status, or None when no answer came back (a dropped connection)
    retry_after  seconds the provider asked us to wait, when it said
    daily        a daily quota is used up: waiting minutes will not help
    """

    def __init__(self, message: str, code: Optional[int] = None, *, retry_after: Optional[float] = None,
                 daily: bool = False):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after
        self.daily = daily


class Backend(Protocol):
    provider: str

    def generate(self, request: Request) -> BackendResponse: ...


class GeminiBackend:
    provider = "gemini"

    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = None  # created on first call, so imports work without a key

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            # Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
            self._client = genai.Client(http_options=types.HttpOptions(
                timeout=int(self._config.timeout_s * 1000),
                retry_options=types.HttpRetryOptions(
                    attempts=self._config.retry_attempts,
                    initial_delay=self._config.retry_initial_delay_s,
                    max_delay=self._config.retry_max_delay_s,
                ),
            ))
        return self._client

    def generate(self, request: Request) -> BackendResponse:
        import httpx
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=request.system,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            seed=request.seed,
        )
        if request.thinking_budget is not None:
            config.thinking_config = types.ThinkingConfig(thinking_budget=request.thinking_budget)
        if request.json_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = request.json_schema

        try:
            response = self._get_client().models.generate_content(
                model=request.model, contents=to_gemini_contents(request.messages), config=config,
            )
        except errors.APIError as e:
            retry_after, daily = rate_limit(e.details, e.message)
            raise BackendError(f"Gemini API error {e.code}: {e.message}", code=e.code,
                               retry_after=retry_after, daily=daily) from e
        except (httpx.TransportError, OSError) as e:  # a dropped connection or timeout, after the SDK's retries
            raise BackendError(f"Gemini connection failed: {type(e).__name__}: {e}") from e

        usage = response.usage_metadata
        finish = response.candidates[0].finish_reason if response.candidates else None
        return BackendResponse(
            text=response.text,
            finish_reason=finish.value if finish is not None else "NO_CANDIDATE",
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            thought_tokens=(usage.thoughts_token_count or 0) if usage else 0,
        )


def rate_limit(details, message: Optional[str] = None) -> tuple[Optional[float], bool]:
    """From a Gemini error: (seconds to wait, if it says, and whether a daily quota ran out).

    A 429's body lists the quotas hit (QuotaFailure, e.g. quotaId
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier") and how long to wait
    (RetryInfo, retryDelay "31s"); older messages say "Please retry in 31.2s"."""
    error = details.get("error", details) if isinstance(details, dict) else {}
    retry_after, daily = None, False
    for item in error.get("details") or []:
        if not isinstance(item, dict):
            continue
        delay = re.fullmatch(r"([\d.]+)s", str(item.get("retryDelay", "")))
        if delay:
            retry_after = float(delay.group(1))
        for violation in item.get("violations") or []:
            daily |= "PerDay" in str(violation.get("quotaId", ""))
    said = re.search(r"retry in ([\d.]+)\s*s", message or "")
    if retry_after is None and said:
        retry_after = float(said.group(1))
    return retry_after, daily


def to_gemini_contents(messages: Sequence[Message]):
    from google.genai import types

    return [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part.from_text(text=m.content)],
        )
        for m in messages
    ]


class FakeBackend:
    """Offline backend for tests and for working without an API key.

    `reply` decides the text for each request; by default it returns "{}" for
    JSON requests and a short placeholder otherwise. Token counts are word
    counts, so tests can predict them exactly.
    """

    provider = "fake"

    def __init__(self, reply: Optional[Callable[[Request], str]] = None):
        self._reply = reply or _default_fake_reply
        self.requests: list[Request] = []

    def generate(self, request: Request) -> BackendResponse:
        self.requests.append(request)
        text = self._reply(request)
        prompt = " ".join([request.system or ""] + [m.content for m in request.messages])
        return BackendResponse(
            text=text,
            finish_reason="STOP",
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            thought_tokens=0,
        )


def _default_fake_reply(request: Request) -> str:
    if request.json_schema is not None:
        return json.dumps({})
    return f"[fake reply to: {request.messages[-1].content[:40]}]"


def make_backend(config: LLMConfig) -> Backend:
    if config.provider == "gemini":
        return GeminiBackend(config)
    return FakeBackend()
