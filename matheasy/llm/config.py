"""Load and validate the shared LLM config (config/llm.yaml)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "llm.yaml"

PROVIDERS = ("gemini", "fake")


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    models: dict[str, str]
    temperature: float
    max_output_tokens: int
    seed: Optional[int]
    thinking_budget: Optional[int]
    session_tokens: int
    retry_attempts: int
    retry_initial_delay_s: float
    retry_max_delay_s: float
    timeout_s: float
    cache_enabled: bool
    cache_dir: Path
    log_dir: Path
    log_include_text: bool
    # Short fingerprint of the whole config, written into every log line.
    hash: str

    def model(self, name: str = "backbone") -> str:
        if name not in self.models:
            raise KeyError(f"no model named {name!r} in config; have {sorted(self.models)}")
        return self.models[name]


def load_config(path: Optional[Path | str] = None) -> LLMConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    try:
        return _from_dict(raw)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"invalid LLM config {path}: {e}") from e


def _from_dict(raw: dict) -> LLMConfig:
    gen, budget, retry = raw["generation"], raw["budget"], raw["retry"]
    cfg = LLMConfig(
        provider=raw["provider"],
        models=dict(raw["models"]),
        temperature=float(gen["temperature"]),
        max_output_tokens=int(gen["max_output_tokens"]),
        seed=gen.get("seed"),
        thinking_budget=gen.get("thinking_budget"),
        session_tokens=int(budget["session_tokens"]),
        retry_attempts=int(retry["attempts"]),
        retry_initial_delay_s=float(retry["initial_delay_s"]),
        retry_max_delay_s=float(retry["max_delay_s"]),
        timeout_s=float(raw["timeout_s"]),
        cache_enabled=bool(raw["cache"]["enabled"]),
        cache_dir=REPO_ROOT / raw["cache"]["dir"],
        log_dir=REPO_ROOT / raw["logging"]["dir"],
        log_include_text=bool(raw["logging"]["include_text"]),
        hash=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12],
    )
    if cfg.provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {cfg.provider!r}")
    if "backbone" not in cfg.models:
        raise ValueError("models.backbone is required")
    for name, model_id in cfg.models.items():
        if "latest" in model_id or "preview" in model_id:
            raise ValueError(f"models.{name} = {model_id!r}: pin a stable model ID")
    if cfg.max_output_tokens <= 0 or cfg.session_tokens <= 0:
        raise ValueError("max_output_tokens and session_tokens must be positive")
    if cfg.retry_attempts < 1:
        raise ValueError("retry.attempts must be at least 1")
    return cfg
