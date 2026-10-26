"""Run log (Role C) in the format D's checker requires (matheasy/evaluation/log_fields.py).

One JSONL line per event. The orchestrator writes `diagnosis` and `hint`; the
session runner writes `session_start`, `turn`, `test` and `session_end`. Every
line carries run_id and session_id, shared with the LLM call log, and the LLM
config hash, so D can check all shards ran on the same settings.

    python -m matheasy.evaluation runs/turn_logs     # D's checker
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .state import TutorState

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "runs" / "turn_logs"


class TurnLogger:
    def __init__(self, run_id: str, config_hash: str, log_dir: Optional[Path] = None):
        self.run_id = run_id
        self.config_hash = config_hash
        self.path = Path(log_dir or DEFAULT_DIR) / f"{run_id}.jsonl"
        self._lock = threading.Lock()

    def log(self, state: TutorState, event: str, **fields) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "session_id": state.session_id,
            "student_id": state.student_id,
            "condition": state.condition,
            "config_hash": self.config_hash,
            "event": event,
            "problem_id": state.problem.id if state.problem else None,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
