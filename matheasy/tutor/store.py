"""Student stores (Role C): what the tutor remembers about each student across sessions.

    InMemoryStore    memory for one process (tests, demos)
    JsonStudentStore the same, saved to a JSON file after every change, so a
                     student's memory survives between sessions and runs
    NoMemoryStore    remembers nothing: for B0 and B1, which have no memory

B's diagnose() reads misconception_prior(); the planner (week 3) will
replace the fixed prior with one learned from each student's history.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .state import StudentMemory

DIAGNOSED_BEFORE_PRIOR = 0.8  # until the planner: a misconception seen before is likely still there


class InMemoryStore:
    def __init__(self):
        self._memories: dict[str, StudentMemory] = {}

    def load(self, student_id: str) -> StudentMemory:
        return self._memories.setdefault(student_id, StudentMemory(student_id))

    def record(self, memory: StudentMemory, misconception_id: str) -> None:
        memory.times_diagnosed[misconception_id] = memory.times_diagnosed.get(misconception_id, 0) + 1

    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]:
        memory = self._memories.get(student_id)
        return DIAGNOSED_BEFORE_PRIOR if memory and memory.times_diagnosed.get(misconception_id) else None


class JsonStudentStore(InMemoryStore):
    def __init__(self, path: Path | str):
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            for sid, data in json.loads(self.path.read_text()).items():
                self._memories[sid] = StudentMemory(sid, dict(data["times_diagnosed"]))

    def record(self, memory: StudentMemory, misconception_id: str) -> None:
        super().record(memory, misconception_id)
        self._memories[memory.student_id] = memory
        self._save()

    def _save(self) -> None:
        data = {sid: {"times_diagnosed": m.times_diagnosed} for sid, m in sorted(self._memories.items())}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)


class NoMemoryStore:
    def load(self, student_id: str) -> StudentMemory:
        return StudentMemory(student_id)

    def record(self, memory: StudentMemory, misconception_id: str) -> None:
        pass

    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]:
        return None
