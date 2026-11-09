"""Student stores (Role C): what the tutor remembers about each student across sessions.

    InMemoryStore    memory for one process (tests, demos)
    JsonStudentStore the same, saved to a JSON file, so a student's memory
                     survives between sessions and runs
    NoMemoryStore    remembers nothing: for B0 and B1, which have no memory

The planner (planner.py) does the updating. B's diagnose() reads
misconception_prior(): the planner's P(holds misconception), or None for a
misconception never seen in this student.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

from . import planner
from .state import Problem, StudentMemory


class InMemoryStore:
    def __init__(self):
        self._memories: dict[str, StudentMemory] = {}

    def load(self, student_id: str) -> StudentMemory:
        return self._memories.setdefault(student_id, StudentMemory(student_id))

    def begin_session(self, memory: StudentMemory, session_index: int) -> None:
        planner.begin_session(memory, session_index)

    def record(self, memory: StudentMemory, misconception_id: str, topics: Sequence[str] = ()) -> None:
        planner.diagnosed(memory, misconception_id, topics)

    def observe(self, memory: StudentMemory, problem: Problem, answer: str, correct: bool,
                topics: Sequence[str] = (), slip: bool = False) -> None:
        planner.answered(memory, problem, answer, correct, topics, slip)

    def end_session(self, memory: StudentMemory) -> None:
        planner.end_session(memory)

    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]:
        memory = self._memories.get(student_id)
        return memory.p_holds.get(misconception_id) if memory else None


class JsonStudentStore(InMemoryStore):
    """Saved after every change (autosave), or only when save() is called: the
    experiment runner saves once per finished session, so a crash mid-session
    leaves the memory as it was before that session."""

    def __init__(self, path: Path | str, *, autosave: bool = True):
        super().__init__()
        self.path = Path(path)
        self.autosave = autosave
        if self.path.exists():
            for sid, data in json.loads(self.path.read_text()).items():
                self._memories[sid] = StudentMemory(sid, **data)

    def begin_session(self, memory: StudentMemory, session_index: int) -> None:
        super().begin_session(memory, session_index)
        self._changed(memory)

    def record(self, memory: StudentMemory, misconception_id: str, topics: Sequence[str] = ()) -> None:
        super().record(memory, misconception_id, topics)
        self._changed(memory)

    def observe(self, memory: StudentMemory, problem: Problem, answer: str, correct: bool,
                topics: Sequence[str] = (), slip: bool = False) -> None:
        super().observe(memory, problem, answer, correct, topics, slip)
        self._changed(memory)

    def end_session(self, memory: StudentMemory) -> None:
        super().end_session(memory)
        self._changed(memory)

    def _changed(self, memory: StudentMemory) -> None:
        self._memories[memory.student_id] = memory
        if self.autosave:
            self.save()

    def save(self) -> None:
        data = {sid: m.to_dict() for sid, m in sorted(self._memories.items())}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)


class NoMemoryStore:
    def load(self, student_id: str) -> StudentMemory:
        return StudentMemory(student_id)

    def begin_session(self, memory: StudentMemory, session_index: int) -> None:
        pass

    def record(self, memory: StudentMemory, misconception_id: str, topics: Sequence[str] = ()) -> None:
        pass

    def observe(self, memory: StudentMemory, problem: Problem, answer: str, correct: bool,
                topics: Sequence[str] = (), slip: bool = False) -> None:
        pass

    def end_session(self, memory: StudentMemory) -> None:
        pass

    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]:
        return None
