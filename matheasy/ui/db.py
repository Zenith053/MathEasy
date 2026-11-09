"""The analysis page's database: accounts, sign-in sessions, every attempt, and each student's memory.

One SQLite file, runs/ui/matheasy.db (git-ignored). Passwords are never stored: each gets a
random salt and an scrypt hash. A session is a random token in a cookie, valid until the
student signs out or the browser closes.

    python -m matheasy.ui.db add-user anna                    # a student; asks for the password
    python -m matheasy.ui.db add-user mr_khan --role teacher
    python -m matheasy.ui.db list

Roles: a student sees only their own working, chart and history; a teacher sees every student's.
The attempts table is the history slip detection needs: what was asked, what the student wrote,
whether it was right, and what the diagnosis said.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import secrets
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..tutor.state import StudentMemory

DB = Path(__file__).resolve().parents[2] / "runs" / "ui" / "matheasy.db"
ROLES = ("student", "teacher")
SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, salt BLOB NOT NULL, hash BLOB NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('student', 'teacher')), created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY, username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    question TEXT NOT NULL, working TEXT NOT NULL, correct INTEGER NOT NULL, slip INTEGER NOT NULL DEFAULT 0,
    wrong_line INTEGER, step TEXT, status TEXT, misconception TEXT, topics TEXT NOT NULL DEFAULT '[]',
    created TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS attempts_by_user ON attempts(username, id);
CREATE TABLE IF NOT EXISTS memory (
    username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE, data TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **SCRYPT)


class Database:
    def __init__(self, path: Path | str = DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------ accounts
    def add_user(self, username: str, password: str, role: str = "student") -> None:
        username = username.strip()
        if not username or len(username) > 40:
            raise ValueError("a username has 1 to 40 characters")
        if len(password) < 6:
            raise ValueError("a password has at least 6 characters")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        salt = secrets.token_bytes(16)
        try:
            self.conn.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (username, salt, _hash(password, salt), role, _now()))
        except sqlite3.IntegrityError:
            raise ValueError(f"the username {username!r} is taken") from None

    def check(self, username: str, password: str) -> Optional[str]:
        """The user's role if the password is right, else None."""
        row = self.conn.execute("SELECT salt, hash, role FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            _hash(password, secrets.token_bytes(16))  # same work either way, so timing does not reveal usernames
            return None
        return row["role"] if hmac.compare_digest(_hash(password, row["salt"]), row["hash"]) else None

    def role(self, username: str) -> Optional[str]:
        row = self.conn.execute("SELECT role FROM users WHERE username = ?", (username,)).fetchone()
        return row["role"] if row else None

    def students(self) -> list[str]:
        return [r["username"] for r in self.conn.execute("SELECT username FROM users WHERE role = 'student' ORDER BY username")]

    def users(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT username, role, created FROM users ORDER BY role, username")]

    # ------------------------------------------------------------ sessions
    def start_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        self.conn.execute("INSERT INTO sessions VALUES (?, ?, ?)", (token, username, _now()))
        return token

    def session_user(self, token: Optional[str]) -> Optional[tuple[str, str]]:
        """(username, role) for a valid token, else None."""
        if not token:
            return None
        row = self.conn.execute("SELECT u.username, u.role FROM sessions s JOIN users u USING (username) "
                                "WHERE s.token = ?", (token,)).fetchone()
        return (row["username"], row["role"]) if row else None

    def end_session(self, token: Optional[str]) -> None:
        if token:
            self.conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    # ------------------------------------------------------------ history and memory
    def add_attempt(self, username: str, question: str, working: str, correct: bool, *, slip: bool = False,
                    wrong_line: Optional[int] = None, step: Optional[str] = None, status: Optional[str] = None,
                    misconception: Optional[str] = None, topics=()) -> None:
        self.conn.execute(
            "INSERT INTO attempts (username, question, working, correct, slip, wrong_line, step, status, misconception, "
            "topics, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (username, question, working, int(correct), int(slip), wrong_line, step, status, misconception,
             json.dumps(list(topics)), _now()))

    def attempts(self, username: str, limit: int = 30) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM attempts WHERE username = ? ORDER BY id DESC LIMIT ?", (username, limit))
        return [{**dict(r), "correct": bool(r["correct"]), "slip": bool(r["slip"]), "topics": json.loads(r["topics"])}
                for r in rows]

    def load_memory(self, username: str) -> StudentMemory:
        row = self.conn.execute("SELECT data FROM memory WHERE username = ?", (username,)).fetchone()
        return StudentMemory(username, **json.loads(row["data"])) if row else StudentMemory(username)

    def save_memory(self, memory: StudentMemory) -> None:
        self.conn.execute("INSERT INTO memory VALUES (?, ?) ON CONFLICT(username) DO UPDATE SET data = excluded.data",
                          (memory.student_id, json.dumps(memory.to_dict())))

    def clear_student(self, username: str) -> None:
        self.conn.execute("DELETE FROM memory WHERE username = ?", (username,))
        self.conn.execute("DELETE FROM attempts WHERE username = ?", (username,))


def main() -> None:
    ap = argparse.ArgumentParser(description="Accounts for the analysis page.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add-user", help="create an account; asks for the password")
    add.add_argument("username")
    add.add_argument("--role", choices=ROLES, default="student")
    sub.add_parser("list", help="list the accounts")
    args = ap.parse_args()
    db = Database()
    if args.cmd == "list":
        for u in db.users():
            print(f"{u['role']:8} {u['username']:24} created {u['created']}")
        return
    password = getpass.getpass(f"Password for {args.username}: ")
    if password != getpass.getpass("Again: "):
        sys.exit("the passwords differ")
    try:
        db.add_user(args.username, password, args.role)
    except ValueError as e:
        sys.exit(str(e))
    print(f"created {args.role} {args.username}")


if __name__ == "__main__":
    main()
