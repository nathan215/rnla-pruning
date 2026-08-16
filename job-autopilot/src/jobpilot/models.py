"""Application model, status lifecycle, and SQLite store.

This is the spine of the tool: every other module reads/writes Applications here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path


class Status(StrEnum):
    DISCOVERED = "discovered"      # scraped, not yet reviewed
    SHORTLISTED = "shortlisted"    # passed scoring / manual review
    TAILORED = "tailored"          # resume + cover letter generated
    APPLIED = "applied"            # submitted
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    GHOSTED = "ghosted"            # no response after N weeks


@dataclass
class Application:
    company: str
    title: str
    url: str
    status: Status = Status.DISCOVERED
    source: str = "manual"          # jobspy site name or "manual"
    location: str = ""
    description: str = ""
    score: float | None = None      # match score vs. master profile, 0..1
    resume_path: str = ""           # tailored PDF, once generated
    notes: str = ""
    id: int | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    location TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    score REAL,
    resume_path TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id),
    status TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    at TEXT NOT NULL
);
"""


class Store:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def add(self, app: Application) -> Application:
        """Insert if the URL is new; return the stored row either way."""
        cur = self._conn.execute(
            """INSERT INTO applications
               (company, title, url, status, source, location, description,
                score, resume_path, notes, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(url) DO NOTHING""",
            (app.company, app.title, app.url, app.status, app.source,
             app.location, app.description, app.score, app.resume_path,
             app.notes, app.created_at),
        )
        # rowcount, not lastrowid: on a skipped conflict, lastrowid still holds
        # the connection's previous insert id.
        if cur.rowcount == 1:
            app.id = cur.lastrowid
            self._log_event(app.id, app.status, "created")
        else:
            row = self._conn.execute(
                "SELECT * FROM applications WHERE url = ?", (app.url,)
            ).fetchone()
            app = _from_row(row)
        self._conn.commit()
        return app

    def set_status(self, app_id: int, status: Status, note: str = "") -> None:
        self._conn.execute(
            "UPDATE applications SET status = ? WHERE id = ?", (status, app_id)
        )
        self._log_event(app_id, status, note)
        self._conn.commit()

    def list(self, status: Status | None = None) -> list[Application]:
        q = "SELECT * FROM applications"
        params: tuple = ()
        if status is not None:
            q += " WHERE status = ?"
            params = (status,)
        q += " ORDER BY created_at DESC"
        return [_from_row(r) for r in self._conn.execute(q, params)]

    def history(self, app_id: int) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT status, note, at FROM events WHERE application_id = ? ORDER BY at",
            (app_id,),
        ))

    def _log_event(self, app_id: int, status: Status, note: str) -> None:
        self._conn.execute(
            "INSERT INTO events (application_id, status, note, at) VALUES (?,?,?,?)",
            (app_id, status, note, datetime.now(timezone.utc).isoformat()),
        )


def _from_row(row: sqlite3.Row) -> Application:
    return Application(
        id=row["id"], company=row["company"], title=row["title"], url=row["url"],
        status=Status(row["status"]), source=row["source"], location=row["location"],
        description=row["description"], score=row["score"],
        resume_path=row["resume_path"], notes=row["notes"],
        created_at=row["created_at"],
    )
