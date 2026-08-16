"""CLI entry points. Phase 1: manual tracking works; later phases fill in the rest."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from .models import Application, Status, Store

app = typer.Typer(help="Job Autopilot", no_args_is_help=True)

CONFIG_PATH = Path("config.yaml")


def _store() -> Store:
    db = "data/jobpilot.sqlite3"
    if CONFIG_PATH.exists():
        cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        db = cfg.get("tracking", {}).get("db_path", db)
    return Store(db)


@app.command()
def add(company: str, title: str, url: str, notes: str = ""):
    """Track a job you found manually."""
    a = _store().add(Application(company=company, title=title, url=url, notes=notes))
    typer.echo(f"#{a.id} {a.company} — {a.title} [{a.status}]")


@app.command()
def status():
    """Show the board: everything in flight, grouped by status."""
    store = _store()
    for s in Status:
        rows = store.list(s)
        if rows:
            typer.secho(f"\n{s.upper()} ({len(rows)})", bold=True)
            for a in rows:
                typer.echo(f"  #{a.id:<4} {a.company:<24} {a.title}")


@app.command()
def mark(app_id: int, new_status: Status, note: str = ""):
    """Advance an application, e.g. `jobpilot mark 3 interview`."""
    _store().set_status(app_id, new_status, note)
    typer.echo(f"#{app_id} → {new_status}")


@app.command()
def scan():
    """Phase 2: pull new postings via JobSpy and score them."""
    raise typer.Exit("Not implemented yet — see README roadmap (Phase 2).")


@app.command()
def tailor(app_id: int):
    """Phase 3: generate a tailored resume + cover letter for one application."""
    raise typer.Exit("Not implemented yet — see README roadmap (Phase 3).")


@app.command()
def apply(app_id: int):
    """Phase 4: open the review screen, then fill and submit on confirm."""
    raise typer.Exit("Not implemented yet — see README roadmap (Phase 4).")


if __name__ == "__main__":
    app()
