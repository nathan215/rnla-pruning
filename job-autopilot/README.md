# Job Autopilot

Personal job-application autopilot: discover postings, rank them against my profile,
generate a tailored resume for each, apply with a single click, and track every
application's status in one place.

> **Note:** This folder is a temporary home. It is meant to be moved into its own
> private repo (`nathan215/job-autopilot`) — see *Migration* at the bottom.

## Goal

One sitting, one screen: a queue of pre-scored jobs, each with a tailored resume
already generated. I review, click **Apply**, and the tool fills the form, submits,
and logs the application. Afterwards it keeps the status board up to date and drafts
LinkedIn outreach messages I can send manually.

## Design principles

1. **Human-in-the-loop by design.** Fully automated "spray and pray" bots (AIHawk-style)
   get accounts flagged and produce low-quality applications. Every submission here is
   triggered by an explicit click; the automation does the *tedious* part (form filling,
   resume tailoring, logging), not the *judgment* part.
2. **Prefer direct ATS forms** (Greenhouse, Lever, Ashby, Workable) over LinkedIn Easy
   Apply. They are stable, unauthenticated, and not against anyone's ToS to fill via
   browser automation on your own behalf.
3. **LinkedIn is read-mostly.** Scraping LinkedIn aggressively or auto-sending
   connection requests/messages violates LinkedIn's User Agreement and is the #1 way to
   get restricted. The tool *drafts* outreach messages; sending them stays manual.
4. **Local-first.** SQLite for state, YAML for the master profile, everything on disk.

## Architecture

```
                 ┌─────────────┐
  JobSpy /  ───▶ │  discover   │  scrape boards → normalize → dedupe
  board URLs     └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   score     │  LLM/embedding match vs. master profile → shortlist
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   tailor    │  master resume (YAML) + job description
                 └──────┬──────┘  → tailored resume (PDF) + cover letter draft
                        ▼
                 ┌─────────────┐
   ME ──click──▶ │   apply     │  Playwright fills the ATS form, I confirm submit
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   track     │  SQLite status board; later: Gmail polling for
                 └─────────────┘  "interview / rejection" emails; outreach drafts
```

Modules (`src/jobpilot/`):

| Module       | Status  | Responsibility                                          |
|--------------|---------|---------------------------------------------------------|
| `models.py`  | working | `Application` model, status lifecycle, SQLite store      |
| `cli.py`     | skeleton| `jobpilot scan / review / tailor / apply / status`       |
| `sources.py` | stub    | JobSpy-based discovery + normalization                   |
| `tailor.py`  | stub    | LLM resume tailoring from `master_resume.yaml` → PDF     |
| `applier.py` | stub    | Playwright form-filling per ATS (Greenhouse first)       |

## Roadmap

- **Phase 1 — Tracker (useful on day 1):** SQLite store + `jobpilot status` board;
  add applications manually or from a URL.
- **Phase 2 — Discovery:** JobSpy integration, dedupe, LLM scoring against profile,
  daily `scan` producing a shortlist.
- **Phase 3 — Tailoring:** master resume in YAML, per-job resume rendering
  (Jinja2 → HTML → PDF), keyword coverage check (Resume-Matcher-style).
- **Phase 4 — One-click apply:** Playwright automation for Greenhouse/Lever/Ashby;
  review screen → click → submit → auto-log.
- **Phase 5 — Status intelligence & outreach:** Gmail polling to auto-advance statuses
  (apply-potato-style), LinkedIn message drafts per application.

## Prior art

See [`docs/similar-projects.md`](docs/similar-projects.md) for the survey of existing
open-source projects (AIHawk, JobSpy, Resume-Matcher, jobsync, jobtrail, …), what each
does well, and what this project borrows vs. does differently.

## Setup (once code lands)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml   # then edit with your profile
jobpilot status
```

## Migration to its own repo

1. Create the private repo: <https://github.com/new> → name `job-autopilot`,
   Private, no README.
2. Then either ask Claude to push this folder there, or locally:
   ```bash
   git clone git@github.com:nathan215/rnla-pruning.git --branch claude/job-application-automation-l8rnkg tmp
   cd tmp/job-autopilot
   git init && git add -A && git commit -m "Initial scaffold"
   git remote add origin git@github.com:nathan215/job-autopilot.git
   git push -u origin main
   ```
