# Similar open-source projects (survey, Aug 2026)

Short version: every piece of Job Autopilot already exists somewhere, but no project
combines **curated one-click apply + per-job tailored resume + status tracking +
outreach drafts** in one local-first tool. The closest is AIHawk, which is fully
automated (no human in the loop), LinkedIn-centric, and heavy.

## Auto-apply bots

| Project | Notes |
|---|---|
| [feder-cr/Jobs_Applier_AI_Agent_AIHawk](https://github.com/feder-cr/jobs_applier_ai_agent_aihawk) | The flagship (featured in TechCrunch/Wired/The Verge). Python; auto-applies with an AI-tailored resume + cover letter per posting. Provider plugins were removed from the repo for copyright reasons; huge fork ecosystem ([us/linkedIn_auto_jobs_applier_with_AI_fast](https://github.com/us/linkedIn_auto_jobs_applier_with_AI_fast), [OnlineGBC/Jobs_Applier_AI_Agent](https://github.com/OnlineGBC/Jobs_Applier_AI_Agent), …). **Borrow:** per-job resume generation flow. **Differ:** it's fire-and-forget; we want review-then-click. |
| LinkedIn "Easy Apply" bot variants | Dozens of Selenium bots that mass-click Easy Apply. High ban risk, low application quality. **Differ:** we avoid this category entirely. |

## Discovery / scraping

| Project | Notes |
|---|---|
| [speedyapply/JobSpy](https://github.com/speedyapply/JobSpy) ([PyPI: python-jobspy](https://pypi.org/project/python-jobspy/)) | Concurrent scraper for LinkedIn, Indeed, Glassdoor, Google, ZipRecruiter → pandas DataFrame. Indeed is the most reliable (no rate limit); LinkedIn rate-limits around page 10 per IP; ~1000 jobs cap per search. **Borrow:** use as the whole discovery layer instead of writing scrapers. |

## Resume tailoring / ATS matching

| Project | Notes |
|---|---|
| [srbhr/Resume-Matcher](https://github.com/srbhr/Resume-Matcher) | Open-source ATS-style matcher: keyword extraction from JD, vector similarity vs. resume, works locally with many LLMs. **Borrow:** keyword-coverage scoring to validate each tailored resume before applying. |
| [sunnypatell/ats-screener](https://github.com/sunnypatell/ats-screener) | Simulates how 6 enterprise ATS platforms (Workday, Taleo, iCIMS, Greenhouse, Lever, SuccessFactors) parse and score a resume. **Borrow:** use as a test harness for our PDF output. |

## Trackers

| Project | Notes |
|---|---|
| [Gsync/jobsync](https://github.com/Gsync/jobsync) | Self-hosted tracker + AI career assistant (resume review, job matching, analytics); LLM can run locally via Ollama. **Borrow:** status/analytics data model ideas. |
| [kaylaehman/jobtrail](https://github.com/kaylaehman/jobtrail) | Self-hosted tracker with a JobSpy-powered discovery pipeline and keyless company enrichment (Wikipedia/Wikidata/SEC EDGAR); `docker compose up`, no external LLM calls. **Borrow:** proof that JobSpy → tracker pipeline works; enrichment idea. |
| [coolbrother/apply-potato](https://github.com/coolbrother/apply-potato) | Scrapes GitHub job lists, filters by eligibility, **monitors application status via Gmail**, syncs to Google Sheets. **Borrow:** Gmail polling for automatic status transitions (Phase 5). |
| [GitHub topic: job-application-tracker](https://github.com/topics/job-application-tracker) | Long tail of trackers; nothing dominant. |

## Gap this project fills

1. **One-click, not zero-click:** every existing applier is fully automated; recruiters
   and platforms increasingly detect/penalize that. A curated queue with human confirm
   is both safer and produces better applications.
2. **ATS-direct:** most bots target LinkedIn Easy Apply; Greenhouse/Lever/Ashby forms
   are more automation-friendly and lower risk.
3. **End-to-end state:** discovery → tailored artifact → submission → email-driven
   status updates in one SQLite database, instead of 3 separate tools + a spreadsheet.

## Compliance notes

- LinkedIn's User Agreement prohibits scraping and automated messaging/connections;
  accounts get restricted for it. Keep LinkedIn interactions manual (tool drafts text,
  you send). JobSpy's LinkedIn scraping is rate-limited and best used sparingly or
  skipped in favor of Indeed/Google.
- Auto-filling an ATS form with your own truthful data, with you confirming each
  submission, is the low-risk zone this project stays in.
