# Current Intended Behavior (Dashboard)

This document is the concise behavior baseline for cleanup, testing, and future refactors.

## Product Flow
- Launch with `uv run main.py` (or `uv run -m src.app`).
- Create/open a project from the welcome stage.
- Use Documents tab to manage source tree selection and conversion to markdown.
- Use Highlights tab to extract highlights from PDF-derived files.
- Use Bulk Analysis tab to run grouped map/combined bulk analysis jobs.
- Use Reports tab to generate and refine report drafts from project inputs.
- There is no standalone Progress tab; Bulk Analysis inline logs are the canonical activity surface.

## File-System Expectations
- Project data lives in project directory (`project.frpd`, `sources.json`).
- Converted markdown lives in `converted_documents/` and mirrors source folder structure.
- Highlight outputs live in `highlights/` and mirror converted paths using `.highlights.md`.
- Bulk analysis outputs live under `bulk_analysis/<group>/`.
- Citation/evidence metadata is stored in `.llestrade/citations.db` per project.

## Metrics and Counts
- Highlights denominator is PDF-only.
- Placeholder `.highlights.md` files are written when PDFs have no highlights.
- Bulk and report prompt runs use project placeholder mapping plus runtime context placeholders.
- Citation-aware bulk/report prompts receive a generated system appendix that defines the allowed local citation labels for that run.
- Generated analyses/reports use inline local citation markers in the format `[C1]`, `[C2]`, etc.
- When the citation pipeline changes incompatibly, derived artifacts (`converted_documents/`, `highlights/`, `bulk_analysis/`, `reports/`, `.llestrade/citations.db`) are expected to be reset and rebuilt.

## Observability Status
- Observability is OTEL-first, with Local Phoenix supported as one export target.
- Worker-stage spans and Pydantic AI model instrumentation are configured separately but run through the same observability runtime.

## Scope Note
- Historical migration and legacy UI plans are archived under `docs/archive/`.
- Active implementation priorities remain in `docs/work_plan.md`.
