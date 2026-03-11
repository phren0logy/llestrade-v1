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
- Generated analyses/reports may include inline citation markers in the format `[CIT:ev_<id>]`.

## Observability Status
- Phoenix wiring and fixture export paths exist.
- Trace retrieval/export via `PhoenixObservability.get_traces()` is currently a stub path and should not be treated as a complete feature.

## Scope Note
- Historical migration and legacy UI plans are archived under `docs/archive/`.
- Active implementation priorities remain in `docs/work_plan.md`.
