# Repository Guidelines

## Project Structure & Module Organization
`main.py` launches the dashboard UI in `src/app/`. The package is organised as:
`src/app/core/` (project/session domain logic), `src/app/ui/stages/` (top-level Qt widgets), `src/app/ui/workspace/` (tab widgets plus `controllers/` + `services/` for tab orchestration), `src/app/workers/` (QRunnable jobs), and `src/app/resources/` (prompts/templates). Shared helpers remain in `src/config/`, `src/core/`, and `src/common/llm/`; scripts live in `scripts/`, tests in `tests/`, and artefacts in `var/test_output/` and `var/logs/`.

## Build, Run, and Development Commands
Install dependencies with `uv sync`; this repo targets Python 3.12+. Launch the dashboard via `uv run main.py` (or `uv run -m src.app`). `./run_debug.sh` or `DEBUG=true uv run main.py` surface instrumentation; `uv run scripts/setup_env.py` verifies credentials.

## Coding Style & Naming Conventions
Python modules follow snake_case filenames with 4-space indentation and type hints. Classes use CapWords, Qt signals stay uppercase with underscores, and worker classes end with `Worker`. Dashboard modules should stay small (~400 lines) with dataclasses in `src/app/core/`. Keep configuration keys lowercase, align prompts to the `topic_action.md` pattern, and document public functions as needed. Run `uv run pytest tests/` before committing.

## Testing Guidelines
Tests rely on `pytest` and `pytest-qt`; name files `test_*.py` and mirror the source tree. Prioritize business-logic tests for `src/app/core/` classes and worker behaviors. Use fixtures for LLM stubs, store artefacts under `var/test_output/`, and run `uv run pytest --cov=. tests/` to track coverage. Capture before/after screenshots when UI flows change.

## Commit & Pull Request Guidelines
Commit messages follow the existing imperative, sentence-cased style (`Fix project creation workflow issues`). Reference relevant checklist items from `docs/work_plan.md` when advancing the dashboard refactor. Group logical changes per commit, summarize user impact, list executed commands, and include screenshots for UI tweaks. Call out env requirements or migrations and tag reviewers closest to the touched module.

## Configuration & Secrets
Copy `config.template.env` to `.env` and populate provider keys (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, Azure credentials). Avoid hardcoding secrets; use the keyring helper in `src/config`. Scrub PII from `logs/` before sharing and rotate credentials after exporting crash data.

## Dashboard Refactor Focus
Treat `docs/work_plan.md` as the source of truth. Priority 0 centers on shipping the dashboard workspace: land `FileTracker`, bulk analysis groups, and `QThreadPool` consolidation before touching legacy cleanup. Keep the project tree lean—`converted_documents/`, `highlights/`, `bulk_analysis/`, `reports/`, `templates/`, and `backups/` cover the current workflow, with bulk-analysis outputs living inside each group folder under `bulk_analysis/`.
Use `docs/current_behavior.md` for a concise statement of current runtime behavior when docs disagree.

### Highlights and Counts
- Highlight outputs live under `highlights/` and mirror `converted_documents/` paths with a `.highlights.md` suffix.
- Highlights are generated for PDFs only. When no highlights are found, a placeholder `.highlights.md` (with a processed timestamp) is written.
- FileTracker computes “pending highlights” from PDF-converted files only; DOCX and other non-PDF sources are excluded from the highlight denominator.
- UI denominators for highlights reflect PDF-only totals (i.e., `Highlights: X of Y` where `Y = pdf_total`).
