# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Snapshot

- **App name**: Llestrade (forensic psych report drafter).
- **Tech stack**: Python 3.12+, PySide6/Qt, uv for dependency management, pytest/pytest-qt for tests.
- **Entry point**: `main.py` → imports `src.app.run()` which builds the Qt application.
- **Behavior baseline**: `docs/current_behavior.md` is the concise source for current runtime behavior.
- **Data directories**: runtime artefacts live under `var/` (logs, test output), user workspaces live under `~/Documents/llestrade/`.

## Current Package Layout

```
src/
├── app/
│   ├── __init__.py        # Public run()/ProjectWorkspace factory
│   ├── core/              # Project/session domain logic, metrics, file tracker
│   ├── ui/
│   │   ├── dialogs/       # Reusable Qt dialogs
│   │   ├── stages/        # Top-level Qt widgets (MainWindow, ProjectWorkspace shell)
│   │   └── workspace/
│   │       ├── bulk_tab.py / highlights_tab.py / reports_tab.py   # Tab widgets
│   │       ├── controllers/                                      # Tab orchestration & state
│   │       └── services/                                         # Worker orchestration helpers
│   ├── workers/            # QRunnable-based background jobs
│   └── resources/          # Bundled prompts/templates
├── common/llm/             # LLM provider abstractions
├── config/                 # App configuration + logging setup
└── core/                   # Shared utilities (PDF, ingest, etc.)
```

Key idea: UI stages stay thin; each workspace tab has a controller/service pair. Business logic lives in `src/app/core/`; workers run via `src/app/workers/`.

## Everyday Commands

```bash
# Install dependencies (recommended)
uv sync

# Run the app (default workspace UI)
uv run main.py

# Run targeted tests (headless Qt)
QT_QPA_PLATFORM=offscreen scripts/run_pytest.sh tests/app/ui/test_workspace_bulk_analysis.py

# Run the whole suite (requires provider credentials for Gemini/Azure tests)
QT_QPA_PLATFORM=offscreen scripts/run_pytest.sh
```

## Coding Notes

- Follow PEP 8 + type hints; keep Qt signals uppercase with underscores.
- Controllers/services should own orchestration and worker lifecycles; Qt widgets remain presentational.
- Tests belong under `tests/app/...` mirroring the source tree; prefer dependency-injected stubs over monkeypatching private state.
- Runtime artefacts (`var/logs`, `var/test_output`) are gitignored—keep outputs there.

## When Adding Features

1. Prefer extending `src/app/ui/workspace/controllers/` or `src/app/core/` rather than bloating widget classes.
2. Expose new worker entry points via services under `src/app/ui/workspace/services/`.
3. Update README/AGENTS.md if the top-level layout changes.
4. Run the relevant pytest targets (`tests/app/ui/...`, `tests/app/core/...`) before opening a PR and summarise the executed commands.
