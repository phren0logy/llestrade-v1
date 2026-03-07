# Development Progress

This document tracks current dashboard-era milestones for Llestrade.

For historical transition notes (legacy UI split/new-ui migration plans), see `docs/archive/progress_2025-07_transition_history.md`.

## 2025-07-05 - Dashboard Foundations Landed
- Replaced the legacy launcher path with the dashboard entry point in `src/app/main_window.py`.
- Finished the workspace shell in `src/app/ui/stages/project_workspace.py` with Documents, Highlights, Bulk Analysis, and Reports tabs.
- Implemented `FileTracker`, bulk analysis group persistence, and updated `ProjectManager` dashboard metrics handling.
- Consolidated worker execution around `QThreadPool` + coordinator patterns in `src/app/workers/`.
- Added dashboard-focused pytest coverage under `tests/app/**`.

## Current Reality Check
- Dashboard workflow is the active product path launched from `main.py`.
- Legacy and migration planning docs are retained under `docs/archive/` for reference only.
- `docs/work_plan.md` remains the source of truth for active implementation priorities.
