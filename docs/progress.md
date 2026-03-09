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

## 2026-03-08 - Gateway-First LLM Plan Adopted
- Confirmed a Gateway-first direction for Pydantic AI integration while retaining legacy provider fallback during rollout.
- Completed contract groundwork for backend interchangeability:
  - Added typed worker stage contracts for report/bulk execution.
  - Added `LLMExecutionBackend` + legacy adapter, and routed worker invocation through the backend seam.
  - Added focused tests for backend contracts and stage-to-trace mapping.
- Updated `docs/work_plan.md` with explicit milestones for report pilot, bulk expansion, cutover gates, and Qt 6.11 parallel validation.
