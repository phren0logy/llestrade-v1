# Llestrade - Consolidated Work Plan (formerly Forensic Report Drafter)

## Priority 0: Dashboard UI Refactoring (IMMEDIATE)

Transform the current wizard-style UI into a dashboard-based workflow that supports long-running operations, multiple bulk analysis groups, and intelligent file existence checking for resume functionality.

### Design Principles

- **Keep it simple**: Use built-in Qt/PySide6 functionality where possible
- **File-based state**: All state visible as files/folders for debugging
- **Breaking changes OK**: This is pre-release software, no backward compatibility needed
- **No overengineering**: Simple solutions preferred
- **Progressive enhancement**: Build on existing working code

### Phase 1: Core Infrastructure & Dashboard (Remaining Work)

#### Legacy UI Reference (Captured 2025-03-11)

- StageManager’s linear gating is intentionally replaced by a scan-driven workflow after project setup.
- Project setup keeps evaluator/output guardrails, auto-sanitizes folder name (spaces → dashes), and informs users about folder creation.
- Source management pivots to folder-level include/exclude (tree checkboxes) with relative paths; root-level files trigger warnings instead of silent skips.
- Conversion helpers remain responsible for PDFs/complex formats; simple text/markdown bypass conversion and duplicates are prevented via in-memory tracking.
- Bulk analysis reuses legacy chunking/logging patterns but surfaces streamlined logs; reports move to a placeholder tab until redesigned.

#### Welcome Stage Stabilization

- [ ] Exercise the refreshed welcome stage (`src/app/ui/stages/welcome_stage.py`) with existing `.frpd` projects and resolve regressions uncovered by legacy project metadata.

#### Workspace Polish

- [ ] Default to the Bulk Analysis tab on open once project setup is complete and FileTracker shows converted folders (`src/app/ui/stages/project_workspace.py`).
- [ ] Keep the Progress tab officially descoped—document that the Bulk Analysis tab’s inline log feed is the canonical activity surface.

#### Automated Conversion Follow-ups

- [ ] After conversions finish, trigger bulk analysis for eligible groups (folders selected in both Documents and Bulk Analysis tabs) using existing chunking for large files.

### Phase 2: Bulk Analysis & Integration (Remaining Work)

#### Bulk Analysis Group Dialog (`src/app/ui/dialogs/bulk_analysis_group_dialog.py`)

- [ ] Add a duplicate-name warning/validation before saving a group configuration.

#### Bulk Workflow QA

- [ ] Test with multiple bulk groups and overlapping folder selections to ensure manifest skipping and prompt hashing behave correctly.

#### Worker Instrumentation & Observability

- [ ] Emit consistent debug logging (job ID + status transitions) across all `DashboardWorker` subclasses.
- [ ] Keep `observability.py` wiring current and add Phoenix spans that include bulk analysis group context.
- [ ] Export Phoenix fixtures for deterministic tests.
- [ ] Document that trace retrieval/export remains limited while `PhoenixObservability.get_traces()` is a stub path.

#### Business Logic Testing

- [ ] Test document conversion with folder preservation.
- [ ] Reuse/migrate legacy test scenarios where applicable into `tests/app/**`.
- [ ] Add mocked smoke tests driven by recorded fixtures (no live API calls).
- [ ] Maintain a separate, optional live suite for provider calls using dedicated credentials.
- [ ] Audit the codebase for “summary” terminology and replace it with “bulk analysis” where appropriate.

### Phase 3: Polish & Cleanup

#### Settings & UI Polish

- [ ] Add minimize-on-close setting.
- [ ] Test resume-after-restart functionality.
- [ ] Verify error handling and surface failures in the workspace footer panel.
- [ ] Implement the cost-tracking stub (even if the final UI ships later).

#### Code Cleanup & Documentation

- [ ] Clean settings/tests that still reference pre-dashboard classes or legacy-only code paths.
- [ ] Document the streamlined dashboard-first workflow in README/CLAUDE/AGENTS as future features land.

### Folder Structure After Refactoring

```
project_dir/
├── project.frpd                # Project metadata (relative source paths, helper, UI state)
├── sources.json                # Serialized tree of included folders (+ warnings for root files)
├── converted_documents/        # Markdown outputs mirroring selected folder structure
│   ├── medical_records/
│   │   ├── report1.md
│   │   └── report2.md
│   └── legal_docs/
│       └── case_summary.md
├── highlights/                 # Highlight outputs mirroring converted_documents (PDF-only)
│   ├── medical_records/
│   │   ├── report1.highlights.md
│   │   └── report2.highlights.md
│   └── legal_docs/
│       └── case_summary.highlights.md
├── bulk_analysis/
│   ├── clinical_records/
│   │   ├── config.json         # Prompts, model, folder subset
│   │   └── outputs/
│   │       ├── medical_records/
│   │       │   ├── report1.md
│   │       │   └── report2.md
│   │       └── legal_docs/
│   │           └── case_summary.md
│   └── legal_documents/
│       ├── config.json
│       └── outputs/
│           └── legal_docs/
│               └── case_summary.md
└── backups/
    └── 2025-01-01T120000Z/    # Snapshot copies created by the app
```

> Original source files remain in their external locations; the project stores only relative references and derived outputs. Highlight files exist only for PDFs and mirror the converted_documents tree. When no highlights are found, a placeholder `.highlights.md` is written with a timestamped note.

### Success Criteria for Priority 0

- [x] Users can create multiple bulk analysis groups (shared converted docs)
- [x] Files can belong to multiple groups without duplication
- [ ] Scan-on-open + manual re-scan convert new files with user confirmation
- [x] Dashboard shows accurate `X of Y` conversion and bulk-analysis counts with root-level warnings where needed
- [x] Converted/bulk output mirrors folder structure via relative paths
- [x] Worker operations remain non-blocking with safe shutdown
- [x] Breaking changes implemented cleanly
- [x] Business logic tests passing
- [ ] Phoenix tracing working for LLM calls with group context

- [x] Documentation: add `highlights/` to the folder structure diagram; clarify that highlight counts use PDFs only.
- [ ] Highlights UX: add "Re-extract highlights" action and surface reasons when a file remains pending.

Note: Highlights denominator uses PDFs only (pending/highlights reflect PDF-eligible files).

## Current Project Status

### Application Architecture

- **Technology**: PySide6 desktop application targeting macOS, Windows, and Linux.
- **UI Transition**:
  - **Legacy UI**: Archived; dashboard is now the default experience launched from `main.py`.
  - **Dashboard UI**: Tabbed workspace under `src/app/ui/stages/project_workspace.py` with Documents, Highlights, Bulk Analysis, and Reports tabs.

### Testing Status

- Dashboard-era tests live under `tests/app/**` (FileTracker, workspace controllers, worker coordinator, placeholder flows).
- Legacy UI regression tests remain for reference but are no longer expanded.
- Deterministic PR checks are automated in GitHub Actions (`tests-pr-deterministic.yml`) and publish coverage artifacts.
- Optional live-provider checks run in a separate workflow (`tests-live-providers.yml`) via manual dispatch or PR label.
- Local test entrypoints: `scripts/run_pytest.sh`, `scripts/run_pytest_pr.sh`, `scripts/run_pytest_live.sh`.

### Codebase Health

- **Minimal langchain usage** (only MarkdownHeaderTextSplitter) ✓
- **Standard file operations** (pathlib, shutil) ✓
- **Large files needing refactoring**:
  - `src/app/ui/workspace/controllers/documents.py` (~810 lines)
  - `src/app/ui/workspace/controllers/bulk.py` (~755 lines)
  - `src/app/ui/workspace/controllers/reports.py` (~1,400 lines)

## LLM Platform Plan (Gateway-First, 2026-03-08)

### Decision

- Pydantic AI Gateway is the target provider layer for new LLM execution work.
- Existing provider integrations remain available as fallback during rollout.
- Breaking changes are allowed while the app is pre-release.

### Execution Milestones

- [x] Foundation: decompose large controllers and introduce worker-stage contracts plus `LLMExecutionBackend`.
- [x] Report pilot: implement a report-only Pydantic AI Gateway backend behind a feature flag.
- [x] Bulk expansion: extend the Gateway backend to bulk map/reduce with parity on cancellation, checkpointing, and retry semantics.
- [x] Cutover: make Gateway the default path for report and bulk workflows after parity tests pass.
- [ ] Cleanup: remove dead/duplicate orchestration paths and consolidate remaining provider wiring.

### Rollout Gates

- [x] Deterministic test lane passes with Gateway enabled (`scripts/run_pytest_pr.sh`).
- [ ] Failure semantics match current behavior (timeouts, empty output, cancellation, provider errors).
- [ ] Observability emits stable stage/job/group trace attributes in both legacy and Gateway paths.
- [x] Document managed vs self-host Gateway operational requirements and fallback switches.

### Qt 6.11 Track (Parallel)

- [ ] Validate PySide/Qt 6.11 compatibility and enumerate required code/test updates.
- [ ] Land compatibility fixes and pin final version after deterministic suite passes.

Notes:
- As of 2026-03-08, the latest PyPI `PySide6` release remains `6.10.2`; compatibility execution against `6.11` is blocked until wheels are published.


## Deferred Backlog

The broader exploratory roadmap that previously followed this section (deep cleanup phases, historical success metrics, and long-range platform ideas) has been moved to:

- `docs/archive/work_plan_backlog_legacy.md`

`docs/work_plan.md` remains the active execution checklist for current dashboard priorities.
