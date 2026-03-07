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

## Phase 1: Deep Code Analysis & Cleanup (Week 1-2)

### 1.1 Add Development Dependencies

```toml
[dependency-groups]
dev = [
    "pytest>=8.4.1",
    "pytest-cov>=5.0.0",      # Coverage reporting
    "pytest-qt>=4.4.0",        # Qt testing support
    "langfuse>=2.0.0",         # LLM observability
    "rich>=13.0.0",            # Better console output
]
```

### 1.2 Dead Code Elimination

- [ ] Remove FastAPI/Electron references from documentation
- [ ] Delete commented-out code blocks
- [ ] Remove unused imports and functions
- [ ] Clean up abandoned architectural experiments
- [ ] Remove duplicate implementations

### 1.3 File Organization & Refactoring

**Files to Split (>800 lines)**:

- [ ] `src/app/ui/workspace/controllers/documents.py` → Separate data/model helpers, tree-building utilities, and UI wiring so each module stays <400 lines.
- [ ] `src/app/ui/workspace/controllers/bulk.py` → Extract run-control logic and manifest formatting into dedicated services.
- [ ] `src/app/ui/workspace/controllers/reports.py` → Break into prompt orchestration, table widgets, and background job coordination modules.

**Project Structure Review**:

- [ ] Keep `src/app/`, `src/core/`, and `src/common/` boundaries clean; move any lingering dashboard helpers out of legacy folders.
- [ ] Review `src/common/` for proper shared components.
- [ ] Identify redundant modules for consolidation.

### 1.4 Leverage PySide6 Components Better

**Components to Adopt**:

- [ ] Migrate from custom JSON settings to `QSettings`
- [ ] Replace individual `QThread` usage with `QThreadPool` where appropriate
- [ ] Implement `QUndoStack` for editors (refinement stage)
- [ ] Use `QProgressDialog` consistently instead of custom progress widgets
- [ ] Add `QFileSystemWatcher` for file monitoring instead of polling

### 1.5 Add LLM Observability

Moved to "Proposed Additions (For Review Later)" to align with Phoenix/OpenInference instrumentation direction.

## Phase 2: Business Logic Testing (Week 2-3)

### 2.1 Core Component Tests (Non-GUI Focus)

**LLM Integration Layer**:

- [ ] Test provider abstraction (`src/common/llm/`)
- [ ] Test chunking strategies
- [ ] Test token counting accuracy
- [ ] Test error handling and retries
- [ ] Test cost calculation

**Document Processing Pipeline**:

- [ ] Test PDF conversion logic
- [ ] Test Word document handling
- [ ] Test markdown generation
- [ ] Test file validation
- [ ] Test batch processing

**Project Management**:

- [ ] Test ProjectManager CRUD operations
- [ ] Test auto-save functionality
- [ ] Test backup creation
- [ ] Test state persistence
- [ ] Test migration between stages

### 2.2 Worker Thread Tests

**Thread Safety**:

- [ ] Verify signal/slot communication
- [ ] Test thread cleanup on cancellation
- [ ] Test resource management
- [ ] Test error propagation
- [ ] Test concurrent operations

### 2.3 Security & Data Flow

**API Key Management**:

- [ ] Test SecureSettings encryption
- [ ] Test keyring integration
- [ ] Test fallback mechanisms
- [ ] Test key rotation

## Phase 3: Smart Refactoring (Week 3-4)

### 3.1 Replace Custom Code with Libraries

**PySide6 Optimizations**:

- [ ] Replace custom settings with `QSettings`
- [ ] Use `QStateMachine` for stage management
- [ ] Leverage `QCompleter` for form auto-completion
- [ ] Use built-in validators (`QRegularExpressionValidator`, etc.)

**Potential Additions** (evaluate carefully):

- [ ] Consider `watchdog` for file system monitoring
- [ ] Evaluate `structlog` for structured logging
- [ ] Consider `pydantic` for data validation

### 3.2 Code Consolidation

**Merge Similar Functionality**:

- [ ] Consolidate progress dialog implementations
- [ ] Unify error handling patterns
- [ ] Standardize worker thread base class
- [ ] Merge duplicate file utilities

**Extract Common Patterns**:

- [ ] Create base classes for common UI patterns
- [ ] Extract shared validation logic
- [ ] Centralize configuration management

## Phase 4: REMOVED - Replaced by Dashboard UI Refactoring

_The wizard-style UI with linear stages has been replaced by the dashboard approach in Priority 0._

## Phase 5: Stabilization (Week 5-6)

### 5.1 Performance Optimization

- [ ] Memory profiling (target: <200MB)
- [ ] Optimize stage transitions (<1 second)
- [ ] Improve LLM response streaming
- [ ] Reduce startup time

### 5.2 Cross-Platform Testing

- [ ] Test on macOS (primary)
- [ ] Test on Windows
- [ ] Test on Linux
- [ ] Fix platform-specific issues

### 5.3 Documentation

- [ ] Update README.md with accurate status
- [ ] Create user guide for the dashboard workflow
- [ ] Document API for developers
- [ ] Add inline code documentation

## Distribution & Packaging

- [x] Centralize resource lookup for frozen bundles (`app_resource_root`, prompt/resource helpers)
- [x] Add PyInstaller spec for the dashboard (`scripts/build_dashboard.spec`)
- [x] Provide per-platform build wrappers (`scripts/build_macos.sh`, `scripts/build_linux.sh`, `scripts/build_windows.ps1`)
- [ ] Validate macOS bundle end-to-end (launch, highlights, bulk analysis)
- [ ] Produce and validate Windows bundle
- [ ] Produce and validate Linux bundle
- [ ] Add packaging jobs to CI with artifact uploads
- [ ] Plan code signing / notarization strategy per platform

## Success Metrics

### Code Quality

- [ ] Test coverage >80% for business logic
- [ ] No files >800 lines
- [ ] All functions <50 lines
- [ ] Cyclomatic complexity <10

### Performance

- [ ] Memory usage <200MB normal operation
- [ ] Stage transitions <1 second
- [ ] No memory leaks over 8-hour session
- [ ] Startup time <3 seconds

### Maintainability

- [ ] Clear separation of concerns
- [ ] Consistent patterns throughout
- [ ] Comprehensive test suite
- [ ] Well-documented code

## Future Considerations (Deferred)

### Enhanced Features

- Cost tracking with comprehensive reporting
- Template gallery system
- Model auto-discovery
- Advanced progress indicators with pause/resume
- Session management and recovery

### Distribution

- Package as standalone executables
- Auto-update system
- Code signing for trusted distribution
- Professional installers (MSI, DMG, AppImage)

### Enterprise Features

- Multi-user support
- LDAP/Active Directory integration
- Audit logging
- HIPAA compliance features
- Cloud deployment options

### AI Enhancements

- Custom fine-tuned models
- Local LLM support (Ollama)
- Intelligent document classification
- Automated quality checks

### Collaboration

- Real-time collaboration
- Comment and annotation system
- Change tracking
- Approval workflows

## Immediate Next Steps

- [ ] Add automatic hand-off from conversion completion to bulk-analysis job scheduling where configured.

## Proposed Additions (For Review Later)

- Summary group migration helper: upgrade `bulk_analysis/*/config.json` from version 1 to 2 with validation.
- Documentation: add `highlights/` to the folder structure diagram; clarify that highlight counts use PDFs only.
- Auto hand-off: optionally kick off bulk analysis after conversions complete when configured per group.
- Skip logic: avoid re-processing existing bulk outputs using timestamp + prompt hash.
- Observability: instrument LLM calls with Phoenix/OpenInference and attach group context; update dev deps snippet accordingly.
- Worker traceability: add consistent job identifiers in logs and progress signals.
- Highlights UX: add "Re-extract highlights" action and surface reasons when a file remains pending.
- Welcome polish: finish remaining cleanup messaging and small UI tweaks.

## Notes

- Breaking changes are acceptable (no backward compatibility needed)
- Focus on simple, reliable solutions over complex engineering
- Preserve folder structure throughout document pipeline
- Use QThreadPool for parallel operations (max 3 workers)
- Test business logic with pytest, defer GUI testing
- Phoenix for LLM observability only
- Archive old code after dashboard implementation
