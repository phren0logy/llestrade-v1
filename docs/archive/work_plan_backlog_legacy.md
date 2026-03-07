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
