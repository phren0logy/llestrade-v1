# Development Progress

This document tracks completed work on Llestrade (formerly Forensic Psych Report Drafter).

## 2025-07-05 - Dashboard Foundations Landed
- Replaced the wizard-era launcher with the dashboard entry point in `src/app/main_window.py`, wiring the welcome stage and workspace controller together so projects can open directly from the new UI.
- Finished the workspace shell in `src/app/ui/stages/project_workspace.py`, including the Documents, Highlights, Bulk Analysis, and Reports tabs plus home navigation; this retired the old `src/new/` scaffolding entirely.
- Implemented FileTracker, the bulk analysis group system, and the updated `ProjectManager` under `src/app/core/`, ensuring projects capture source selections, conversion helpers, and dashboard metrics on disk.
- Consolidated workers into `src/app/workers/` with a shared `QThreadPool` coordinator, new conversion/highlight/bulk/report worker classes, and manifest-aware skip logic for re-runs.
- Delivered the dashboard documents and bulk tabs (tree selectors, conversion banners, activity log, run pending/all flows) plus placeholder-aware reports UI, aligned with the new workspace file layout.
- Added pytest coverage for the dashboard-era modules under `tests/app/**` (FileTracker reconcilers, workspace controllers, worker coordinator, placeholder analysis, etc.), replacing the placeholder `tests/new_ui` plan.

## 2025-07-04 - Fixed Project Advancement from Setup to Import Stage
- Added `project_data` property to ProjectManager to provide data in format expected by stages
- Added `project_name` property to extract name from project path
- Updated StageManager to properly propagate project reference to ALL stages after creation
- Fixed metadata update to accept ProjectMetadata objects directly
- Ensured all stages receive project reference when project is created
- Resolved issue where users couldn't advance past project setup stage
- Created and ran tests to verify the fixes work correctly

## 2025-07-04 - QStackedWidget UI Simplification
- Replaced complex dynamic widget replacement system with QStackedWidget
- Pre-create all stage widgets at startup for predictable memory usage
- Simplified StageManager to remove dynamic widget creation/deletion
- Added reset() method to BaseStage for clearing state when revisiting
- Fixed stage initialization errors in AnalysisStage and ReportGenerationStage
- Updated both stages to work with get_available_providers_and_models() return format
- Added PySide6 UI best practices section to CLAUDE.md
- Resolved blank screen issue when creating new project
- Follows Qt/PySide best practices for simpler, more maintainable code

## 2025-07-04 - Project Setup Stage Architecture Improvements
- Fixed stage initialization order to prevent accessing project_data before initialization
- Refactored BaseStage to not auto-call load_state in constructor
- Stage manager now calls load_state and _validate after stage creation
- Added proper error handling with user-friendly error dialogs
- Added detailed logging to navigation updates for debugging
- Fixed load_state to properly handle None project for new projects
- Registered missing ReportGenerationStage in startup
- Added project assignment after project creation in stage manager
- Improved error messages and logging throughout stage loading process

## 2025-07-04 - Project Setup Stage Fixes
- Fixed blank screen when creating new project (removed incorrect project manager assignment)
- Removed evaluation date field from project setup form
- Fixed output directory to use app settings default with case name as subfolder
- Added automatic output directory update when case name changes
- Enabled advancing from project setup by implementing complete_setup method
- Modified stage manager to handle project creation before advancing to import stage
- Output directory now creates case-specific folder with sanitized name

## 2025-07-04 - Welcome Screen Settings Consolidation
- Renamed "API Key Status" section to "Settings" on welcome screen
- Changed "Configure API Keys" button to "Open Settings"
- Added evaluator name configuration status to settings display
- Updated quick start guide to mention settings configuration
- Improved project setup stage to open settings dialog directly
- Consolidated all app-level settings (user info, defaults, API keys) in one dialog
- Fixed APIKeyDialog integration in settings dialog (passed correct settings object)
- Properly embedded API key configuration as a tab within the settings dialog
- Simplified settings dialog: removed User tab, kept only evaluator name in Defaults tab
- Removed unnecessary fields (title, license number, email) for cleaner single-user experience
- Fixed visual glitches in welcome screen settings display (proper layout cleanup)

## 2025-07-04 - Report Generation Stage Implementation
- Implemented ReportGenerationStage for the new UI (stage 6 of 7)
- Created ReportGenerationThread worker for async report creation
- Added support for both integrated analysis and template-based reports
- Integrated with existing LLM providers (Anthropic, Gemini, Azure OpenAI)
- Updated StageManager to properly load the new stage
- Fixed "New Project" button issue in welcome stage (removed incompatible debug code)
- Updated documentation to reflect 6/7 stages now complete
- Only Refinement stage remains to complete the new UI

## 2025-07-04 - Project Setup Page Improvements
- Removed API key indicators from project setup (moved to app-level settings)
- Added automatic evaluator name population from application settings
- Created comprehensive Settings dialog with user info, defaults, and API keys
- Fixed DocumentImportStage initialization order bug causing attribute errors
- Fixed QLayout warning by properly clearing layouts before adding new ones
- Evaluator name is now a single-user app setting, not project-specific

## 2025-07-04 - Documentation Consolidation
- Consolidated 6 overlapping documentation files into 4 focused documents
- Updated main README.md with current project status and two-UI explanation
- Created comprehensive roadmap.md combining feature ideas with development priorities
- Removed outdated files with incorrect January 2025 dates
- Established clear documentation hierarchy

## 2025-07-03 - New UI Implementation

### Overview

Completed Phase 1 of the new UI implementation, establishing the foundation for parallel UI development while fixing critical thread safety issues in the legacy application.

## Completed Tasks

### 1. Fixed Critical Thread Safety Issues ✅

- **Problem**: Direct UI access from worker threads causing malloc double-free crashes
- **Solution**: Replaced all direct UI access with Qt signal/slot mechanism
- **Files Updated**:
  - `LLMSummaryThread`: Added `_safe_emit_status()` method, removed status_panel parameter
  - `IntegratedAnalysisThread`: Same pattern, fixed 32 instances of direct UI access
  - Updated all test files to match new signatures
- **Result**: Thread-safe UI updates, preventing memory crashes

### 2. Reorganized Project Structure ✅

- **Created Directory Structure**:
  ```
  src/
  ├── legacy/ui/     # Current UI (moved from /ui)
  ├── common/llm/    # Shared LLM code (moved from /llm)
  └── new/           # New UI implementation
      ├── core/      # Foundation classes
      ├── stages/    # Workflow stages
      ├── widgets/   # Reusable components
      └── workers/   # Background threads
  ```
- **Updated Imports**: Created automated script that updated 21 files
- **Backward Compatibility**: Created symlinks for smooth transition

### 3. Implemented Foundation Classes ✅

- **SecureSettings** (`src/new/core/secure_settings.py`)

  - OS keychain integration for API keys
  - Falls back to encrypted file storage
  - Window state persistence
  - Recent projects management

- **ProjectManager** (`src/new/core/project_manager.py`)

  - Handles .frpd project files
  - Auto-save every 60 seconds
  - Automatic backups (keeps last 10)
  - Cost tracking by provider and stage
  - Workflow state persistence

- **StageManager** (`src/new/core/stage_manager.py`)
  - Controls stage transitions
  - Ensures proper cleanup between stages
  - Navigation state management
  - Dynamic stage loading

### 4. Created New UI Entry Point ✅

- **main_new.py**: Functional new UI with basic window
- **Smart Launcher**: `main.py` routes to either UI based on `--new-ui` flag
- **Both UIs Working**: Can run side-by-side for testing

## Key Achievements

1. **Zero Disruption**: Legacy UI continues to work perfectly
2. **Clean Architecture**: Proper separation of concerns
3. **Thread Safety**: All UI updates now thread-safe
4. **Memory Management**: Foundation for <200MB memory target
5. **Professional Structure**: Ready for team development

## Next Steps

### Week 2 Priorities

1. **ProjectSetupStage** (2-3 days)
   - Case information form
   - API key validation
   - Template selection
2. **Workflow Sidebar** (2 days)
   - Visual progress indicator
   - Stage navigation
3. **Cost Tracking Widget** (1 day)

   - Real-time display
   - Export functionality

4. **Welcome Screen** (2 days)
   - Recent projects
   - Quick start wizard

## Technical Notes

### Running the Application

```bash
# Legacy UI (default)
./main.py
./run_app.sh

# New UI
./main.py --new-ui
./run_new_ui.sh
USE_NEW_UI=true ./main.py
```

### Import Changes

- UI imports: `from ui.` → `from src.legacy.ui.`
- LLM imports: `from llm.` → `from src.common.llm.`

### Testing

All existing tests updated and passing. Ready to create parallel test suite for new UI.

## Risks & Mitigations

1. **Risk**: Import errors during transition
   - **Mitigation**: Symlinks maintain compatibility
2. **Risk**: Memory leaks in new stages
   - **Mitigation**: BaseStage class enforces cleanup pattern
3. **Risk**: API key security
   - **Mitigation**: OS keychain with encrypted fallback

## Conclusion

Phase 1 successfully completed. The foundation is solid, both UIs are functional, and we're ready to build the new stage-based workflow. The architecture follows PySide6 best practices and positions us well for the remaining development.
