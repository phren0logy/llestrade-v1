# Plan for Consistent Filenaming (`example.md` -> `example_summary.md`)

**Context:** Ensure that if an original file was `example.pdf`, its converted markdown `example.md` results in a summary `example_summary.md`. The current application logic appears to already support this. This plan focuses on verifying this and making tests more explicit.

**Phase 1: Application Code Review & Confirmation (No changes anticipated)**

1.  **Objective**: Confirm current filename generation logic in `llm_summary_thread.py` and `analysis_tab.py` adheres to `basename.md` -> `basename_summary.md`.
2.  **Actions**:
    - Review `os.path.splitext(os.path.basename(markdown_path))[0]` usage.
3.  **Expected Outcome**: Confirmation of correct current implementation.

**Phase 2: Test Enhancements for Explicit Verification**

1.  **Objective**: Update tests to robustly verify the naming convention, including varied filenames.
2.  **File**: `tests/test_analysis_tab_file_operations.py`
3.  **Actions**:
    - **In `test_summarize_with_llm_writes_to_summaries_subdir`**:
      - Use descriptive mock input filenames (e.g., `"annual_report.md"`).
      - Add a case with multiple dots in the basename (e.g., `"project_alpha.v2.doc.md"`).
      - Assert specific output summary names (e.g., `annual_report_summary.md`, `project_alpha.v2.doc_summary.md`).
    - **In `test_combine_summaries_reads_from_subdir_writes_to_root`**:
      - Update mocked summary filenames to match the new convention.
      - Verify extracted basenames in the combined summary content.

**Phase 3 considerations (from previous overall plan):**

- General testing best practices (file content verification, clarity of mocks).
- Consideration for a dedicated `test_llm_summary_thread.py` in the future.
