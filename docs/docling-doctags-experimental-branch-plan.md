# Experimental DocTags-Only Branch: VLM-First Via Local Docling Serve URL

## Summary
- Create an experimental branch that removes all conversion backends except Docling and persists only DocTags as the converted-document artifact.
- Use Docling Serve over HTTP as the only backend integration, with a single app-level setting: a local no-auth `base_url`.
- Use the Docling VLM pipeline first, with `granite_docling` as the primary preset.
- Keep the standard Docling pipeline as the explicit fallback/control path, selectable at the project level, not as a silent retry path.
- Move converted-document metadata and citation state into `.llestrade/citations.db`; do not use stored Markdown, stored JSON sidecars, or Markdown front matter in this branch.
- After leaving planning, track implementation in Linear under `pyside-llestrade` as one parent issue plus child issues.

## Important Interfaces
- App-level Docling backend setting:
  - `base_url`
  - default assumption: `http://127.0.0.1:5001`
  - no API key, no auth headers, no native/local-runtime management in this phase
- Project-level conversion settings:
  - `pipeline_mode`: `vlm_primary` | `standard_only`
  - `vlm_preset`: default `granite_docling`
  - `standard_profile`: named standard-pipeline preset for OCR/layout/table mode
- Persist exactly one converted artifact per source:
  - `converted_documents/<relative/source/path>.<original-ext>.doctags.txt`
- Extend `.llestrade/citations.db` to store:
  - converted artifact path
  - original source references and checksum
  - pipeline mode/preset/profile used
  - page counts and conversion timestamps
  - DocTags-derived evidence segments and bbox/grid citation rows

## Key Changes
- Conversion and storage
  - Remove Azure DI, local PDF extraction, Pandoc DOCX conversion, Markdown copy-through, and helper-selection UI from the branch.
  - Route document conversion through Docling Serve only.
  - For PDFs, default to Docling Serve `pipeline=vlm` with `vlm_pipeline_preset=granite_docling`.
  - For unsupported-by-VLM formats, and projects explicitly set to fallback, call the standard Docling pipeline through the same Docling Serve endpoint.
  - Do not store Markdown or JSON sidecars on disk.
- Metadata and citations
  - Stop using YAML front matter for converted documents.
  - Promote `.llestrade/citations.db` into the branch’s general converted-document metadata store.
  - Add a DocTags parser that:
    - splits pages via `<page_break>`
    - extracts text-bearing block segments and deterministic evidence IDs
    - reads location tags into normalized grid bbox records
    - assigns synthetic `source_id`s for citation anchoring
  - Build evidence ledgers, citation verification, and source lookup from DB-backed DocTags segments instead of page-marked Markdown plus Azure JSON.
- Downstream workflow adaptation
  - Bulk analysis, prompt preview, and report evidence assembly must discover and consume `.doctags.txt` inputs directly.
  - Replace Markdown-header chunking with DocTags-aware chunking based on page and block boundaries.
  - Move source-file resolution, page counts, and converter metadata lookups from front matter to the DB.
  - Keep reports, bulk-analysis outputs, and highlights in their current output formats; the experiment changes converted-input storage, not final authored outputs.
- Backend settings scope
  - Add one simple UI/settings field for the Docling Serve URL.
  - Validate that it is a usable HTTP(S) base URL format, but do not add auth, deployment helpers, platform detection, or native-process support yet.
  - Native MLX or platform-specific runtime support is deferred to a later phase.

## Test Plan
- Project/config tests proving the branch exposes only Docling settings and records `base_url`, pipeline mode, preset, and profile correctly.
- Conversion tests proving supported documents persist exactly one converted artifact: `.doctags.txt`.
- PDF tests proving VLM-first projects call Docling Serve with `granite_docling`, while fallback projects call the standard pipeline.
- Non-PDF tests proving standard Docling remains available for the supported rich-document surface while still emitting only DocTags on disk.
- DB tests proving converted-document metadata, source references, and citation rows live in `.llestrade/citations.db` without front matter.
- Citation tests proving DocTags segmentation yields stable evidence IDs, page numbers, and grid bbox records.
- Bulk-analysis, prompt-preview, and report tests proving `.doctags.txt` inputs are discovered, chunked, previewed, and cited correctly.
- Settings tests proving malformed Docling Serve URLs are rejected and unreachable-server failures surface clearly.
- Regression tests proving existing Markdown-converted projects are treated as stale/incompatible and require reconversion.

## Linear Tracking
- Target project: `pyside-llestrade`
- Shape: one parent issue plus child issues
- Planned child issue breakdown:
  - Branch scaffold and Docling-only backend contract
  - Docling Serve URL settings and client integration
  - VLM-first `granite_docling` path
  - Standard fallback pipeline profile
  - DocTags parser plus citation/metadata DB refactor
  - Bulk analysis, prompt preview, and report ingestion on DocTags
  - File-tracker/highlight/source-resolution changes off front matter
  - Benchmark/eval suite comparing VLM-first vs standard fallback
- Planned labels:
  - `pyside-llestrade`
  - `phase-1-conversion`
  - `contract`
  - `integration`
  - `testing`
  - `evals`

## Assumptions and Defaults
- “Highest quality VLM first” means `granite_docling` unless evaluation disproves it.
- The first phase assumes a user-managed local no-auth Docling Serve instance is already running and reachable by URL.
- Native/local runtime management, auth, and platform-specific deployment support are intentionally deferred.
- The branch is intentionally incompatible with existing converted Markdown artifacts; projects should be reconverted.
- The single-format constraint applies to persisted converted-document artifacts and internal input handling, not to generated reports or highlight outputs.
