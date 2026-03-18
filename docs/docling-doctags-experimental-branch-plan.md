# Experimental DocTags-Only Branch: Local Docling MLX

## Summary
- This branch is a hard break from the Azure/Markdown conversion pipeline.
- Converted-document artifacts are DocTags only: `converted_documents/<source>.doctags.txt`.
- Conversion runs locally on Apple Silicon through Docling with MLX, not through Docling Serve.
- The branch is PDF-only in this first cut because Sao is the pilot corpus.
- Review-time metadata and citation state live in `.llestrade/citations.db`, not markdown front matter or JSON sidecars.

## Runtime Contract
- Supported source type: PDF only.
- Expected runtime: macOS on Apple Silicon with local `docling` and `mlx` dependencies installed.
- Default conversion mode: `pipeline_mode=vlm_primary`.
- Default VLM preset: `granite_docling`.
- `standard_only` remains an explicit project-level fallback mode, but there is no remote backend URL in this phase.
- Unsupported runtime environments should fail clearly at conversion time instead of silently degrading.

## Artifact Contract
- Exactly one converted artifact is written per PDF source:
  - `converted_documents/<relative/source/path>.pdf.doctags.txt`
- Generated outputs remain markdown:
  - map outputs
  - combined outputs
  - reports
  - highlights
- Citation review resolves those outputs back to source PDFs through the DB.

## Metadata And Citations
- `.llestrade/citations.db` stores:
  - converted artifact path
  - source relative and absolute PDF path
  - source checksum
  - pipeline metadata
  - page counts
  - DocTags-derived evidence segments
  - normalized bbox/grid geometry
- Prompt preview, bulk prompts, reduce prompts, reports, and citation review should prefer DB-backed source metadata over filename or front-matter inference.

## Implementation Notes
- The local converter uses:
  - `DocumentConverter`
  - `VlmPipeline`
  - `VlmConvertOptions.from_preset("granite_docling", engine_options=MlxVlmEngineOptions())`
- DocTags text is rendered for prompt/report consumption, while bbox review uses normalized geometry extracted from `<loc_...>` tokens.
- File tracking and highlight eligibility treat DocTags artifacts as the converted-document source of truth.

## Scope For This Phase
- In scope:
  - local Docling MLX conversion
  - PDF-only converted inputs
  - DB-backed citation metadata
  - bulk analysis / reduce / reports / prompt preview consuming `.doctags.txt`
  - citation review using DocTags-derived geometry
- Out of scope:
  - Docling Serve
  - cross-platform runtime support
  - non-PDF conversion
  - backward compatibility with markdown-era converted artifacts

## Validation
- Sao is the first pilot project.
- Required validation steps:
  - conversion succeeds on a representative Sao subset
  - prompt preview shows generated citation appendices
  - map/reduce/report outputs use local `[C#]` citations
  - citation review jumps to the correct PDF page and bbox
  - reset and reconversion are required for stale pre-branch artifacts
