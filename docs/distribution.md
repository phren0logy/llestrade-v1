# Distribution Prep Notes

## Packaging Strategy
- Use PyInstaller to freeze `main.py` as the entry point.
- Include `src/app/resources/` as bundled data so prompts/templates ship with the binary.
- Load resources at runtime via `config.paths.app_resource_root()`, which transparently resolves paths when frozen (uses `sys._MEIPASS`).
- Generate a lockfile for the build with `uv export --format requirements-txt > build/requirements.txt` before running PyInstaller to keep the frozen environment reproducible.

## Runtime Secrets
- The application no longer auto-loads placeholder keys from `config.template.env`; users must provide their own `.env` or enter credentials through the UI (stored via `SecureSettings`).
- Do **not** ship a populated `.env`. Document the required variables (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AZURE_*`) in release notes.

## Dependency Considerations
- `PySide6` requires its Qt plugin folders (`platforms`, `imageformats`, `styles`) and an accompanying `qt.conf`. Configure the PyInstaller spec to collect these directories via `PySide6.QtCore.QLibraryInfo`.
- `pypandoc-binary` embeds Pandoc; ensure `PYPANDOC_PANDOC` is discovered or copy the binary into the bundle with `--add-binary` if detection fails.
- `keyring` backends vary per OS. The frozen build should include the default backend for each target platform; add optional backends as needed.
- Network SDKs (`anthropic`, `openai`, `google-genai`, `azure-ai-documentintelligence`) are heavyweight; keep them but consider conditional loading in the UI if footprint becomes an issue.

## Build Next Steps
1. Use the PyInstaller spec at `scripts/build_dashboard.spec`, which now stages Qt plugins/resources automatically and writes to `dist/<platform>/`.
2. On each platform, prefer the packaging helper under `packaging/<platform>/`:
   - macOS: `./packaging/macos/build_app.sh [--skip-icon] [--fresh-dist]`
   - Windows: `.\packaging\windows\build_app.ps1 [-FreshDist]`
   - Linux: `./packaging/linux/build_app.sh [--fresh-dist]`
   These wrappers standardise cache locations, support clean rebuilds, and call `uv run pyinstaller --clean --noconfirm scripts/build_dashboard.spec` under the hood. One-off migration utilities are quarantined in `scripts/legacy/`, with compatibility wrappers retained at original script paths.
3. Extend CI to execute the bundling pipeline per platform and attach artifacts (macOS `.app`/bundle, Windows `.exe`, Linux AppDir/AppImage or tarball).

## Prompt & Placeholder Packaging

- Placeholder lists ship from `src/app/resources/placeholder_sets/`. The placeholder editor reconciles bundled copies with the user’s custom directory; include new sets here so they are available out of the box.
- Bulk/report workers rely on YAML front matter (injected during conversion) to recover PDF provenance; QA should confirm converted markdown inside packaged builds still includes `sources` metadata so placeholders like `{source_pdf_relative_path}` resolve.
