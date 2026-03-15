# Testing Strategy

## Goals
- Keep pull request feedback deterministic and fast.
- Isolate live-provider tests so network/API secrets never block normal development.
- Focus coverage on core/worker behavior most likely to regress during refactors.

## Marker Taxonomy
- `unit`: fast deterministic unit tests.
- `core`: dashboard core/domain tests.
- `worker`: worker and orchestration tests.
- `ui`: Qt/controller/widget tests.
- `integration`: multi-module integration tests.
- `live_provider`: tests that call real external LLM providers.
- `slow`: intentionally long-running tests.

Live-provider tests are skipped by default unless `RUN_LIVE_PROVIDER_TESTS=1`.

## Local Commands
- `scripts/run_pytest.sh tests/` for general local runs.
- `scripts/run_pytest_pr.sh` for deterministic PR-equivalent checks.
- `scripts/run_pytest_live.sh` for optional real-provider validation.

Live-provider notes:
- Normal pytest runs use isolated test settings, isolated QSettings, and a fake keyring backend.
- `scripts/run_pytest_live.sh` prefers `.env.live` when present.
- `.env.live` may contain plain env vars or `op://...` 1Password references; the script resolves them with `op run --env-file=.env.live`.
- Live-provider tests no longer read credentials from the app keychain.

## CI Lanes
- `tests-pr-deterministic.yml`: required deterministic suite with coverage artifact.
- `tests-live-providers.yml`: optional lane via `workflow_dispatch` or PR label `run-live-provider-tests`.

## Coverage Policy
- PR deterministic lane reports coverage for `src/app/core`, `src/app/workers`, and `src/common/llm`.
- Citation coverage should include `src/app/core/citations.py` (indexing, verification, chunked Azure ingestion).
- Coverage trends should be reviewed on high-risk refactors; thresholds can be raised incrementally once the suite stabilizes.
