#!/usr/bin/env bash
set -euo pipefail

# Disable third-party pytest plugin auto-discovery to prevent stray plugins
# (e.g., phoenix) from spawning background services and hanging test runs.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Explicitly load pytest-qt when available so fixtures like `qtbot` are present.
PYTEST_ARGS=()
if uv run python -c "import pytestqt.plugin" >/dev/null 2>&1; then
  PYTEST_ARGS+=(-p pytestqt.plugin)
else
  echo "[run_pytest] pytest-qt plugin not installed; Qt fixture tests may skip or fail." >&2
fi

# Explicitly load pytest-cov when available since plugin autoload is disabled.
if uv run python -c "import pytest_cov.plugin" >/dev/null 2>&1; then
  PYTEST_ARGS+=(-p pytest_cov.plugin)
else
  echo "[run_pytest] pytest-cov plugin not installed; --cov options will fail." >&2
fi

exec uv run python -m pytest "${PYTEST_ARGS[@]}" "$@"
