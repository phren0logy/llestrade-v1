#!/usr/bin/env bash
set -euo pipefail

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export RUN_LIVE_PROVIDER_TESTS=0

if [[ $# -eq 0 ]]; then
  set -- \
    --cov=src/app/core \
    --cov=src/app/workers \
    --cov=src/common/llm \
    --cov-report=term-missing \
    --cov-report=xml \
    tests
fi

exec "$(dirname "$0")/run_pytest.sh" -m "not live_provider and not slow" "$@"
