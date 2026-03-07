#!/usr/bin/env bash
set -euo pipefail

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export RUN_LIVE_PROVIDER_TESTS=1

if [[ $# -eq 0 ]]; then
  set -- tests
fi

exec "$(dirname "$0")/run_pytest.sh" -m "live_provider" "$@"
