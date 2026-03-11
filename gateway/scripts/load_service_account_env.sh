#!/usr/bin/env bash

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf 'Source this script instead of executing it:\n' >&2
  printf '  source gateway/scripts/load_service_account_env.sh\n' >&2
  exit 1
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

load_local_env
load_service_account_env
