#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

version_id="${1:-}"
message="${2:-Rollback from gateway/scripts/rollback.sh}"
[[ -n "${version_id}" ]] || die "Usage: gateway/scripts/rollback.sh <version-id> [message]"

load_local_env
for cmd in npx op security; do
  require_command "${cmd}"
done

load_service_account_env
load_cloudflare_api_token

[[ -f "${GATEWAY_RENDER_DIR}/wrangler.jsonc" ]] || die "Rendered wrangler.jsonc not found. Run gateway/scripts/bootstrap.sh first."

exec npx wrangler rollback "${version_id}" --cwd "${GATEWAY_RENDER_DIR}" --name "${GATEWAY_WORKER_NAME}" --message "${message}" --yes
