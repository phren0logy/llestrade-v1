#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

load_local_env
for cmd in npx op security; do
  require_command "${cmd}"
done

load_service_account_env
load_cloudflare_api_token

[[ -f "${GATEWAY_RENDER_DIR}/wrangler.jsonc" ]] || die "Rendered wrangler.jsonc not found. Run gateway/scripts/bootstrap.sh first."

exec npx wrangler tail --cwd "${GATEWAY_RENDER_DIR}"
