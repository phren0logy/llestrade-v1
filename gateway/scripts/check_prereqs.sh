#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

load_local_env

for cmd in git node npm npx op security curl uv; do
  require_command "${cmd}"
done

if keychain_read >/dev/null 2>&1; then
  log "Found 1Password service-account token in macOS Keychain."
else
  die "Missing Keychain bootstrap token. Run gateway/scripts/seed_service_account_token.sh first."
fi

load_service_account_env
if ! op whoami >/dev/null 2>&1; then
  die "1Password service-account authentication failed. Re-seed the Keychain token if it was rotated."
fi

load_cloudflare_api_token
if [[ -z "${CLOUDFLARE_API_TOKEN}" ]]; then
  die "Cloudflare API token is empty."
fi

if ! uv run python -c "import pydantic_ai" >/dev/null 2>&1; then
  die "The repo environment is missing pydantic_ai. Run 'uv sync' before using verify.sh or deploy.sh."
fi

log "Prerequisites look good."
