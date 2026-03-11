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

anthropic_api_key="$(op_read_ref "${GATEWAY_ANTHROPIC_API_KEY_REF}")" || die "Unable to read Anthropic API key from 1Password."
status_api_key="$(op_read_ref "${GATEWAY_STATUS_API_KEY_REF}")" || die "Unable to read status API key from 1Password."
[[ -n "${anthropic_api_key}" ]] || die "Anthropic API key was empty."
[[ -n "${status_api_key}" ]] || die "Status API key was empty."

log "Syncing Worker secrets."
printf '%s' "${anthropic_api_key}" | npx wrangler secret put ANTHROPIC_API_KEY --cwd "${GATEWAY_RENDER_DIR}" >/dev/null
printf '%s' "${status_api_key}" | npx wrangler secret put STATUS_AUTH_API_KEY --cwd "${GATEWAY_RENDER_DIR}" >/dev/null

log "Worker secrets are up to date."
