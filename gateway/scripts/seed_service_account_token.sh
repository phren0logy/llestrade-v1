#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

load_local_env
require_command op
require_command security

log "Reading service-account token from 1Password reference ${GATEWAY_SERVICE_ACCOUNT_TOKEN_REF}."
token="$(op_read_ref "${GATEWAY_SERVICE_ACCOUNT_TOKEN_REF}")" || die "Unable to read the service-account token from 1Password."
[[ -n "${token}" ]] || die "Service-account token was empty."

security add-generic-password \
  -U \
  -a "${GATEWAY_KEYCHAIN_ACCOUNT}" \
  -s "${GATEWAY_KEYCHAIN_SERVICE}" \
  -w "${token}" >/dev/null

log "Stored the service-account token in macOS Keychain (${GATEWAY_KEYCHAIN_SERVICE}/${GATEWAY_KEYCHAIN_ACCOUNT})."
