#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

load_local_env
for cmd in op security; do
  require_command "${cmd}"
done

load_service_account_env
app_api_key="$(op_read_ref "${GATEWAY_APP_API_KEY_REF}")" || die "Unable to read the gateway app API key from 1Password."
[[ -n "${app_api_key}" ]] || die "Gateway app API key was empty."

cat <<EOF
export FRD_ENABLE_PYDANTIC_AI_GATEWAY=true
export PYDANTIC_AI_GATEWAY_BASE_URL=$(shell_quote "${GATEWAY_PUBLIC_BASE_URL}")
export PYDANTIC_AI_GATEWAY_API_KEY=$(shell_quote "${app_api_key}")
EOF
