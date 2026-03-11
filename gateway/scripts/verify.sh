#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

load_local_env
for cmd in curl op security uv; do
  require_command "${cmd}"
done

load_service_account_env

app_api_key="$(op_read_ref "${GATEWAY_APP_API_KEY_REF}")" || die "Unable to read the gateway app API key from 1Password."
status_api_key="$(op_read_ref "${GATEWAY_STATUS_API_KEY_REF}")" || die "Unable to read the gateway status API key from 1Password."
[[ -n "${app_api_key}" ]] || die "Gateway app API key was empty."
[[ -n "${status_api_key}" ]] || die "Gateway status API key was empty."

unauth_code="$(curl -sS -o /dev/null -w '%{http_code}' "${GATEWAY_PUBLIC_BASE_URL}/status/")"
[[ "${unauth_code}" == "401" ]] || die "Expected /status/ without auth to return 401, got ${unauth_code}."

auth_code="$(curl -sS -o /dev/null -w '%{http_code}' -H "Authorization: ${status_api_key}" "${GATEWAY_PUBLIC_BASE_URL}/status/")"
[[ "${auth_code}" == "200" ]] || die "Expected /status/ with status API key to return 200, got ${auth_code}."

GATEWAY_VERIFY_BASE_URL="${GATEWAY_PUBLIC_BASE_URL}" \
GATEWAY_VERIFY_API_KEY="${app_api_key}" \
uv run python - <<'PY'
import os

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.gateway import gateway_provider

gateway = gateway_provider(
    "anthropic",
    api_key=os.environ["GATEWAY_VERIFY_API_KEY"],
    base_url=os.environ["GATEWAY_VERIFY_BASE_URL"],
)
agent = Agent(
    model=AnthropicModel("claude-sonnet-4-5", provider=gateway),
    system_prompt="Reply with the single word OK.",
)
result = agent.run_sync("Return exactly OK.", model_settings={"max_tokens": 16, "temperature": 0})
if not str(result.output).strip().startswith("OK"):
    raise SystemExit(f"Gateway verification returned unexpected output: {result.output!r}")
PY

log "Gateway verification succeeded."
