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

metadata_response_file="$(mktemp)"
trap 'rm -f "${metadata_response_file}"' EXIT
metadata_code="$(curl -sS -o "${metadata_response_file}" -w '%{http_code}' -H "Authorization: ${app_api_key}" "${GATEWAY_PUBLIC_BASE_URL}/metadata/models?provider=anthropic")"
[[ "${metadata_code}" == "200" ]] || die "Expected /metadata/models?provider=anthropic with app API key to return 200, got ${metadata_code}: $(cat "${metadata_response_file}")"
metadata_code="$(curl -sS -o "${metadata_response_file}" -w '%{http_code}' -H "Authorization: ${app_api_key}" "${GATEWAY_PUBLIC_BASE_URL}/metadata/models?provider=anthropic_bedrock")"
[[ "${metadata_code}" == "200" ]] || die "Expected /metadata/models?provider=anthropic_bedrock with app API key to return 200, got ${metadata_code}: $(cat "${metadata_response_file}")"

GATEWAY_VERIFY_BASE_URL="${GATEWAY_PUBLIC_BASE_URL}" \
GATEWAY_VERIFY_API_KEY="${app_api_key}" \
uv run python - <<'PY'
import os

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.gateway import gateway_provider

anthropic_gateway = gateway_provider(
    "anthropic",
    api_key=os.environ["GATEWAY_VERIFY_API_KEY"],
    base_url=os.environ["GATEWAY_VERIFY_BASE_URL"],
)
anthropic_agent = Agent(
    model=AnthropicModel("claude-sonnet-4-5", provider=anthropic_gateway),
    system_prompt="Reply with the single word OK.",
)
anthropic_result = anthropic_agent.run_sync(
    "Return exactly OK.",
    model_settings={"max_tokens": 16, "temperature": 0},
)
if not str(anthropic_result.output).strip().startswith("OK"):
    raise SystemExit(f"Anthropic gateway verification returned unexpected output: {anthropic_result.output!r}")

openai_gateway = gateway_provider(
    "openai",
    api_key=os.environ["GATEWAY_VERIFY_API_KEY"],
    base_url=os.environ["GATEWAY_VERIFY_BASE_URL"],
)
openai_agent = Agent(
    model=OpenAIChatModel("gpt-4.1", provider=openai_gateway),
    system_prompt="Reply with the single word OK.",
)
openai_result = openai_agent.run_sync(
    "Return exactly OK.",
    model_settings={"max_tokens": 16, "temperature": 0},
)
if not str(openai_result.output).strip().startswith("OK"):
    raise SystemExit(f"OpenAI gateway verification returned unexpected output: {openai_result.output!r}")

google_gateway = gateway_provider(
    "gemini",
    api_key=os.environ["GATEWAY_VERIFY_API_KEY"],
    base_url=os.environ["GATEWAY_VERIFY_BASE_URL"],
)
google_agent = Agent(
    model=GoogleModel("gemini-2.5-pro", provider=google_gateway),
    system_prompt="Reply with the single word OK.",
)
google_result = google_agent.run_sync(
    "Return exactly OK.",
    model_settings={"max_tokens": 64, "temperature": 0},
)
if not str(google_result.output).strip().startswith("OK"):
    raise SystemExit(f"Google Vertex gateway verification returned unexpected output: {google_result.output!r}")

bedrock_gateway = gateway_provider(
    "bedrock",
    api_key=os.environ["GATEWAY_VERIFY_API_KEY"],
    base_url=os.environ["GATEWAY_VERIFY_BASE_URL"],
)
bedrock_agent = Agent(
    model=BedrockConverseModel("us.anthropic.claude-sonnet-4-6", provider=bedrock_gateway),
    system_prompt="Reply with the single word OK.",
)
bedrock_result = bedrock_agent.run_sync(
    "Return exactly OK.",
    model_settings={"max_tokens": 16, "temperature": 0},
)
if not str(bedrock_result.output).strip().startswith("OK"):
    raise SystemExit(f"Bedrock gateway verification returned unexpected output: {bedrock_result.output!r}")
PY

log "Gateway verification succeeded."
