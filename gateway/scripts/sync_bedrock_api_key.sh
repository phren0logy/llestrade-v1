#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ITEM_TITLE="aws-bedrock"
SERVICE_NAME="bedrock.amazonaws.com"
CREDENTIAL_AGE_DAYS="${GATEWAY_BEDROCK_CREDENTIAL_AGE_DAYS:-30}"

require_command uv
require_command op
require_command python3
require_command security

load_local_env
load_service_account_env

aws_cli() {
  uv tool run --from awscli aws "$@"
}

resolve_iam_user_name() {
  if [[ $# -gt 0 && -n "${1}" ]]; then
    printf '%s\n' "${1}"
    return 0
  fi

  local arn
  arn="$(aws_cli sts get-caller-identity --query Arn --output text)"
  if [[ "${arn}" =~ :user/(.+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi

  die "Unable to infer an IAM user from AWS caller identity (${arn}). Pass the IAM user name as the first argument."
}

existing_item_id() {
  OP_SERVICE_ACCOUNT_TOKEN="${OP_SERVICE_ACCOUNT_TOKEN}" \
    op item get "${ITEM_TITLE}" --vault "${GATEWAY_ONEPASSWORD_VAULT}" --format json 2>/dev/null | \
    python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get("id",""))'
}

upsert_item() {
  local item_id="$1"
  local user_name="$2"
  local credential_id="$3"
  local credential_alias="$4"
  local bearer_token="$5"
  local expires_at="$6"
  local create_date="$7"

  local notes
  notes=$(
    python3 - "${credential_id}" "${credential_alias}" "${user_name}" "${create_date}" "${expires_at}" <<'PY'
import sys
credential_id, credential_alias, user_name, create_date, expires_at = sys.argv[1:6]
lines = [
    "AWS Bedrock gateway bearer token and rotation metadata.",
    "",
    f"service-specific-credential-id: {credential_id}",
    f"service-credential-alias: {credential_alias}",
    f"iam-user: {user_name}",
    f"created-at: {create_date}",
]
if expires_at:
    lines.append(f"expires-at: {expires_at}")
print("\n".join(lines))
PY
  )

  local payload
  payload=$(
    OP_SERVICE_ACCOUNT_TOKEN="${OP_SERVICE_ACCOUNT_TOKEN}" \
      op item template get "API Credential" | \
      python3 - \
        "${ITEM_TITLE}" \
        "${notes}" \
        "${credential_alias}" \
        "${bearer_token}" \
        "${credential_id}" \
        "${user_name}" \
        "${expires_at}" <<'PY'
import json
import sys

title, notes, credential_alias, bearer_token, credential_id, user_name, expires_at = sys.argv[1:8]
item = json.load(sys.stdin)
fields = item.setdefault("fields", [])
expires_date = expires_at.split("T", 1)[0] if expires_at else "0"

def set_field(field_id: str, value: str, field_type: str = "STRING", *, label: str | None = None, purpose: str | None = None) -> None:
    for field in fields:
        if field.get("id") == field_id or field.get("label") == field_id:
            field["value"] = value
            field["type"] = field_type
            if label is not None:
                field["label"] = label
            if purpose is not None:
                field["purpose"] = purpose
            return
    field = {"id": field_id, "label": label or field_id, "type": field_type, "value": value}
    if purpose is not None:
        field["purpose"] = purpose
    fields.append(field)

item["title"] = title
set_field("notesPlain", notes, "STRING", purpose="NOTES")
set_field("hostname", "bedrock.amazonaws.com")
set_field("username", credential_alias)
set_field("bearer-token", bearer_token, "CONCEALED")
set_field("service-credential-id", credential_id)
set_field("service-credential-alias", credential_alias)
set_field("iam-user", user_name)
set_field("credential-type", "Bearer Token")
set_field("expires", expires_date, "DATE")
set_field("expires-at", expires_at)

print(json.dumps(item))
PY
  )

  if [[ -n "${item_id}" ]]; then
    printf '%s' "${payload}" | OP_SERVICE_ACCOUNT_TOKEN="${OP_SERVICE_ACCOUNT_TOKEN}" op item edit "${item_id}" --vault "${GATEWAY_ONEPASSWORD_VAULT}" -
    return 0
  fi

  printf '%s' "${payload}" | OP_SERVICE_ACCOUNT_TOKEN="${OP_SERVICE_ACCOUNT_TOKEN}" op item create --vault "${GATEWAY_ONEPASSWORD_VAULT}" -
}

main() {
  local user_name
  user_name="$(resolve_iam_user_name "${1:-}")"

  log "Creating AWS Bedrock service-specific credential for IAM user ${user_name}."
  local response
  response="$(
    aws_cli iam create-service-specific-credential \
      --user-name "${user_name}" \
      --service-name "${SERVICE_NAME}" \
      --credential-age-days "${CREDENTIAL_AGE_DAYS}" \
      --output json
  )"

  local parsed
  parsed="$(
    python3 - "${response}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
credential = payload.get("ServiceSpecificCredential") or {}

secret = (
    credential.get("ServiceCredentialSecret")
    or credential.get("ServicePassword")
    or ""
)
alias = (
    credential.get("ServiceCredentialAlias")
    or credential.get("ServiceUserName")
    or ""
)
credential_id = credential.get("ServiceSpecificCredentialId") or ""
user_name = credential.get("UserName") or ""
create_date = credential.get("CreateDate") or ""
expires_at = credential.get("ExpirationDate") or ""

if not secret:
    raise SystemExit("AWS CLI response did not include a Bedrock bearer token.")

print("\n".join([user_name, credential_id, alias, secret, expires_at, create_date]))
PY
  )"

  local parsed_user_name credential_id credential_alias bearer_token expires_at create_date
  parsed_user_name="$(printf '%s\n' "${parsed}" | sed -n '1p')"
  credential_id="$(printf '%s\n' "${parsed}" | sed -n '2p')"
  credential_alias="$(printf '%s\n' "${parsed}" | sed -n '3p')"
  bearer_token="$(printf '%s\n' "${parsed}" | sed -n '4p')"
  expires_at="$(printf '%s\n' "${parsed}" | sed -n '5p')"
  create_date="$(printf '%s\n' "${parsed}" | sed -n '6p')"

  local item_id
  item_id="$(existing_item_id || true)"
  upsert_item "${item_id}" "${parsed_user_name:-${user_name}}" "${credential_id}" "${credential_alias}" "${bearer_token}" "${expires_at}" "${create_date}" >/dev/null

  log "Stored the Bedrock bearer token in 1Password item ${ITEM_TITLE} (${GATEWAY_ONEPASSWORD_VAULT})."
  if [[ -n "${credential_id}" ]]; then
    log "Credential ID: ${credential_id}"
  fi
  if [[ -n "${expires_at}" ]]; then
    log "Expires at: ${expires_at}"
  fi
}

main "$@"
