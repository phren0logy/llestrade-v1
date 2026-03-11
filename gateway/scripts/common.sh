#!/usr/bin/env bash

set -euo pipefail

GATEWAY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_ROOT="$(cd "${GATEWAY_SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${GATEWAY_ROOT}/.." && pwd)"
GATEWAY_VAR_DIR="${REPO_ROOT}/var/gateway"
GATEWAY_STATE_FILE="${GATEWAY_VAR_DIR}/state.json"
GATEWAY_LOCAL_ENV="${GATEWAY_VAR_DIR}/local.env"
GATEWAY_UPSTREAM_DIR="${GATEWAY_VAR_DIR}/upstream/pydantic-ai-gateway"
GATEWAY_RENDER_ROOT="${GATEWAY_VAR_DIR}/rendered/pydantic-ai-gateway"
GATEWAY_RENDER_DIR="${GATEWAY_RENDER_ROOT}/deploy"
GATEWAY_TEMPLATE_DIR="${GATEWAY_ROOT}/templates"
GATEWAY_LOCK_FILE="${GATEWAY_TEMPLATE_DIR}/upstream.lock.json"

log() {
  printf '[gateway] %s\n' "$*"
}

die() {
  printf '[gateway] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

ensure_var_dir() {
  mkdir -p "${GATEWAY_VAR_DIR}" "${GATEWAY_VAR_DIR}/upstream" "${GATEWAY_VAR_DIR}/rendered"
}

node_json_string() {
  node -e 'process.stdout.write(JSON.stringify(process.argv[1]))' "$1"
}

shell_quote() {
  printf '%q' "$1"
}

load_local_env() {
  ensure_var_dir
  if [[ ! -f "${GATEWAY_LOCAL_ENV}" ]]; then
    die "Missing ${GATEWAY_LOCAL_ENV}. Start from ${GATEWAY_TEMPLATE_DIR}/local.env.example."
  fi
  set -a
  # shellcheck disable=SC1090
  source "${GATEWAY_LOCAL_ENV}"
  set +a

  : "${GATEWAY_PUBLIC_HOSTNAME:?Missing GATEWAY_PUBLIC_HOSTNAME}"
  : "${GATEWAY_WORKER_NAME:?Missing GATEWAY_WORKER_NAME}"
  : "${GATEWAY_ONEPASSWORD_VAULT:?Missing GATEWAY_ONEPASSWORD_VAULT}"
  : "${GATEWAY_CLOUDFLARE_ACCOUNT_ID:?Missing GATEWAY_CLOUDFLARE_ACCOUNT_ID}"
  : "${GATEWAY_SERVICE_ACCOUNT_TOKEN_REF:?Missing GATEWAY_SERVICE_ACCOUNT_TOKEN_REF}"
  : "${GATEWAY_CLOUDFLARE_API_TOKEN_REF:?Missing GATEWAY_CLOUDFLARE_API_TOKEN_REF}"
  : "${GATEWAY_ANTHROPIC_API_KEY_REF:?Missing GATEWAY_ANTHROPIC_API_KEY_REF}"
  : "${GATEWAY_OPENAI_API_KEY_REF:?Missing GATEWAY_OPENAI_API_KEY_REF}"
  : "${GATEWAY_GOOGLE_VERTEX_SERVICE_ACCOUNT_REF:?Missing GATEWAY_GOOGLE_VERTEX_SERVICE_ACCOUNT_REF}"
  : "${GATEWAY_APP_API_KEY_REF:?Missing GATEWAY_APP_API_KEY_REF}"
  : "${GATEWAY_STATUS_API_KEY_REF:?Missing GATEWAY_STATUS_API_KEY_REF}"

  export GATEWAY_KEYCHAIN_SERVICE="${GATEWAY_KEYCHAIN_SERVICE:-llestrade.gateway}"
  export GATEWAY_KEYCHAIN_ACCOUNT="${GATEWAY_KEYCHAIN_ACCOUNT:-op-service-account-token}"
  export GATEWAY_PUBLIC_BASE_URL="https://${GATEWAY_PUBLIC_HOSTNAME}"
  export GATEWAY_LIMITS_DB_NAME="${GATEWAY_WORKER_NAME}-limits"
  export GATEWAY_KV_NAMESPACE_TITLE="${GATEWAY_WORKER_NAME}-cache"
}

lock_field() {
  local field="$1"
  node -e '
    const fs = require("node:fs");
    const payload = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
    const value = payload[process.argv[2]];
    if (value === undefined) process.exit(2);
    process.stdout.write(String(value));
  ' "${GATEWAY_LOCK_FILE}" "${field}" || die "Unable to read ${field} from ${GATEWAY_LOCK_FILE}"
}

save_state() {
  local limits_db_id="$1"
  local kv_namespace_id="$2"
  local deployed_at="$3"
  local upstream_ref="$4"
  local version_id="${5:-}"

  mkdir -p "$(dirname "${GATEWAY_STATE_FILE}")"
  cat >"${GATEWAY_STATE_FILE}" <<EOF
{
  "worker_name": $(node_json_string "${GATEWAY_WORKER_NAME}"),
  "public_hostname": $(node_json_string "${GATEWAY_PUBLIC_HOSTNAME}"),
  "public_base_url": $(node_json_string "${GATEWAY_PUBLIC_BASE_URL}"),
  "limits_db_name": $(node_json_string "${GATEWAY_LIMITS_DB_NAME}"),
  "limits_db_id": $(node_json_string "${limits_db_id}"),
  "kv_namespace_title": $(node_json_string "${GATEWAY_KV_NAMESPACE_TITLE}"),
  "kv_namespace_id": $(node_json_string "${kv_namespace_id}"),
  "upstream_ref": $(node_json_string "${upstream_ref}"),
  "rendered_deploy_dir": $(node_json_string "${GATEWAY_RENDER_DIR}"),
  "last_deployed_at": $(node_json_string "${deployed_at}"),
  "last_version_id": $(node_json_string "${version_id}")
}
EOF
}

state_field() {
  local field="$1"
  if [[ ! -f "${GATEWAY_STATE_FILE}" ]]; then
    return 1
  fi
  node -e '
    const fs = require("node:fs");
    const payload = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
    const value = payload[process.argv[2]];
    if (value === undefined || value === null || value === "") process.exit(1);
    process.stdout.write(String(value));
  ' "${GATEWAY_STATE_FILE}" "${field}"
}

keychain_read() {
  security find-generic-password -s "${GATEWAY_KEYCHAIN_SERVICE}" -a "${GATEWAY_KEYCHAIN_ACCOUNT}" -w
}

load_service_account_env() {
  local token
  token="$(keychain_read)" || die "Unable to read OP_SERVICE_ACCOUNT_TOKEN from macOS Keychain. Run seed_service_account_token.sh first."
  export OP_SERVICE_ACCOUNT_TOKEN="${token}"
}

op_read_ref() {
  op read "$1"
}

load_cloudflare_api_token() {
  local token
  token="$(op_read_ref "${GATEWAY_CLOUDFLARE_API_TOKEN_REF}")" || die "Unable to read Cloudflare API token from 1Password."
  export CLOUDFLARE_API_TOKEN="${token}"
  export CLOUDFLARE_ACCOUNT_ID="${GATEWAY_CLOUDFLARE_ACCOUNT_ID}"
}

render_text_template() {
  local template_path="$1"
  local destination_path="$2"
  shift 2

  local rendered
  rendered="$(cat "${template_path}")"
  while (($#)); do
    local key="$1"
    local value="$2"
    shift 2
    rendered="${rendered//${key}/${value}}"
  done
  printf '%s\n' "${rendered}" >"${destination_path}"
}

render_gateway_config() {
  local app_api_key="$1"
  mkdir -p "${GATEWAY_RENDER_DIR}/src"
  render_text_template \
    "${GATEWAY_TEMPLATE_DIR}/config.ts.template" \
    "${GATEWAY_RENDER_DIR}/src/config.ts" \
    "__APP_API_KEY__" "$(node_json_string "${app_api_key}")"
}

render_wrangler_config() {
  local limits_db_id="$1"
  local kv_namespace_id="$2"
  local upstream_ref="$3"
  render_text_template \
    "${GATEWAY_TEMPLATE_DIR}/wrangler.jsonc.template" \
    "${GATEWAY_RENDER_DIR}/wrangler.jsonc" \
    "__WORKER_NAME__" "${GATEWAY_WORKER_NAME}" \
    "__CUSTOM_DOMAIN__" "${GATEWAY_PUBLIC_HOSTNAME}" \
    "__LIMITS_DB_NAME__" "${GATEWAY_LIMITS_DB_NAME}" \
    "__LIMITS_DB_ID__" "${limits_db_id}" \
    "__KV_NAMESPACE_ID__" "${kv_namespace_id}" \
    "__UPSTREAM_REF__" "${upstream_ref}"
}

sync_upstream_repo() {
  local repo_url
  local upstream_ref
  repo_url="$(lock_field repo_url)"
  upstream_ref="$(lock_field ref)"

  if [[ ! -d "${GATEWAY_UPSTREAM_DIR}/.git" ]]; then
    git clone "${repo_url}" "${GATEWAY_UPSTREAM_DIR}"
  fi

  git -C "${GATEWAY_UPSTREAM_DIR}" fetch --all --tags --prune
  git -C "${GATEWAY_UPSTREAM_DIR}" checkout "${upstream_ref}"
}

prepare_rendered_deploy_dir() {
  local deploy_subpath
  deploy_subpath="$(lock_field deploy_subpath)"

  rm -rf "${GATEWAY_RENDER_ROOT}"
  mkdir -p "$(dirname "${GATEWAY_RENDER_ROOT}")"
  cp -R "${GATEWAY_UPSTREAM_DIR}" "${GATEWAY_RENDER_ROOT}"
  rm -rf "${GATEWAY_RENDER_ROOT}/.git"
  [[ -d "${GATEWAY_RENDER_ROOT}/${deploy_subpath}" ]] || die "Expected deploy path ${deploy_subpath} inside rendered upstream repo."
}

parse_wranger_id_from_output() {
  sed -nE 's/.*(database_id|id) = "([^"]+)".*/\2/p'
}

parse_deploy_url_from_output() {
  sed -nE 's#.*(https://[^ ]+).*#\1#p' | head -n 1
}

lookup_d1_id() {
  local db_name="$1"
  npx wrangler d1 list --json --cwd "${GATEWAY_RENDER_DIR}" | node -e '
    const fs = require("node:fs");
    const input = fs.readFileSync(0, "utf8");
    const payload = JSON.parse(input);
    const db = payload.find((entry) => entry.name === process.argv[1]);
    if (!db || !db.uuid) process.exit(1);
    process.stdout.write(String(db.uuid));
  ' "${db_name}"
}

lookup_kv_namespace_id() {
  local title="$1"
  npx wrangler kv namespace list --cwd "${GATEWAY_RENDER_DIR}" | node -e '
    const fs = require("node:fs");
    const input = fs.readFileSync(0, "utf8");
    const payload = JSON.parse(input);
    const namespace = payload.find((entry) => entry.title === process.argv[1]);
    if (!namespace || !namespace.id) process.exit(1);
    process.stdout.write(String(namespace.id));
  ' "${title}"
}
