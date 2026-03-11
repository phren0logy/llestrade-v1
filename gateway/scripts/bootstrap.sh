#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

load_local_env
for cmd in git node npm npx op security; do
  require_command "${cmd}"
done

load_service_account_env
load_cloudflare_api_token

ensure_var_dir
sync_upstream_repo
prepare_rendered_deploy_dir

app_api_key="$(op_read_ref "${GATEWAY_APP_API_KEY_REF}")" || die "Unable to read the gateway app API key from 1Password."
[[ -n "${app_api_key}" ]] || die "Gateway app API key was empty."

limits_db_id="$(state_field limits_db_id || true)"
kv_namespace_id="$(state_field kv_namespace_id || true)"
upstream_ref="$(lock_field ref)"

if [[ -z "${limits_db_id}" ]]; then
  limits_db_id="$(lookup_d1_id "${GATEWAY_LIMITS_DB_NAME}" || true)"
fi

if [[ -z "${limits_db_id}" ]]; then
  log "Creating D1 database ${GATEWAY_LIMITS_DB_NAME}."
  d1_output="$(npx wrangler d1 create "${GATEWAY_LIMITS_DB_NAME}" --cwd "${GATEWAY_RENDER_DIR}" 2>&1)"
  limits_db_id="$(printf '%s\n' "${d1_output}" | parse_wranger_id_from_output | tail -n 1)"
  if [[ -z "${limits_db_id}" ]]; then
    limits_db_id="$(lookup_d1_id "${GATEWAY_LIMITS_DB_NAME}" || true)"
  fi
  [[ -n "${limits_db_id}" ]] || die "Unable to parse D1 database ID from Wrangler output."
  save_state "${limits_db_id}" "${kv_namespace_id}" "" "${upstream_ref}" ""
fi

if [[ -z "${kv_namespace_id}" ]]; then
  kv_namespace_id="$(lookup_kv_namespace_id "${GATEWAY_KV_NAMESPACE_TITLE}" || true)"
fi

if [[ -z "${kv_namespace_id}" ]]; then
  log "Creating KV namespace ${GATEWAY_KV_NAMESPACE_TITLE}."
  kv_output="$(npx wrangler kv namespace create "${GATEWAY_KV_NAMESPACE_TITLE}" --cwd "${GATEWAY_RENDER_DIR}" 2>&1)"
  kv_namespace_id="$(printf '%s\n' "${kv_output}" | parse_wranger_id_from_output | tail -n 1)"
  if [[ -z "${kv_namespace_id}" ]]; then
    kv_namespace_id="$(lookup_kv_namespace_id "${GATEWAY_KV_NAMESPACE_TITLE}" || true)"
  fi
  [[ -n "${kv_namespace_id}" ]] || die "Unable to parse KV namespace ID from Wrangler output."
  save_state "${limits_db_id}" "${kv_namespace_id}" "" "${upstream_ref}" ""
fi

render_gateway_config "${app_api_key}"
render_wrangler_config "${limits_db_id}" "${kv_namespace_id}" "${upstream_ref}"

log "Installing upstream Worker dependencies."
(cd "${GATEWAY_RENDER_ROOT}" && npm install)

log "Ensuring D1 schema is applied."
(cd "${GATEWAY_RENDER_DIR}" && npx wrangler d1 execute "${GATEWAY_LIMITS_DB_NAME}" --remote --file ../gateway/limits-schema.sql)

save_state "${limits_db_id}" "${kv_namespace_id}" "" "${upstream_ref}" ""
log "Bootstrap complete. Rendered deploy dir: ${GATEWAY_RENDER_DIR}"
