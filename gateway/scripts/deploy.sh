#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

load_local_env
for cmd in npx op security git npm node curl uv; do
  require_command "${cmd}"
done

"${GATEWAY_SCRIPT_DIR}/bootstrap.sh"
"${GATEWAY_SCRIPT_DIR}/sync_secrets.sh"

load_service_account_env
load_cloudflare_api_token

log "Deploying Worker ${GATEWAY_WORKER_NAME} to ${GATEWAY_PUBLIC_BASE_URL}."
set +e
deploy_output="$(cd "${GATEWAY_RENDER_DIR}" && npx wrangler deploy 2>&1)"
deploy_status=$?
set -e
printf '%s\n' "${deploy_output}"
if [[ "${deploy_status}" -ne 0 ]]; then
  exit "${deploy_status}"
fi

deployed_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
limits_db_id="$(state_field limits_db_id || true)"
kv_namespace_id="$(state_field kv_namespace_id || true)"
upstream_ref="$(lock_field ref)"
version_id="$(printf '%s\n' "${deploy_output}" | sed -nE 's/^Current Version ID: ([a-f0-9-]+)$/\1/p' | tail -n 1)"
save_state "${limits_db_id}" "${kv_namespace_id}" "${deployed_at}" "${upstream_ref}" "${version_id}"

log "Running post-deploy verification."
"${GATEWAY_SCRIPT_DIR}/verify.sh"
