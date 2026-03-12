# Secrets

The scripts expect all real credentials to stay outside git. Tracked files contain placeholders only.

## Keychain

macOS Keychain stores one secret:

- service: `llestrade.gateway`
- account: `op-service-account-token`
- value: raw 1Password service-account token

You can override the service and account names in `var/gateway/local.env`.

## 1Password References

`var/gateway/local.env` points at these 1Password secrets:

- `GATEWAY_SERVICE_ACCOUNT_TOKEN_REF`
- `GATEWAY_CLOUDFLARE_API_TOKEN_REF`
- `GATEWAY_ANTHROPIC_API_KEY_REF`
- `GATEWAY_OPENAI_API_KEY_REF`
- `GATEWAY_GOOGLE_VERTEX_SERVICE_ACCOUNT_REF`
- `GATEWAY_APP_API_KEY_REF`
- `GATEWAY_STATUS_API_KEY_REF`

Use `op://vault/item/field` references. Item UUIDs are supported.

## Secret Roles

- Cloudflare API token:
  used by Wrangler to create D1, create KV, deploy the Worker, tail logs, and roll back versions
- Anthropic API key:
  stored as the Worker secret `ANTHROPIC_API_KEY`
- OpenAI API key:
  stored as the Worker secret `OPENAI_API_KEY`
- Google Vertex service account JSON:
  stored as the Worker secret `GOOGLE_VERTEX_SERVICE_ACCOUNT_JSON`
- gateway app API key:
  rendered into the upstream `src/config.ts` so the desktop app can authenticate to the gateway
- status API key:
  stored as the Worker secret `STATUS_AUTH_API_KEY` and used only by operator scripts

## Rotation

If the service-account token changes:

1. update the value in 1Password
2. rerun `gateway/scripts/seed_service_account_token.sh`

If the Cloudflare, Anthropic, OpenAI, Google Vertex, app, or status secret changes:

1. update the 1Password item
2. rerun `gateway/scripts/bootstrap.sh`
3. rerun `gateway/scripts/sync_secrets.sh`
4. rerun `gateway/scripts/deploy.sh`
