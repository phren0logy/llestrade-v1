# Troubleshooting

## `Missing .../var/gateway/local.env`

Copy the template first:

```bash
mkdir -p var/gateway
cp gateway/templates/local.env.example var/gateway/local.env
```

## `Unable to read OP_SERVICE_ACCOUNT_TOKEN from macOS Keychain`

The Keychain bootstrap step has not been completed, or the entry was removed. Rerun:

```bash
gateway/scripts/seed_service_account_token.sh
```

## `1Password service-account authentication failed`

The cached token may be stale or the service account may no longer have access to the referenced vault/items. Reseed the Keychain token and confirm the service account can read the required secrets.

## Wrangler resource creation or deploy fails

Common causes:

- the Cloudflare API token lacks Worker, KV, D1, or domain permissions
- the custom hostname in `GATEWAY_PUBLIC_HOSTNAME` is not available in the Cloudflare zone
- the Worker name conflicts with an existing deployment in another environment

## `/status/` does not return `401` without auth

The gateway may be serving a stale deploy, or the request may not be reaching the custom domain you expect. Confirm the app and scripts are using the hostname from `GATEWAY_PUBLIC_HOSTNAME`.

## `/status/` returns `500` with `Default Password Detected`

`STATUS_AUTH_API_KEY` was not synced correctly. Rerun:

```bash
gateway/scripts/sync_secrets.sh
gateway/scripts/deploy.sh
```

## The Anthropic verification call fails

Check:

- `ANTHROPIC_API_KEY` is present in 1Password and valid
- the Worker secret sync completed successfully
- the gateway app API key in `src/config.ts` matches the value printed by `gateway/scripts/print_app_env.sh`
