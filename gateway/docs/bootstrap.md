# Bootstrap

## Prerequisites

- macOS with the `security` CLI
- `git`
- `node`, `npm`, `npx`
- `uv`
- `op` CLI
- `curl`
- a Cloudflare API token with Worker, KV, D1, and domain-management permissions
- a 1Password service account with access to the LLestrade vault entries used by the gateway scripts

## Local Config

Copy the template:

```bash
mkdir -p var/gateway
cp gateway/templates/local.env.example var/gateway/local.env
```

Then edit `var/gateway/local.env` with:

- the real gateway hostname
- the Worker name
- the Cloudflare account id
- the correct 1Password secret references

## Seed the 1Password Service Account

The first run needs one interactive `op` read so the service-account token can be cached in macOS Keychain.

```bash
gateway/scripts/seed_service_account_token.sh
```

After that, the other scripts load `OP_SERVICE_ACCOUNT_TOKEN` from Keychain automatically.

## Bootstrap the Gateway

```bash
gateway/scripts/check_prereqs.sh
gateway/scripts/bootstrap.sh
```

`bootstrap.sh` does the following:

- clones or updates the pinned upstream `pydantic-ai-gateway` repo into `var/gateway/upstream/pydantic-ai-gateway`
- copies the full upstream repo into `var/gateway/rendered/pydantic-ai-gateway/`
- renders `src/config.ts` with the shared app API key and Anthropic/OpenAI/Google Vertex routing
- creates the D1 database and KV namespace if they are not already tracked in `var/gateway/state.json`
- renders `wrangler.jsonc` for the custom domain
- installs the upstream Worker dependencies with `npm install`
- applies `gateway/limits-schema.sql` to the remote D1 database

## Initial Deploy

```bash
gateway/scripts/sync_secrets.sh
gateway/scripts/deploy.sh
```

The deploy script also runs `gateway/scripts/verify.sh` at the end.
