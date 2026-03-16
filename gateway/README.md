# Gateway Operations

This directory manages the self-hosted Pydantic AI Gateway used by Llestrade's report and bulk-analysis path.

The deployed shape is intentionally small:

- one public runtime hostname: `gateway.<your-domain>`
- one shared app API key for the desktop app
- one separate `STATUS_AUTH_API_KEY` for script-driven health checks
- Anthropic, AWS Bedrock Claude, OpenAI, and Google Vertex as the enabled upstream providers
- provider-specific adaptive capacity control inside the gateway Worker
- no public admin UI or status hostname

The upstream gateway is headless. Configuration is rendered into an untracked working copy under `var/gateway/rendered/pydantic-ai-gateway/` and deployed to Cloudflare Workers from there.

The current upstream deploy package does not ship a `package-lock.json`, so bootstrap uses `npm install` rather than `npm ci`.
Bootstrap also applies the upstream D1 schema automatically after provisioning or re-rendering the worker.

## First Run

1. Copy `gateway/templates/local.env.example` to `var/gateway/local.env` and fill in the real hostname plus 1Password secret references.
   Include the Cloudflare account id so Wrangler does not need to enumerate memberships.
2. Seed the 1Password service-account token into macOS Keychain:
   `gateway/scripts/seed_service_account_token.sh`
3. Check local prerequisites:
   `gateway/scripts/check_prereqs.sh`
4. Bootstrap the rendered Worker project and Cloudflare resources:
   `gateway/scripts/bootstrap.sh`
5. Sync Worker secrets:
   `gateway/scripts/sync_secrets.sh`
6. Deploy and verify:
   `gateway/scripts/deploy.sh`

## Normal Operations

- Re-render upstream, provision missing resources, and refresh the working copy:
  `gateway/scripts/bootstrap.sh`
- Push updated Anthropic, Bedrock, OpenAI, Google Vertex, or status secrets:
  `gateway/scripts/sync_secrets.sh`
- Deploy:
  `gateway/scripts/deploy.sh`
- Verify:
  `gateway/scripts/verify.sh`
- Inspect authenticated capacity state:
  `curl -H "Authorization: $STATUS_AUTH_API_KEY" https://gateway.<your-domain>/status/`
- Inspect authenticated gateway-backed model metadata for a route/provider:
  `curl -H "Authorization: $PYDANTIC_AI_GATEWAY_API_KEY" "https://gateway.<your-domain>/metadata/models?provider=openai"`
  `curl -H "Authorization: $PYDANTIC_AI_GATEWAY_API_KEY" "https://gateway.<your-domain>/metadata/models?provider=anthropic_bedrock"`
- Tail Worker logs:
  `gateway/scripts/tail.sh`
- Roll back to an earlier Worker version:
  `gateway/scripts/rollback.sh <version-id> "reason"`
- Print app runtime exports:
  `gateway/scripts/print_app_env.sh`

## Auth Model

- 1Password service-account auth is used by every script after bootstrap.
- The service-account token itself is cached in macOS Keychain.
- Wrangler uses a Cloudflare API token loaded from 1Password; no interactive `wrangler login` flow is required.
- The desktop app authenticates to the gateway with `PYDANTIC_AI_GATEWAY_API_KEY`.
- `/status/` is script-only and protected by `STATUS_AUTH_API_KEY`.
- `/metadata/models` uses the same gateway app API key as normal model requests and returns route-safe model metadata for the configured upstream credentials.

## Custom Domain and Access

The production endpoint is the custom domain in `GATEWAY_PUBLIC_HOSTNAME`. The app should use that hostname through `PYDANTIC_AI_GATEWAY_BASE_URL`.

Cloudflare Access is intentionally not placed in front of the runtime gateway hostname in v1, because the current desktop app does not participate in Access login flows or token exchange. If a browser-facing operator surface is added later, Google Workspace via Cloudflare Access is the intended human auth model for that separate hostname.

See `gateway/docs/bootstrap.md`, `gateway/docs/operations.md`, `gateway/docs/secrets.md`, and `gateway/docs/troubleshooting.md` for the detailed workflow.
