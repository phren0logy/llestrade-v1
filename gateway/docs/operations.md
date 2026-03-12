# Operations

## Deploy an Update

Use this flow whenever you change the upstream pin, gateway templates, or secret references:

```bash
gateway/scripts/bootstrap.sh
gateway/scripts/sync_secrets.sh
gateway/scripts/deploy.sh
```

## Verify the Running Gateway

```bash
gateway/scripts/verify.sh
```

Verification includes:

- `/status/` returns `401` without auth
- `/status/` returns `200` with `STATUS_AUTH_API_KEY`
- `/status/` includes a `capacity` section with per-provider state
- a live `pydantic_ai` Anthropic request succeeds against the custom domain
- a live `pydantic_ai` OpenAI request succeeds against the custom domain
- a live `pydantic_ai` Google Vertex request succeeds against the custom domain

## Tail Logs

```bash
gateway/scripts/tail.sh
```

## Roll Back

List recent deployments with Wrangler:

```bash
npx wrangler deployments list --cwd var/gateway/rendered/deploy --name <worker-name> --json
```

Then roll back:

```bash
gateway/scripts/rollback.sh <version-id> "reason"
```

## Print App Exports

```bash
gateway/scripts/print_app_env.sh
```

Export those values before launching the app when you want it to use the self-hosted gateway.

## Updating the Upstream Pin

Update `gateway/templates/upstream.lock.json` with the new commit SHA, then rerun bootstrap, secrets sync, deploy, and verify.
