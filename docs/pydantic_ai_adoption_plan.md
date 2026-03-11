# Pydantic AI Adoption Plan

## Summary

This document records which [Pydantic AI](https://ai.pydantic.dev/) and [Pydantic AI Gateway](https://ai.pydantic.dev/gateway/) utilities should replace current custom code in Llestrade, which are worth adopting later, and which are not a good fit for the current desktop document-analysis architecture.

The goal is not to turn the app into a generic agent framework. The goal is to reduce custom provider plumbing, improve multi-provider support, and use upstream abstractions where they are stronger than the app's current worker-specific implementations.

Current local baseline:

- Installed package: `pydantic-ai-slim 1.67.0`
- Current app pattern: worker-driven imperative LLM calls, custom provider wrappers, custom token budgeting, custom retry/fallback paths, and custom Phoenix/OpenTelemetry spans

## Current Progress

The worker runtime has already advanced materially beyond the original starting point for this plan.

Completed in the current worker/backend layer:

- `direct` Pydantic AI requests now power both the direct-provider and gateway-backed worker paths in [`src/app/workers/llm_backend.py`](../src/app/workers/llm_backend.py).
- Provider/model normalization is centralized in [`src/app/workers/llm_backend.py`](../src/app/workers/llm_backend.py), including gateway model IDs and provider-specific aliases such as Bedrock Claude mappings.
- `UsageLimits` is now used in the worker backend layer for request budgeting, with provider-aware pre-request token counting where supported.
- Gateway-specific retry transport and model-level concurrency limiting are implemented in [`src/app/workers/llm_backend.py`](../src/app/workers/llm_backend.py).
- The old worker-runtime fallback escape hatch has been removed. Worker execution no longer falls back to `src.common.llm.factory` or legacy `provider.generate(...)` paths.
- The old `LegacyProviderBackend` name has been retired in favor of `PydanticAIDirectBackend`.
- Full `ModelResponse` passthrough is implemented in the worker backend. Success paths return raw Pydantic AI responses and failure paths raise exceptions.

Still open:

- model-level instrumentation via Pydantic AI / InstrumentedModel
- upstream failover via `FallbackModel` and/or Gateway routing groups
- broader migration of non-worker legacy code under [`src/common/llm/`](../src/common/llm/) and [`src/config/app_config.py`](../src/config/app_config.py)
- later-phase items such as structured outputs, evals, toolsets, and durable execution

## Current State

The current LLM execution path is centered on custom abstractions:

- [`src/app/workers/llm_backend.py`](../src/app/workers/llm_backend.py) provides `LLMExecutionBackend`, `PydanticAIDirectBackend`, and `PydanticAIGatewayBackend`.
- [`src/app/workers/bulk_analysis_worker.py`](../src/app/workers/bulk_analysis_worker.py) and [`src/app/workers/bulk_reduce_worker.py`](../src/app/workers/bulk_reduce_worker.py) perform custom prompt budgeting, chunk sizing, and provider invocation.
- [`src/app/workers/report_common.py`](../src/app/workers/report_common.py) and [`src/app/workers/report_worker.py`](../src/app/workers/report_worker.py) build prompts and call providers imperatively.
- [`src/common/llm/tokens.py`](../src/common/llm/tokens.py) implements mixed token-counting strategies, including provider-specific fallbacks and character-based estimates.
- [`src/config/observability.py`](../src/config/observability.py) emits app-specific Phoenix/OpenTelemetry spans around worker stages.
- The worker backend now uses Pydantic AI `direct` plus raw `ModelResponse` passthrough for worker execution. Remaining custom logic is primarily in prompt assembly, chunk planning, checkpointing, and the surrounding worker orchestration.

This architecture is reasonable for a desktop app with long-running jobs and file-based checkpoints, but it currently duplicates several upstream capabilities that Pydantic AI already provides.

## Recommended Now

### 1. Use Direct Model Requests for Imperative Worker Calls

Use Pydantic AI's [`direct` API](https://ai.pydantic.dev/direct/) as the preferred low-level request surface for worker-style execution.

Why:

- The docs explicitly position `model_request`, `model_request_sync`, and the streaming variants as the right fit when building your own abstractions around model calls.
- This better matches the app's existing worker architecture than wrapping single-shot calls in `Agent(...)` inside the backend.

What it should replace:

- Most of the custom imperative `Agent(...).run_sync(...)` usage in [`src/app/workers/llm_backend.py`](../src/app/workers/llm_backend.py).

Replacement status:

- Completed for the worker runtime.
- The app-level worker abstractions, cancellation, checkpointing, and prompt assembly still remain, but low-level request execution has already moved onto Pydantic AI `direct`.

### 2. Use Full ModelResponse Passthrough for Worker Results

Use raw [`ModelResponse`](https://ai.pydantic.dev/api/messages/) objects as the worker-backend success result instead of flattening them into a custom `content/raw` envelope.

Why:

- Pydantic AI represents reasoning through `ModelResponse.parts`, including `ThinkingPart`, rather than through a flat reasoning string.
- Flattening to `response.text` discards structured response parts and makes reasoning support brittle.
- A full passthrough boundary keeps the app aligned with upstream abstractions and avoids inventing a partial custom mirror of Pydantic AI's response model.

What it should replace:

- flattened `response.text`-only normalization in [`src/app/workers/llm_backend.py`](../src/app/workers/llm_backend.py)
- `response.raw["reasoning"]` / `response.raw["thinking"]` recovery logic in worker call sites
- worker-side reasoning heuristics that depend on guessed response shape instead of real Pydantic AI response parts

Replacement status:

- Implemented for the worker runtime.
- Backend success now returns raw `ModelResponse`; backend failure raises exceptions instead of returning a flattened success/error envelope.
- Workers should continue consuming `response.text` for final answer text, `response.parts` for reasoning-aware behavior, and `response.usage` for token accounting.

### 3. Use Model Strings and Provider/Profile Resolution Instead of Expanding Custom Provider Branches

Use the model-string approach documented in the [Models overview](https://ai.pydantic.dev/models/overview/) and [Gateway docs](https://ai.pydantic.dev/gateway/) rather than continuing to grow custom provider branching logic.

Why:

- Pydantic AI automatically resolves the model class, provider, and profile when you use strings like `<provider>:<model>`.
- Gateway adds the `gateway/` prefix, using the documented `gateway/<provider>:<model>` form.
- This is the idiomatic multi-provider path for Anthropic, Gemini, OpenAI, Bedrock, and other supported providers.

What it should replace:

- The hard-coded provider switch tree in [`src/app/workers/llm_backend.py`](../src/app/workers/llm_backend.py).

Replacement status:

- Largely complete for worker execution.
- Direct-provider and gateway-backed workers now resolve provider/model naming centrally in [`src/app/workers/llm_backend.py`](../src/app/workers/llm_backend.py).
- Remaining legacy compatibility is outside the worker runtime, primarily in [`src/common/llm/`](../src/common/llm/) and [`src/config/app_config.py`](../src/config/app_config.py).

### 4. Use UsageLimits for Pre-Request Token Enforcement Where Supported

Use [`UsageLimits`](https://ai.pydantic.dev/api/usage/) and `count_tokens_before_request=True` for providers that support real token preflight.

Why:

- This is the upstream mechanism for request-time usage enforcement.
- It reduces duplicated input-budget logic in bulk and report workers.

What it should replace:

- Parts of the custom prompt budget enforcement in [`src/app/workers/bulk_analysis_worker.py`](../src/app/workers/bulk_analysis_worker.py) and [`src/app/core/bulk_analysis_runner.py`](../src/app/core/bulk_analysis_runner.py).

Replacement status:

- Implemented in the worker backend layer.
- The remaining custom logic is the app-specific part that should stay custom: chunk planning, file/document-level budgeting, evidence ledgers, and hierarchical chunking.

Provider caveat:

- In the current installed version, Anthropic, Google, and Bedrock support real `count_tokens(...)` preflight.
- Installed `OpenAIChatModel` still raises `NotImplementedError` for pre-request token counting, so OpenAI and Azure OpenAI must continue to use fallback counting for this specific feature.

### 5. Use ConcurrencyLimitedModel for Shared Provider/Gateway Throughput Caps

Adopt [`ConcurrencyLimitedModel`](https://ai.pydantic.dev/api/concurrency/) for model-level concurrency limits.

Why:

- Worker threads and Qt scheduling are not the same as provider request concurrency control.
- This gives a cleaner place to enforce shared request limits across bulk/report jobs hitting the same provider or gateway route.

What it should replace:

- Future custom concurrency throttling around provider or gateway request dispatch.

Replacement status:

- Implemented for the worker backend layer.
- Keep the current worker orchestration and job coordination; model-level concurrency is now used only for network request limiting.

### 6. Use Instrumented Models for LLM-Layer Telemetry

Adopt Pydantic AI's [Debugging and Monitoring / Logfire integration](https://ai.pydantic.dev/logfire/) and [`InstrumentedModel`](https://ai.pydantic.dev/api/models/instrumented/) for model-level telemetry.

Why:

- Upstream instrumentation already captures model requests, messages, tool calls, and usage.
- The instrumentation API supports `include_content=False`, which is important for a PII-heavy forensic workflow.

What it should replace:

- Custom LLM-only tracing where the app is just surfacing request/response behavior.

Replacement status:

- Partial replacement.
- Keep current app-level worker/stage spans in [`src/config/observability.py`](../src/config/observability.py). They express domain workflow context that model instrumentation does not replace.

### 7. Use Pydantic AI Retry Transports for Pydantic-AI-Backed Provider Paths

Use the retry utilities in [HTTP Request Retries](https://ai.pydantic.dev/retries/) for paths that are executed through Pydantic AI providers or Gateway providers.

Why:

- The upstream transport wrappers are the idiomatic place for HTTP retry behavior.
- This is cleaner than scattering retry logic across multiple provider implementations.

What it should replace:

- Selected provider-specific retry code, especially where the app is only re-implementing standard HTTP retry behavior.

Replacement status:

- Implemented for Gateway-backed worker execution, and standardized for the direct Pydantic AI worker backend as well.
- Keep custom retry behavior when it is tied to provider SDK semantics or worker-level resume/cancellation behavior.

### 8. Use FallbackModel and Gateway Routing for Cross-Provider Failover

Prefer [`FallbackModel`](https://ai.pydantic.dev/api/models/fallback/) and Gateway routing groups over app-specific failover logic.

Why:

- This is the upstream abstraction for ordered model failover.
- Gateway also supports routing and multi-provider access with one key, which is a better future home for provider failover than worker-level branching.

What it should replace:

- Future custom app-side provider failover logic.

Replacement status:

- Partial replacement initially.
- Gateway routing groups are the better long-term fit once Anthropic, Gemini, and OpenAI are all enabled behind the gateway.
- The current worker runtime intentionally does not do app-side fallback anymore; the next failover step should use upstream routing/fallback abstractions rather than reintroducing local backend escape hatches.

## Useful Later

### Structured Outputs

Pydantic AI's [Output system](https://ai.pydantic.dev/output/) is a strong fit for workflows that currently parse semi-structured text responses.

Likely future use:

- Selected report metadata extraction
- Bulk-analysis summary metadata or manifest generation
- Provider-agnostic structured intermediate artifacts

Why not first:

- The current app is still mostly prompt-to-markdown, not prompt-to-schema.

Adopt when:

- A worker currently depends on brittle text formatting or post-hoc parsing that could be replaced with `output_type`, `NativeOutput`, `ToolOutput`, or `PromptedOutput`.

### Output Validators and Validation Context

Output validators described in the [Output docs](https://ai.pydantic.dev/output/) are useful when the app needs the model to retry until structured output meets domain rules.

Likely future use:

- Citation metadata checks
- Report section schema validation
- Guardrails around model-produced structured summaries

Why not first:

- Most current workflows are still freeform markdown generation.

### Dependencies and RunContext

The [Dependencies](https://ai.pydantic.dev/dependencies/) system is useful when a run needs injected runtime services or context, including test overrides via `Agent.override`.

Likely future use:

- Standardized injection of project metadata, settings, evidence stores, or shared HTTP clients

Why not first:

- The current worker objects already own most runtime dependencies directly.

Adopt when:

- A workflow is moved from custom imperative workers toward a more declarative Pydantic AI execution layer.

### Toolsets and MCP Tool Integration

The [Toolsets](https://ai.pydantic.dev/toolsets/) and [Function Tools](https://ai.pydantic.dev/tools/) docs are useful if the app later adds agentic lookup or external service tools.

Likely future use:

- Internal knowledge tools
- Controlled MCP integrations
- Reusable task-specific tool collections

Why not first:

- The current app is document-processing first, not tool-calling first.

### Testing Utilities

The [Testing guide](https://ai.pydantic.dev/testing/) provides `TestModel`, `FunctionModel`, `Agent.override`, `capture_run_messages`, and `ALLOW_MODEL_REQUESTS`.

Likely future use:

- Replacing some bespoke model doubles as more logic moves onto native Pydantic AI surfaces
- Improving behavioral assertions around prompts, tool use, and output contracts

Why not first:

- The app still routes most logic through its own worker/backend abstraction rather than direct agent/model usage.

### Pydantic Evals

[Pydantic Evals](https://ai.pydantic.dev/evals/) and [span-based evaluation](https://ai.pydantic.dev/evals/evaluators/span-based/) are strong future fits for regression testing prompt and orchestration behavior.

Likely future use:

- Regression coverage for report quality and bulk-analysis behaviors
- Trace-based checks that verify retries, tool usage, or path selection rather than just output text

Why not first:

- It adds a second testing/eval framework before the core execution path is fully standardized.

Adopt when:

- The app has stable model-level instrumentation and a narrower set of prompt/execution paths worth regression-evaluating.

### Durable Execution

The [durable execution overview](https://ai.pydantic.dev/durable_execution/overview/) is worth revisiting only if the app moves beyond the current desktop/file-checkpoint model.

Likely future use:

- Long-running cloud or distributed workflows
- Human-in-the-loop or resumable multi-step processes outside the current local-worker model

Why not first:

- The current app is a desktop Qt application with file-based checkpoints and resumable worker manifests.

### Pydantic Graph

Pydantic Graph is worth considering later only if the app evolves from worker pipelines into explicit graph-defined orchestration.

Likely future use:

- Formalizing multi-step orchestration once the runtime model is more agentic or service-oriented

Why not first:

- Current worker flows are simpler than a full graph runtime and are already checkpointed at the file/process level.

## Not a Fit Right Now

### Provider-Managed Built-In Tools Are Not a Replacement for the Citation Pipeline

The [Built-in Tools](https://ai.pydantic.dev/builtin-tools/) docs describe provider-executed tools such as `FileSearchTool`, `WebSearchTool`, and managed RAG/file search features.

Do not use these as a replacement for the current citation/evidence system:

- The app depends on Azure Document Intelligence output, deterministic page geometry, and explicit evidence ledgers.
- Provider-managed file search is not equivalent to the current provenance and citation requirements.

This means built-in tools may be useful for future auxiliary features, but not as a replacement for the current document ingestion and citation pipeline.

### Durable Execution Is Not the First Move for a Desktop App

Durable execution is valuable, but it is not the right first simplification step for the current architecture.

Do not replace:

- file-based checkpointing
- worker manifests
- local resume logic

until there is a product decision to move toward service-hosted, distributed, or human-in-the-loop orchestration.

### Model-Level Telemetry Does Not Replace Domain Workflow Spans

Use both:

- model-level instrumentation for LLM request/tool/usage visibility
- app-level spans for bulk/report/conversion workflow stages and job context

Do not remove current worker/stage observability just because Pydantic AI can emit model spans.

## Provider Support and Caveats

### Confirmed Current Baseline

- Installed locally: `pydantic-ai-slim 1.67.0`
- Confirmed current app providers: Anthropic, Gemini, OpenAI, Azure OpenAI, Anthropic Bedrock

### Real Token Preflight

Confirmed in the installed version:

- `AnthropicModel.count_tokens(...)`: supported
- `GoogleModel.count_tokens(...)`: supported
- `BedrockConverseModel.count_tokens(...)`: supported
- `OpenAIChatModel.count_tokens(...)`: not implemented for pre-request counting

Implication:

- Real preflight token counting can be standardized for Anthropic, Gemini, and Bedrock now.
- OpenAI and Azure OpenAI should use the same abstraction, but must still fall back for this specific feature until upstream support exists.

### Reasoning / Thinking Support Is Not Yet Uniform

Pydantic AI already provides a more uniform reasoning representation through [`ModelResponse.parts`](https://ai.pydantic.dev/api/messages/) and [`ThinkingPart`](https://ai.pydantic.dev/api/messages/), which is the main reason the worker backend should stop flattening responses.

But reasoning enablement is still provider-specific in the current upstream docs:

- Anthropic uses `anthropic_thinking`
- Google uses `google_thinking_config`
- OpenAI Responses uses `openai_reasoning_effort` and `openai_reasoning_summary`

Implication:

- the app should keep `use_reasoning` as a product-level intent flag
- provider-specific reasoning settings should be centralized in the backend
- the app should not expect a single provider-agnostic reasoning-mode switch yet

Roadmap reading:

- There is no explicit upstream roadmap commitment to one provider-agnostic reasoning-mode API.
- The direction appears to be toward more uniform reasoning representation, not fully uniform reasoning configuration semantics.
- Reasoning-related upstream issues are best read as directional signals rather than promises.

### Multi-Provider Model Naming

Use documented model naming conventions:

- Direct provider form: `<provider>:<model>`
- Gateway form: `gateway/<provider>:<model>`

Examples:

- `anthropic:claude-sonnet-4-5`
- `google-vertex:gemini-3-flash-preview`
- `openai:gpt-5.2`
- `gateway/anthropic:claude-sonnet-4-5`
- `gateway/openai:gpt-5.2`
- `gateway/bedrock:amazon.nova-micro-v1:0`

### Gateway Scope

Gateway is a strong fit for:

- single-key multi-provider access
- centralized routing and failover
- centralized cost and access management
- unified observability surface

But Gateway does not remove the need for app-level policy around:

- document chunking
- local checkpoints
- citation evidence handling
- project/workspace-specific workflow state

## Proposed Migration Order

1. Replace low-level custom `Agent(...)` invocation wrappers with Pydantic AI `direct` requests or a thinner backend wrapper built on `direct`.
2. Centralize provider/model resolution around documented model strings and Gateway prefixes.
3. Standardize token preflight on backend-supported `count_tokens(...)` plus `UsageLimits`, while keeping app-specific chunk planning.
4. Add model-level concurrency limits, standardized retry transports, and provider/gateway failover through upstream abstractions.
5. Replace flattened worker response handling with full `ModelResponse` passthrough and exception-based failure handling.
6. Adopt structured outputs and validators for selected workflows that currently depend on fragile text parsing.
7. Add Pydantic Evals and span-based regression coverage after model-level instrumentation is stable.
8. Revisit durable execution only if the product moves beyond the current desktop/file-checkpoint model.

Status against this sequence:

- Steps 1-5 are largely complete for worker execution, except for the failover portion of step 4.
- Structured outputs are the next major adoption area, but they should follow targeted workflow selection rather than a broad rewrite.
- The biggest remaining platform gaps are failover, deeper instrumentation follow-through, and migration of non-worker legacy code.

## Sources

- [Pydantic AI home](https://ai.pydantic.dev/)
- [Direct Model Requests](https://ai.pydantic.dev/direct/)
- [Models overview](https://ai.pydantic.dev/models/overview/)
- [Gateway overview](https://ai.pydantic.dev/gateway/)
- [Providers API](https://ai.pydantic.dev/api/providers/)
- [Messages API](https://ai.pydantic.dev/api/messages/)
- [Usage / UsageLimits](https://ai.pydantic.dev/api/usage/)
- [Concurrency](https://ai.pydantic.dev/api/concurrency/)
- [FallbackModel](https://ai.pydantic.dev/api/models/fallback/)
- [Thinking](https://ai.pydantic.dev/thinking/)
- [Output](https://ai.pydantic.dev/output/)
- [Dependencies](https://ai.pydantic.dev/dependencies/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
- [Function Tools](https://ai.pydantic.dev/tools/)
- [Built-in Tools](https://ai.pydantic.dev/builtin-tools/)
- [Testing](https://ai.pydantic.dev/testing/)
- [HTTP Request Retries](https://ai.pydantic.dev/retries/)
- [Changelog](https://ai.pydantic.dev/changelog/)
- [Logfire / Debugging and Monitoring](https://ai.pydantic.dev/logfire/)
- [Instrumented Models API](https://ai.pydantic.dev/api/models/instrumented/)
- [Pydantic Evals](https://ai.pydantic.dev/evals/)
- [Span-Based Evaluation](https://ai.pydantic.dev/evals/evaluators/span-based/)
- [Durable Execution Overview](https://ai.pydantic.dev/durable_execution/overview/)
- [Roadmap issue](https://github.com/pydantic/pydantic-ai/issues/913)
- [Reasoning response support](https://github.com/pydantic/pydantic-ai/issues/907)
- [OpenAI encrypted thinking tokens](https://github.com/pydantic/pydantic-ai/issues/2127)
- [ThinkingPart support for OpenAI chat models](https://github.com/pydantic/pydantic-ai/issues/2701)
