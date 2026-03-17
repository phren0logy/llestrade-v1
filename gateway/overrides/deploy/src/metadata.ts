import { findProvider, type Provider as PricingProvider } from '@pydantic/genai-prices'
import type { ApiKeyInfo, GatewayOptions, ProviderProxy } from '@pydantic/ai-gateway'
import { hash } from './db'

const MODELS_CACHE_TTL_MS = 60 * 60 * 1000
const PRICING_SNAPSHOT_TTL_MS = 60 * 60 * 1000
const PRICING_SNAPSHOT_URL = 'https://raw.githubusercontent.com/pydantic/genai-prices/refs/heads/main/prices/data.json'
const ANTHROPIC_VERSION = '2023-06-01'
const OPENAI_GENERATIVE_PREFIXES = ['gpt-', 'o1', 'o3', 'o4', 'chatgpt-4o-latest', 'codex-mini'] as const
const OPENAI_EXCLUDED_TOKENS = [
  'audio',
  'image',
  'tts',
  'transcribe',
  'realtime',
  'embedding',
  'moderation',
  'search',
] as const
const GEMINI_EXCLUDED_TOKENS = ['image', 'tts', 'embedding', 'veo', 'imagen', 'aqa', 'learnlm'] as const
const BEDROCK_PROVIDER_LABEL = 'AWS Bedrock (Claude)'
const BEDROCK_CLAUDE_MODELS = [
  {
    modelId: 'us.anthropic.claude-sonnet-4-6',
    displayName: 'Claude Sonnet 4.6',
    anthropicModelId: 'claude-sonnet-4-6',
  },
  {
    modelId: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    displayName: 'Claude Sonnet 4.5',
    anthropicModelId: 'claude-sonnet-4-5',
  },
  {
    modelId: 'us.anthropic.claude-opus-4-6-v1',
    displayName: 'Claude Opus 4.6',
    anthropicModelId: 'claude-opus-4-6',
  },
  {
    modelId: 'us.anthropic.claude-opus-4-1-20250805-v1:0',
    displayName: 'Claude Opus 4.1',
    anthropicModelId: 'claude-opus-4-1-20250805',
  },
] as const

type LifecycleStatus = 'stable' | 'preview' | 'deprecated' | 'retired' | 'unknown'
type ReasoningControl = 'toggle' | 'effort' | 'level' | 'budget'
type SourceMarker = 'provider-native' | 'provider-heuristic' | 'genai-prices' | 'unknown'

interface ReasoningCapabilities {
  source: SourceMarker
  exact: boolean
  supports_reasoning_controls: boolean | null
  can_disable_reasoning: boolean | null
  controls: ReasoningControl[]
  notes?: string
}

interface ModelSources {
  availability: SourceMarker
  display_name: SourceMarker
  lifecycle_status: SourceMarker
  context_window: SourceMarker
  max_output_tokens: SourceMarker
  pricing: SourceMarker
  reasoning_capabilities: SourceMarker
}

export interface MetadataModelRecord {
  provider_id: string
  route: string
  upstream_provider_id: string
  model_id: string
  display_name: string
  lifecycle_status: LifecycleStatus
  context_window: number | null
  max_output_tokens: number | null
  pricing_input_per_million: number | null
  pricing_cached_input_per_million: number | null
  pricing_output_per_million: number | null
  pricing_tiered: boolean
  reasoning_capabilities: ReasoningCapabilities
  sources: ModelSources
}

export interface MetadataCatalogProviderRecord {
  provider_id: string
  route: string
  upstream_provider_id: string
  label: string
  models: MetadataModelRecord[]
}

interface CacheEntry {
  expiresAt: number
  value?: MetadataModelRecord[]
  pending?: Promise<MetadataModelRecord[]>
}

interface PricingMetadata {
  contextWindow: number | null
  maxOutputTokens: number | null
  inputPerMillion: number | null
  cachedInputPerMillion: number | null
  outputPerMillion: number | null
  tiered: boolean
  source: SourceMarker
}

type PricingSnapshotResponse = PricingProvider[] | { providers?: PricingProvider[] }

interface OpenAIModelResponse {
  id: string
  created?: number
  owned_by?: string
}

interface OpenAIListResponse {
  data?: OpenAIModelResponse[]
}

interface AnthropicModelResponse {
  id: string
  display_name?: string
  type?: string
}

interface AnthropicListResponse {
  data?: AnthropicModelResponse[]
}

interface VertexModelResponse {
  name?: string
  versionId?: string
  displayName?: string
  launchStage?: string
  versionState?: string
  inputTokenLimit?: number
  outputTokenLimit?: number
  supportedActions?: string[]
}

interface VertexListResponse {
  publisherModels?: VertexModelResponse[]
}

interface ServiceAccount {
  client_email: string
  private_key: string
}

const metadataCache = new Map<string, CacheEntry>()
const googleTokenCache = new Map<string, { expiresAt: number; token: string }>()
let pricingSnapshotCache:
  | {
      expiresAt: number
      value?: PricingProvider[]
      pending?: Promise<PricingProvider[] | null>
    }
  | undefined

export function resetMetadataCachesForTest(): void {
  metadataCache.clear()
  googleTokenCache.clear()
  pricingSnapshotCache = undefined
}

export async function metadataCatalog(request: Request, options: GatewayOptions): Promise<Response> {
  const authResult = await authenticateGatewayAppKey(request, options)
  if (authResult instanceof Response) {
    return authResult
  }

  try {
    const providers = await loadCatalogForApiKey(authResult)
    return new Response(JSON.stringify({ providers }, null, 2), {
      headers: { 'Content-Type': 'application/json' },
    })
  } catch (error) {
    return new Response(`Failed to load metadata catalog: ${(error as Error).message}`, {
      status: 502,
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    })
  }
}

export async function metadataModels(
  request: Request,
  url: URL,
  options: GatewayOptions,
): Promise<Response> {
  const authResult = await authenticateGatewayAppKey(request, options)
  if (authResult instanceof Response) {
    return authResult
  }

  const routeResult = resolveMetadataRoute(url, authResult)
  if ('response' in routeResult) {
    return routeResult.response
  }

  try {
    const cacheKey = await buildCacheKey(routeResult.route, authResult.id, routeResult.providerProxies)
    const models = await getCachedModels(cacheKey, () => loadModelsForRoute(routeResult))
    return new Response(JSON.stringify(models, null, 2), {
      headers: { 'Content-Type': 'application/json' },
    })
  } catch (error) {
    return new Response(`Failed to load metadata: ${(error as Error).message}`, {
      status: 502,
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    })
  }
}

async function authenticateGatewayAppKey(request: Request, options: GatewayOptions): Promise<ApiKeyInfo | Response> {
  const key = getAuthKey(request)
  if (key instanceof Response) {
    return key
  }
  if (key.length > 200) {
    return new Response('Unauthorized - Key too long', { status: 401 })
  }

  const apiKeyInfo = await options.keysDb.getApiKey(key)
  if (!apiKeyInfo) {
    return new Response('Unauthorized - Key not found', { status: 401 })
  }
  if (apiKeyInfo.status !== 'active') {
    return new Response(`Unauthorized - Key ${apiKeyInfo.status}`, { status: 403 })
  }
  return apiKeyInfo
}

function getAuthKey(request: Request): string | Response {
  const authorization = request.headers.get('authorization')
  const xApiKey = request.headers.get('x-api-key')
  const normalize = (value: string | null): string | null =>
    value ? (value.toLowerCase().startsWith('bearer ') ? value.slice(7) : value) : null

  const authKey = normalize(authorization)
  const apiKey = normalize(xApiKey)
  if (authKey && apiKey) {
    return new Response('Unauthorized - Both Authorization and X-API-Key headers are set, use only one', {
      status: 401,
    })
  }
  return authKey ?? apiKey ?? new Response('Unauthorized - Missing Authorization Header', { status: 401 })
}

function resolveMetadataRoute(
  url: URL,
  apiKeyInfo: ApiKeyInfo,
):
  | { route: string; requestedProviderId: string; providerProxies: (ProviderProxy & { key: string })[] }
  | { response: Response } {
  const routeParam = url.searchParams.get('route')?.trim()
  const providerParam = url.searchParams.get('provider')?.trim()
  const requested = routeParam || providerParam
  if (!requested) {
    return { response: new Response('Missing route or provider query parameter', { status: 400 }) }
  }

  const providerProxyMapping = Object.fromEntries(apiKeyInfo.providers.map((provider) => [provider.key, provider]))
  const routingGroup = apiKeyInfo.routingGroups?.[requested]
  if (routingGroup) {
    const providerProxies = routingGroup
      .map((item) => providerProxyMapping[item.key])
      .filter((provider): provider is ProviderProxy & { key: string } => Boolean(provider))
    if (providerProxies.length === 0) {
      return { response: new Response(`No providers available for route ${requested}`, { status: 404 }) }
    }
    return { route: requested, requestedProviderId: providerParam ?? requested, providerProxies }
  }

  if (providerProxyMapping[requested]) {
    return {
      route: requested,
      requestedProviderId: providerParam ?? requested,
      providerProxies: [providerProxyMapping[requested]!],
    }
  }

  const byProviderId = apiKeyInfo.providers.filter((provider) => provider.providerId === requested)
  if (byProviderId.length > 0) {
    return {
      route: requested,
      requestedProviderId: providerParam ?? requested,
      providerProxies: byProviderId,
    }
  }

  const supported = [...new Set([...Object.keys(providerProxyMapping), ...Object.keys(apiKeyInfo.routingGroups ?? {})])]
    .sort()
    .join(', ')
  return {
    response: new Response(`Route not found: ${requested}. Supported values: ${supported}`, { status: 404 }),
  }
}

async function buildCacheKey(
  route: string,
  apiKeyId: number,
  providerProxies: (ProviderProxy & { key: string })[],
): Promise<string> {
  const parts = await Promise.all(
    providerProxies.map(async (provider) => {
      const credentialsHash = await hash(provider.credentials)
      return `${provider.key}:${provider.providerId}:${provider.baseUrl}:${credentialsHash}`
    }),
  )
  return `metadata:${apiKeyId}:${route}:${parts.sort().join('|')}`
}

async function getCachedModels(cacheKey: string, loader: () => Promise<MetadataModelRecord[]>): Promise<MetadataModelRecord[]> {
  const now = Date.now()
  const cached = metadataCache.get(cacheKey)
  if (cached?.value && cached.expiresAt > now) {
    return cached.value
  }
  if (cached?.pending) {
    return cached.pending
  }

  const pending = loader()
    .then((value) => {
      metadataCache.set(cacheKey, { expiresAt: Date.now() + MODELS_CACHE_TTL_MS, value })
      return value
    })
    .finally(() => {
      const current = metadataCache.get(cacheKey)
      if (current?.pending) {
        metadataCache.set(cacheKey, { expiresAt: current.expiresAt, value: current.value })
      }
    })

  metadataCache.set(cacheKey, {
    expiresAt: now + MODELS_CACHE_TTL_MS,
    value: cached?.value,
    pending,
  })
  return pending
}

async function loadCatalogForApiKey(apiKeyInfo: ApiKeyInfo): Promise<MetadataCatalogProviderRecord[]> {
  const providerProxies = apiKeyInfo.providers.filter((provider): provider is ProviderProxy & { key: string } => {
    return typeof provider.key === 'string' && supportsMetadataProvider(provider.providerId)
  })

  const settled = await Promise.allSettled(
    providerProxies.map((providerProxy) => loadCatalogProvider(apiKeyInfo.id, providerProxy)),
  )

  const providers: MetadataCatalogProviderRecord[] = []
  const errors: string[] = []
  for (const result of settled) {
    if (result.status === 'fulfilled') {
      if (result.value !== null) {
        providers.push(result.value)
      }
    } else {
      errors.push(result.reason instanceof Error ? result.reason.message : String(result.reason))
    }
  }

  if (providers.length === 0 && errors.length > 0) {
    throw new Error(errors.join('; '))
  }

  return providers.sort(compareCatalogProviders)
}

async function loadCatalogProvider(
  apiKeyId: number,
  providerProxy: ProviderProxy & { key: string },
): Promise<MetadataCatalogProviderRecord | null> {
  const route = providerProxy.key
  const providerId = canonicalProviderId(providerProxy)
  const cacheKey = await buildCacheKey(route, apiKeyId, [providerProxy])
  const models = await getCachedModels(cacheKey, () =>
    loadModelsForRoute({
      route,
      requestedProviderId: providerId,
      providerProxies: [providerProxy],
    }),
  )

  if (models.length === 0) {
    return null
  }

  return {
    provider_id: providerId,
    route,
    upstream_provider_id: providerProxy.providerId,
    label: formatProviderLabel(providerProxy),
    models,
  }
}

async function loadModelsForRoute(routeResult: {
  route: string
  requestedProviderId: string
  providerProxies: (ProviderProxy & { key: string })[]
}): Promise<MetadataModelRecord[]> {
  const settled = await Promise.allSettled(
    routeResult.providerProxies.map((providerProxy) =>
      loadModelsForProvider(routeResult.route, routeResult.requestedProviderId, providerProxy),
    ),
  )

  const models: MetadataModelRecord[] = []
  const errors: string[] = []
  for (const result of settled) {
    if (result.status === 'fulfilled') {
      models.push(...result.value)
    } else {
      errors.push(result.reason instanceof Error ? result.reason.message : String(result.reason))
    }
  }

  if (models.length === 0 && errors.length > 0) {
    throw new Error(errors.join('; '))
  }

  const deduped = new Map<string, MetadataModelRecord>()
  for (const model of models) {
    const key = `${model.provider_id}:${model.upstream_provider_id}:${model.model_id}`
    if (!deduped.has(key)) {
      deduped.set(key, model)
    }
  }
  return Array.from(deduped.values()).sort(compareModelRecords)
}

async function loadModelsForProvider(
  route: string,
  requestedProviderId: string,
  providerProxy: ProviderProxy & { key: string },
): Promise<MetadataModelRecord[]> {
  switch (providerProxy.providerId) {
    case 'openai':
      return await loadOpenAIModels(route, requestedProviderId, providerProxy)
    case 'anthropic':
      return await loadAnthropicModels(route, requestedProviderId, providerProxy)
    case 'bedrock':
      return await loadBedrockModels(route, requestedProviderId, providerProxy)
    case 'google-vertex':
      return await loadGoogleVertexModels(route, requestedProviderId, providerProxy)
    default:
      return []
  }
}

function supportsMetadataProvider(providerId: string): boolean {
  return providerId === 'openai' || providerId === 'anthropic' || providerId === 'google-vertex' || providerId === 'bedrock'
}

async function loadOpenAIModels(
  route: string,
  requestedProviderId: string,
  providerProxy: ProviderProxy & { key: string },
): Promise<MetadataModelRecord[]> {
  const pricingProvider = await resolvePricingProvider('openai')
  const baseUrl = stripTrailingSlash(providerProxy.baseUrl)
  const modelsUrl = baseUrl.endsWith('/v1') ? `${baseUrl}/models` : `${baseUrl}/v1/models`
  const response = await fetch(modelsUrl, {
    headers: { Authorization: `Bearer ${providerProxy.credentials}` },
    signal: AbortSignal.timeout(20_000),
  })
  if (!response.ok) {
    throw new Error(`OpenAI metadata request failed (${response.status})`)
  }
  const payload = (await response.json()) as OpenAIListResponse
  const models = Array.isArray(payload.data) ? payload.data : []
  return models
    .filter((model) => isOpenAIGenerativeModel(model.id))
    .map((model) =>
      buildNormalizedRecord({
        route,
        requestedProviderId,
        upstreamProviderId: providerProxy.providerId,
        modelId: model.id,
        displayName: normalizeDisplayName(model.id),
        lifecycleStatus: inferLifecycleStatus(model.id),
        providerContextWindow: null,
        providerMaxOutputTokens: null,
        pricingProvider,
        reasoning: inferReasoningCapabilities(providerProxy.providerId, model.id),
      }),
    )
}

async function loadAnthropicModels(
  route: string,
  requestedProviderId: string,
  providerProxy: ProviderProxy & { key: string },
): Promise<MetadataModelRecord[]> {
  const pricingProvider = await resolvePricingProvider('anthropic')
  const response = await fetch(`${stripTrailingSlash(providerProxy.baseUrl)}/v1/models`, {
    headers: {
      'x-api-key': providerProxy.credentials,
      'anthropic-version': ANTHROPIC_VERSION,
    },
    signal: AbortSignal.timeout(20_000),
  })
  if (!response.ok) {
    throw new Error(`Anthropic metadata request failed (${response.status})`)
  }
  const payload = (await response.json()) as AnthropicListResponse
  const models = Array.isArray(payload.data) ? payload.data : []
  return models.map((model) =>
    buildNormalizedRecord({
      route,
      requestedProviderId,
      upstreamProviderId: providerProxy.providerId,
      modelId: model.id,
      displayName: model.display_name?.trim() || normalizeDisplayName(model.id),
      lifecycleStatus: inferLifecycleStatus(model.id),
      providerContextWindow: null,
      providerMaxOutputTokens: null,
      pricingProvider,
      reasoning: inferReasoningCapabilities(providerProxy.providerId, model.id),
    }),
  )
}

async function loadGoogleVertexModels(
  route: string,
  requestedProviderId: string,
  providerProxy: ProviderProxy & { key: string },
): Promise<MetadataModelRecord[]> {
  const pricingProvider = await resolvePricingProvider('google')
  const accessToken = await getGoogleAccessToken(providerProxy.credentials)
  const response = await fetch(
    `${stripTrailingSlash(providerProxy.baseUrl)}/v1beta1/publishers/google/models?pageSize=1000&listAllVersions=true&view=FULL`,
    {
      headers: { Authorization: `Bearer ${accessToken}` },
      signal: AbortSignal.timeout(20_000),
    },
  )
  if (!response.ok) {
    throw new Error(`Google Vertex metadata request failed (${response.status})`)
  }
  const payload = (await response.json()) as VertexListResponse
  const models = Array.isArray(payload.publisherModels) ? payload.publisherModels : []
  return models
    .filter((model) => isVertexGenerativeModel(model))
    .map((model) => {
      const modelId = extractVertexModelId(model.name)
      return buildNormalizedRecord({
        route,
        requestedProviderId,
        upstreamProviderId: providerProxy.providerId,
        modelId,
        displayName: model.displayName?.trim() || normalizeDisplayName(modelId),
        lifecycleStatus: inferVertexLifecycle(model),
        providerContextWindow: numberOrNull(model.inputTokenLimit),
        providerMaxOutputTokens: numberOrNull(model.outputTokenLimit),
        pricingProvider,
        reasoning: inferReasoningCapabilities(providerProxy.providerId, modelId),
      })
    })
}

async function loadBedrockModels(
  route: string,
  requestedProviderId: string,
  providerProxy: ProviderProxy & { key: string },
): Promise<MetadataModelRecord[]> {
  const pricingProvider = await resolvePricingProvider('anthropic')
  return BEDROCK_CLAUDE_MODELS.map((model) =>
    buildNormalizedRecord({
      route,
      requestedProviderId,
      upstreamProviderId: providerProxy.providerId,
      modelId: model.modelId,
      displayName: model.displayName,
      lifecycleStatus: inferLifecycleStatus(model.modelId),
      providerContextWindow: null,
      providerMaxOutputTokens: null,
      pricingProvider,
      pricingModelId: model.anthropicModelId,
      availabilitySource: 'provider-heuristic',
      displayNameSource: 'provider-heuristic',
      lifecycleSource: 'provider-heuristic',
      reasoning: inferReasoningCapabilities(providerProxy.providerId, model.modelId),
    }),
  )
}

function buildNormalizedRecord(input: {
  route: string
  requestedProviderId: string
  upstreamProviderId: string
  modelId: string
  displayName: string
  lifecycleStatus: LifecycleStatus
  providerContextWindow: number | null
  providerMaxOutputTokens: number | null
  pricingProvider: PricingProvider | null
  pricingModelId?: string
  availabilitySource?: SourceMarker
  displayNameSource?: SourceMarker
  lifecycleSource?: SourceMarker
  reasoning: ReasoningCapabilities
}): MetadataModelRecord {
  const pricing = resolvePricingMetadata(input.pricingProvider, input.pricingModelId ?? input.modelId)
  return {
    provider_id: input.requestedProviderId,
    route: input.route,
    upstream_provider_id: input.upstreamProviderId,
    model_id: input.modelId,
    display_name: input.displayName,
    lifecycle_status: input.lifecycleStatus,
    context_window: input.providerContextWindow ?? pricing.contextWindow,
    max_output_tokens: input.providerMaxOutputTokens ?? pricing.maxOutputTokens,
    pricing_input_per_million: pricing.inputPerMillion,
    pricing_cached_input_per_million: pricing.cachedInputPerMillion,
    pricing_output_per_million: pricing.outputPerMillion,
    pricing_tiered: pricing.tiered,
    reasoning_capabilities: input.reasoning,
    sources: {
      availability: input.availabilitySource ?? 'provider-native',
      display_name: input.displayNameSource ?? 'provider-native',
      lifecycle_status:
        input.lifecycleSource ?? (input.lifecycleStatus === 'unknown' ? 'provider-heuristic' : 'provider-native'),
      context_window: input.providerContextWindow !== null ? 'provider-native' : pricing.source,
      max_output_tokens: input.providerMaxOutputTokens !== null ? 'provider-native' : pricing.source,
      pricing: pricing.source,
      reasoning_capabilities: input.reasoning.source,
    },
  }
}

async function resolvePricingProvider(providerId: string): Promise<PricingProvider | null> {
  const providers = await getPricingProviders()
  const liveProvider = providers?.find((provider) => provider.id === providerId)
  if (liveProvider) {
    return liveProvider
  }
  return findProvider({ providerId })
}

async function getPricingProviders(): Promise<PricingProvider[] | null> {
  const now = Date.now()
  if (pricingSnapshotCache?.value && pricingSnapshotCache.expiresAt > now) {
    return pricingSnapshotCache.value
  }
  if (pricingSnapshotCache?.pending) {
    return pricingSnapshotCache.pending
  }

  const previousValue = pricingSnapshotCache?.value ?? []
  const pending = fetchPricingProviders()
    .then((providers) => {
      if (providers && providers.length > 0) {
        pricingSnapshotCache = {
          expiresAt: Date.now() + PRICING_SNAPSHOT_TTL_MS,
          value: providers,
        }
        return providers
      }
      pricingSnapshotCache = {
        expiresAt: Date.now() + PRICING_SNAPSHOT_TTL_MS,
        value: previousValue,
      }
      return previousValue
    })
    .catch(() => {
      pricingSnapshotCache = {
        expiresAt: Date.now() + PRICING_SNAPSHOT_TTL_MS,
        value: previousValue,
      }
      return previousValue
    })
    .finally(() => {
      if (pricingSnapshotCache?.pending === pending) {
        pricingSnapshotCache = {
          expiresAt: pricingSnapshotCache.expiresAt,
          value: pricingSnapshotCache.value,
        }
      }
    })

  pricingSnapshotCache = {
    expiresAt: now + PRICING_SNAPSHOT_TTL_MS,
    value: previousValue,
    pending,
  }
  return pending
}

async function fetchPricingProviders(): Promise<PricingProvider[] | null> {
  const response = await fetch(PRICING_SNAPSHOT_URL, {
    signal: AbortSignal.timeout(20_000),
  })
  if (!response.ok) {
    throw new Error(`Pricing snapshot request failed (${response.status})`)
  }
  const payload = (await response.json()) as PricingSnapshotResponse
  if (Array.isArray(payload)) {
    return payload
  }
  return Array.isArray(payload.providers) ? payload.providers : null
}

function resolvePricingMetadata(provider: PricingProvider | null, modelId: string): PricingMetadata {
  if (!provider) {
    return emptyPricingMetadata()
  }

  const model = matchPricingModel(provider, modelId)
  if (!model) {
    return emptyPricingMetadata()
  }

  const contextWindow = numberOrNull((model as { context_window?: unknown }).context_window)
  const prices = (model as { prices?: Record<string, unknown> }).prices ?? {}
  const inputPrice = normalizePriceValue(prices.input_mtok)
  const cachedInputPrice = normalizePriceValue(prices.cache_read_mtok)
  const outputPrice = normalizePriceValue(prices.output_mtok)
  const tiered = isTieredPrice(prices.input_mtok) || isTieredPrice(prices.cache_read_mtok) || isTieredPrice(prices.output_mtok)

  return {
    contextWindow,
    maxOutputTokens: null,
    inputPerMillion: inputPrice,
    cachedInputPerMillion: cachedInputPrice,
    outputPerMillion: outputPrice,
    tiered,
    source: 'genai-prices',
  }
}

function emptyPricingMetadata(): PricingMetadata {
  return {
    contextWindow: null,
    maxOutputTokens: null,
    inputPerMillion: null,
    cachedInputPerMillion: null,
    outputPerMillion: null,
    tiered: false,
    source: 'unknown',
  }
}

function matchPricingModel(provider: PricingProvider, modelId: string) {
  const exact = provider.models.find((model) => model.id === modelId)
  if (exact) {
    return exact
  }

  const normalized = modelId.toLowerCase()
  const prefix = provider.models
    .filter((model) => normalized.startsWith(model.id.toLowerCase()))
    .sort((left, right) => right.id.length - left.id.length)[0]
  if (prefix) {
    return prefix
  }

  return provider.models.find((model) => model.id.toLowerCase() === normalized)
}

function normalizePriceValue(value: unknown): number | null {
  if (typeof value === 'number') {
    return value
  }
  if (typeof value === 'object' && value && 'base' in value && typeof value.base === 'number') {
    return value.base
  }
  return null
}

function isTieredPrice(value: unknown): boolean {
  return typeof value === 'object' && value !== null && Array.isArray((value as { tiers?: unknown[] }).tiers)
}

function inferReasoningCapabilities(providerId: string, modelId: string): ReasoningCapabilities {
  const normalized = modelId.toLowerCase()
  if (providerId === 'anthropic' || providerId === 'bedrock') {
    return {
      source: 'provider-heuristic',
      exact: false,
      supports_reasoning_controls: true,
      can_disable_reasoning: true,
      controls: ['toggle', 'budget'],
      notes: 'Anthropic thinking controls inferred from model family.',
    }
  }
  if (providerId === 'openai') {
    if (normalized.startsWith('gpt-5') || normalized.startsWith('o1') || normalized.startsWith('o3') || normalized.startsWith('o4')) {
      return {
        source: 'provider-heuristic',
        exact: false,
        supports_reasoning_controls: true,
        can_disable_reasoning: null,
        controls: ['effort'],
        notes: 'OpenAI reasoning effort inferred from model family.',
      }
    }
    return {
      source: 'provider-heuristic',
      exact: false,
      supports_reasoning_controls: false,
      can_disable_reasoning: true,
      controls: [],
    }
  }
  if (providerId === 'google-vertex') {
    if (normalized.startsWith('gemini-2.5')) {
      return {
        source: 'provider-heuristic',
        exact: false,
        supports_reasoning_controls: true,
        can_disable_reasoning: null,
        controls: ['budget'],
        notes: 'Gemini 2.5 thinking budget inferred from model family.',
      }
    }
    if (normalized.startsWith('gemini-3')) {
      return {
        source: 'provider-heuristic',
        exact: false,
        supports_reasoning_controls: true,
        can_disable_reasoning: null,
        controls: ['level'],
        notes: 'Gemini 3 thinking level inferred from model family.',
      }
    }
  }
  return {
    source: 'unknown',
    exact: false,
    supports_reasoning_controls: null,
    can_disable_reasoning: null,
    controls: [],
  }
}

function isOpenAIGenerativeModel(modelId: string): boolean {
  const normalized = modelId.toLowerCase()
  if (!OPENAI_GENERATIVE_PREFIXES.some((prefix) => normalized.startsWith(prefix))) {
    return false
  }
  return !OPENAI_EXCLUDED_TOKENS.some((token) => normalized.includes(token))
}

function isVertexGenerativeModel(model: VertexModelResponse): boolean {
  const modelId = extractVertexModelId(model.name)
  const normalized = modelId.toLowerCase()
  if (!normalized.startsWith('gemini')) {
    return false
  }
  if (GEMINI_EXCLUDED_TOKENS.some((token) => normalized.includes(token))) {
    return false
  }
  if (Array.isArray(model.supportedActions) && model.supportedActions.length > 0) {
    return model.supportedActions.some((action) =>
      ['generateContent', 'streamGenerateContent', 'countTokens'].includes(action),
    )
  }
  return true
}

function extractVertexModelId(name?: string): string {
  if (!name) {
    return 'unknown'
  }
  const parts = name.split('/')
  return parts[parts.length - 1] || name
}

function inferVertexLifecycle(model: VertexModelResponse): LifecycleStatus {
  const stage = (model.launchStage || '').toUpperCase()
  const state = (model.versionState || '').toUpperCase()
  if (stage.includes('DEPRECATED') || state.includes('DEPRECATED')) {
    return 'deprecated'
  }
  if (stage.includes('EXPERIMENTAL') || stage.includes('PREVIEW') || state.includes('PREVIEW')) {
    return 'preview'
  }
  if (stage.includes('GA') || stage.includes('GENERAL')) {
    return 'stable'
  }
  return inferLifecycleStatus(extractVertexModelId(model.name))
}

function inferLifecycleStatus(modelId: string): LifecycleStatus {
  const normalized = modelId.toLowerCase()
  if (normalized.includes('deprecated')) {
    return 'deprecated'
  }
  if (normalized.includes('retired')) {
    return 'retired'
  }
  if (normalized.includes('preview') || normalized.includes('beta') || normalized.includes('experimental')) {
    return 'preview'
  }
  return 'stable'
}

function normalizeDisplayName(modelId: string): string {
  return modelId
    .split(/[-_.:]/)
    .filter(Boolean)
    .map((part) => (part.length <= 3 ? part.toUpperCase() : part[0]!.toUpperCase() + part.slice(1)))
    .join(' ')
}

function formatProviderLabel(providerProxy: ProviderProxy & { key: string }): string {
  return upstreamProviderLabel(providerProxy.providerId)
}

function canonicalProviderId(providerProxy: ProviderProxy & { key: string }): string {
  switch (providerProxy.providerId) {
    case 'bedrock':
      return 'anthropic_bedrock'
    case 'google-vertex':
      return 'gemini'
    default:
      return providerProxy.providerId
  }
}

function upstreamProviderLabel(providerId: string): string {
  switch (providerId) {
    case 'openai':
      return 'OpenAI'
    case 'anthropic':
      return 'Anthropic'
    case 'bedrock':
      return BEDROCK_PROVIDER_LABEL
    case 'google-vertex':
      return 'Google Gemini'
    default:
      return normalizeDisplayName(providerId)
    }
}

function compareCatalogProviders(left: MetadataCatalogProviderRecord, right: MetadataCatalogProviderRecord): number {
  return left.label.localeCompare(right.label) || left.provider_id.localeCompare(right.provider_id)
}

function compareModelRecords(left: MetadataModelRecord, right: MetadataModelRecord): number {
  const lifecycleOrder: Record<LifecycleStatus, number> = {
    stable: 0,
    preview: 1,
    deprecated: 2,
    retired: 3,
    unknown: 4,
  }
  const lifecycleDelta = lifecycleOrder[left.lifecycle_status] - lifecycleOrder[right.lifecycle_status]
  if (lifecycleDelta !== 0) {
    return lifecycleDelta
  }
  return left.model_id.localeCompare(right.model_id)
}

function stripTrailingSlash(value: string): string {
  return value.endsWith('/') ? value.slice(0, -1) : value
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

async function getGoogleAccessToken(credentials: string): Promise<string> {
  const credentialsHash = await hash(credentials)
  const cached = googleTokenCache.get(credentialsHash)
  if (cached && cached.expiresAt > Date.now()) {
    return cached.token
  }

  const serviceAccount = parseServiceAccount(credentials)
  const jwt = await signGoogleJwt(serviceAccount)
  const body = new URLSearchParams({
    grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
    assertion: jwt,
  })

  const response = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
    signal: AbortSignal.timeout(10_000),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Failed to get GCP access token (${response.status}): ${text}`)
  }
  const payload = (await response.json()) as { access_token?: string; expires_in?: number }
  if (!payload.access_token) {
    throw new Error('GCP access token response did not include access_token')
  }
  googleTokenCache.set(credentialsHash, {
    token: payload.access_token,
    expiresAt: Date.now() + Math.max(60, payload.expires_in ?? 3000) * 1000,
  })
  return payload.access_token
}

function parseServiceAccount(credentials: string): ServiceAccount {
  let payload: unknown
  try {
    payload = JSON.parse(credentials)
  } catch (error) {
    throw new Error(`provider credentials are not valid JSON: ${(error as Error).message}`)
  }
  if (!payload || typeof payload !== 'object') {
    throw new Error('provider credentials must be a JSON object')
  }
  const clientEmail = (payload as { client_email?: unknown }).client_email
  const privateKey = (payload as { private_key?: unknown }).private_key
  if (typeof clientEmail !== 'string' || typeof privateKey !== 'string') {
    throw new Error('provider credentials must include client_email and private_key')
  }
  return { client_email: clientEmail, private_key: privateKey }
}

async function signGoogleJwt(serviceAccount: ServiceAccount): Promise<string> {
  const now = Math.floor(Date.now() / 1000)
  const payload = {
    iss: serviceAccount.client_email,
    scope: 'https://www.googleapis.com/auth/cloud-platform',
    aud: 'https://oauth2.googleapis.com/token',
    exp: now + 3600,
    iat: now,
  }

  const header = base64UrlEncode(JSON.stringify({ alg: 'RS256', typ: 'JWT' }))
  const body = base64UrlEncode(JSON.stringify(payload))
  const signingInput = `${header}.${body}`
  const privateKeyPem = serviceAccount.private_key.replace(/-{5}[A-Z]+ PRIVATE KEY-{5}/g, '').replace(/\s/g, '')
  const privateKeyBytes = Uint8Array.from(atob(privateKeyPem), (char) => char.charCodeAt(0))
  const key = await crypto.subtle.importKey(
    'pkcs8',
    privateKeyBytes,
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
    false,
    ['sign'],
  )
  const signature = await crypto.subtle.sign('RSASSA-PKCS1-v1_5', key, new TextEncoder().encode(signingInput))
  return `${signingInput}.${base64UrlEncodeBuffer(signature)}`
}

function base64UrlEncode(value: string): string {
  return btoa(value).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '')
}

function base64UrlEncodeBuffer(value: ArrayBuffer): string {
  return base64UrlEncode(String.fromCharCode(...new Uint8Array(value)))
}
