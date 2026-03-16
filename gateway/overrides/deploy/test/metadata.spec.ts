import { env, fetchMock, SELF } from 'cloudflare:test'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import SQL from '../../gateway/limits-schema.sql?raw'
import { resetCapacityControllerForTest } from '../src/capacity'
import { resetMetadataCachesForTest } from '../src/metadata'

const RESET_SQL = `\
DROP TABLE IF EXISTS spend;
DROP TABLE IF EXISTS keyStatus;

${SQL}`

beforeEach(async () => {
  resetCapacityControllerForTest()
  resetMetadataCachesForTest()
  const keys = await env.KV.list()
  if (keys.keys.length !== 0) {
    throw new Error('KV store is not empty before test.')
  }
  await env.limitsDB.prepare(RESET_SQL).run()
  fetchMock.activate()
})

afterEach(() => {
  fetchMock.assertNoPendingInterceptors()
})

function mockPricingSnapshot(
  payload?: Array<{
    id: string
    models: Array<Record<string, unknown>>
  }>,
) {
  return fetchMock
    .get('https://raw.githubusercontent.com')
    .intercept({ method: 'GET', path: '/pydantic/genai-prices/refs/heads/main/prices/data.json' })
    .reply(200, payload ?? [])
}

describe('metadata', () => {
  it('requires a valid gateway app key', async () => {
    const noAuth = await SELF.fetch('https://example.com/metadata/models?provider=openai')
    expect(noAuth.status).toBe(401)

    const wrongAuth = await SELF.fetch('https://example.com/metadata/models?provider=openai', {
      headers: { authorization: 'wrong' },
    })
    expect(wrongAuth.status).toBe(401)

    const noCatalogAuth = await SELF.fetch('https://example.com/metadata/catalog')
    expect(noCatalogAuth.status).toBe(401)
  })

  it('returns provider-native OpenAI models', async () => {
    mockPricingSnapshot([
      {
        id: 'openai',
        models: [
          {
            id: 'gpt-5-mini',
            context_window: 400000,
            prices: { input_mtok: 0.25, output_mtok: 2 },
          },
          {
            id: 'gpt-5.4',
            context_window: 400000,
            prices: { input_mtok: 1.25, output_mtok: 10 },
          },
        ],
      },
    ])
    fetchMock
      .get('http://localhost:8005')
      .intercept({ method: 'GET', path: '/openai/v1/models' })
      .reply(200, {
        data: [{ id: 'gpt-5.4' }, { id: 'gpt-5-mini' }],
      })

    const response = await SELF.fetch('https://example.com/metadata/models?provider=openai', {
      headers: { authorization: 'healthy-key' },
    })
    expect(response.status).toBe(200)

    const payload = (await response.json()) as Array<{
      provider_id: string
      upstream_provider_id: string
      model_id: string
      display_name: string
      lifecycle_status: string
    }>
    expect(payload.map((item) => item.model_id)).toEqual(['gpt-5-mini', 'gpt-5.4'])
    expect(payload[0]?.provider_id).toBe('openai')
    expect(payload[0]?.upstream_provider_id).toBe('openai')
    expect(payload[1]?.display_name).toBe('GPT 5 4')
    expect(payload[1]?.lifecycle_status).toBe('stable')
  })

  it('returns a provider-first metadata catalog for the current app key', async () => {
    mockPricingSnapshot([
      {
        id: 'openai',
        models: [{ id: 'gpt-5.4', context_window: 400000, prices: { input_mtok: 1.25, output_mtok: 10 } }],
      },
      {
        id: 'anthropic',
        models: [
          {
            id: 'claude-sonnet-4-20250514',
            context_window: 200000,
            prices: { input_mtok: 3, output_mtok: 15 },
          },
        ],
      },
    ])
    fetchMock
      .get('http://localhost:8005')
      .intercept({ method: 'GET', path: '/openai/v1/models' })
      .reply(200, {
        data: [{ id: 'gpt-5.4' }],
      })

    fetchMock
      .get('http://localhost:8005')
      .intercept({ method: 'GET', path: '/anthropic/v1/models' })
      .reply(200, {
        data: [{ id: 'claude-sonnet-4-20250514', display_name: 'Claude Sonnet 4' }],
      })

    const response = await SELF.fetch('https://example.com/metadata/catalog', {
      headers: { authorization: 'healthy-key' },
    })
    expect(response.status).toBe(200)

    const payload = (await response.json()) as {
      providers: Array<{
        provider_id: string
        route: string
        upstream_provider_id: string
        label: string
        models: Array<{ model_id: string }>
      }>
    }

    expect(payload.providers.map((provider) => provider.provider_id)).toEqual([
      'anthropic',
      'anthropic_bedrock',
      'openai',
    ])
    expect(payload.providers.map((provider) => provider.route)).toEqual(['anthropic', 'bedrock', 'openai'])
    expect(payload.providers.map((provider) => provider.upstream_provider_id)).toEqual([
      'anthropic',
      'bedrock',
      'openai',
    ])
    expect(payload.providers.map((provider) => provider.label)).toEqual([
      'Anthropic',
      'AWS Bedrock (Claude)',
      'OpenAI',
    ])
    expect(payload.providers[0]?.models.map((model) => model.model_id)).toEqual(['claude-sonnet-4-20250514'])
    expect(payload.providers[1]?.models.map((model) => model.model_id)).toEqual([
      'anthropic.claude-opus-4-1-20250805-v1:0',
      'anthropic.claude-opus-4-6-v1',
      'anthropic.claude-sonnet-4-5-v1',
    ])
    expect(payload.providers[2]?.models.map((model) => model.model_id)).toEqual(['gpt-5.4'])
  })

  it('returns curated Bedrock Claude models under anthropic_bedrock', async () => {
    mockPricingSnapshot([
      {
        id: 'anthropic',
        models: [
          {
            id: 'claude-sonnet-4-5',
            context_window: 1000000,
            prices: { input_mtok: 3, output_mtok: 15 },
          },
          {
            id: 'claude-opus-4-6',
            context_window: 200000,
            prices: { input_mtok: 15, output_mtok: 75 },
          },
          {
            id: 'claude-opus-4-1-20250805',
            context_window: 200000,
            prices: { input_mtok: 15, output_mtok: 75 },
          },
        ],
      },
    ])

    const response = await SELF.fetch('https://example.com/metadata/models?provider=anthropic_bedrock', {
      headers: { authorization: 'healthy-key' },
    })
    expect(response.status).toBe(200)

    const payload = (await response.json()) as Array<{
      provider_id: string
      upstream_provider_id: string
      model_id: string
      display_name: string
      context_window: number | null
      sources: { display_name: string; pricing: string; reasoning_capabilities: string }
    }>

    expect(payload.map((item) => item.model_id)).toEqual([
      'anthropic.claude-opus-4-1-20250805-v1:0',
      'anthropic.claude-opus-4-6-v1',
      'anthropic.claude-sonnet-4-5-v1',
    ])
    expect(payload[0]).toEqual(
      expect.objectContaining({
        provider_id: 'anthropic_bedrock',
        upstream_provider_id: 'bedrock',
        display_name: 'Claude Opus 4.1',
        context_window: 200000,
        sources: expect.objectContaining({
          display_name: 'provider-heuristic',
          pricing: 'genai-prices',
          reasoning_capabilities: 'provider-heuristic',
        }),
      }),
    )
  })

  it('enriches Anthropic metadata from the live pricing snapshot', async () => {
    mockPricingSnapshot([
      {
        id: 'anthropic',
        models: [
          {
            id: 'claude-sonnet-4-6',
            context_window: 1000000,
            prices: { input_mtok: { base: 3, tiers: [{ start: 200000, price: 6 }] }, output_mtok: 15 },
          },
        ],
      },
    ])
    fetchMock
      .get('http://localhost:8005')
      .intercept({ method: 'GET', path: '/anthropic/v1/models' })
      .reply(200, {
        data: [{ id: 'claude-sonnet-4-6', display_name: 'Claude Sonnet 4.6' }],
      })

    const response = await SELF.fetch('https://example.com/metadata/models?provider=anthropic', {
      headers: { authorization: 'healthy-key' },
    })
    expect(response.status).toBe(200)

    const payload = (await response.json()) as Array<{
      model_id: string
      context_window: number | null
      sources: { context_window: string; pricing: string }
    }>
    expect(payload).toEqual([
      expect.objectContaining({
        model_id: 'claude-sonnet-4-6',
        context_window: 1000000,
        sources: expect.objectContaining({
          context_window: 'genai-prices',
          pricing: 'genai-prices',
        }),
      }),
    ])
  })

  it('falls back to the bundled pricing snapshot when the live pricing fetch fails', async () => {
    fetchMock
      .get('https://raw.githubusercontent.com')
      .intercept({ method: 'GET', path: '/pydantic/genai-prices/refs/heads/main/prices/data.json' })
      .reply(503, 'unavailable')
    fetchMock
      .get('http://localhost:8005')
      .intercept({ method: 'GET', path: '/anthropic/v1/models' })
      .reply(200, {
        data: [{ id: 'claude-sonnet-4-5', display_name: 'Claude Sonnet 4.5' }],
      })

    const response = await SELF.fetch('https://example.com/metadata/models?provider=anthropic', {
      headers: { authorization: 'healthy-key' },
    })
    expect(response.status).toBe(200)

    const payload = (await response.json()) as Array<{
      model_id: string
      context_window: number | null
      sources: { context_window: string }
    }>
    expect(payload).toEqual([
      expect.objectContaining({
        model_id: 'claude-sonnet-4-5',
        context_window: 1000000,
        sources: expect.objectContaining({
          context_window: 'genai-prices',
        }),
      }),
    ])
  })
})
