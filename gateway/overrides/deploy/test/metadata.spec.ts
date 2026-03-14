import { env, fetchMock, SELF } from 'cloudflare:test'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import SQL from '../../gateway/limits-schema.sql?raw'
import { resetCapacityControllerForTest } from '../src/capacity'

const RESET_SQL = `\
DROP TABLE IF EXISTS spend;
DROP TABLE IF EXISTS keyStatus;

${SQL}`

beforeEach(async () => {
  resetCapacityControllerForTest()
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

    fetchMock
      .get('https://oauth2.googleapis.com')
      .intercept({ method: 'POST', path: '/token' })
      .reply(200, {
        access_token: 'vertex-token',
        expires_in: 300,
      })

    fetchMock
      .get('https://us-central1-aiplatform.googleapis.com')
      .intercept({
        method: 'GET',
        path: '/v1beta1/publishers/google/models?pageSize=1000&listAllVersions=true&view=FULL',
        headers: { Authorization: 'Bearer vertex-token' },
      })
      .reply(200, {
        publisherModels: [
          {
            name: 'publishers/google/models/gemini-2.5-pro',
            displayName: 'Gemini 2.5 Pro',
            launchStage: 'GA',
            inputTokenLimit: 1048576,
            outputTokenLimit: 65536,
            supportedActions: ['generateContent'],
          },
        ],
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

    expect(payload.providers.map((provider) => provider.provider_id)).toEqual(['anthropic', 'gemini', 'openai'])
    expect(payload.providers.map((provider) => provider.route)).toEqual(['anthropic', 'google-vertex', 'openai'])
    expect(payload.providers.map((provider) => provider.upstream_provider_id)).toEqual([
      'anthropic',
      'google-vertex',
      'openai',
    ])
    expect(payload.providers.map((provider) => provider.label)).toEqual(['Anthropic', 'Google Gemini', 'OpenAI'])
    expect(payload.providers[0]?.models.map((model) => model.model_id)).toEqual(['claude-sonnet-4-20250514'])
    expect(payload.providers[1]?.models.map((model) => model.model_id)).toEqual(['gemini-2.5-pro'])
    expect(payload.providers[2]?.models.map((model) => model.model_id)).toEqual(['gpt-5.4'])
  })
})
