import { env, fetchMock, SELF } from 'cloudflare:test'
import OpenAI from 'openai'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'
import SQL from '../../gateway/limits-schema.sql?raw'
import { resetCapacityControllerForTest } from '../src/capacity'
import { config } from '../src/config'

beforeAll(async () => {
  try {
    const response = await fetch('http://localhost:8005')
    expect(response.status, 'The Proxy VCR seems to be facing issues, please check the logs.').toBe(204)
  } catch {
    throw new Error('Proxy VCR is not running. Run `make run-proxy-vcr` to enable tests.')
  }
})

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

describe('index', () => {
  it('responds with index html', async () => {
    const response = await SELF.fetch('https://example.com')
    expect(response.status).toBe(200)
  })
})

function recordOtelBatch(otelBatch: string[]) {
  fetchMock
    .get('https://logfire.pydantic.dev')
    .intercept({ method: 'POST', path: '/v1/traces', headers: { Authorization: 'write-token' } })
    .reply(({ body }) => {
      if (typeof body === 'string') {
        otelBatch.push(body)
      } else {
        throw new Error('Unexpected response body type')
      }
      return { statusCode: 200, body }
    })
}

describe('deploy', () => {
  it('status auth works', async () => {
    const noAuth = await SELF.fetch('https://example.com/status/')
    expect(noAuth.status).toBe(401)
    expect(await noAuth.text()).toMatchInlineSnapshot(`"Unauthorized - Missing "Authorization" Header"`)

    const wrongAuth = await SELF.fetch('https://example.com/status/', { headers: { authorization: 'wrong' } })
    expect(wrongAuth.status).toBe(401)
    expect(await wrongAuth.text()).toMatchInlineSnapshot(`"Unauthorized - Invalid API Key"`)

    const ok = await SELF.fetch('https://example.com/status/', { headers: { authorization: 'testing' } })
    expect(ok.status).toBe(200)
    const payload = (await ok.json()) as { capacity: { enabled: boolean; providers: Record<string, unknown> } }
    expect(payload.capacity.enabled).toBe(true)
    expect(Object.keys(payload.capacity.providers)).toEqual(['anthropic', 'google-vertex', 'openai'])
  })

  it('metadata endpoint requires a valid gateway app key', async () => {
    const noAuth = await SELF.fetch('https://example.com/metadata/models?provider=openai')
    expect(noAuth.status).toBe(401)

    const wrongAuth = await SELF.fetch('https://example.com/metadata/models?provider=openai', {
      headers: { authorization: 'wrong' },
    })
    expect(wrongAuth.status).toBe(401)
  })

  it('metadata endpoint returns provider-native OpenAI models', async () => {
    const appKey = Object.keys(config.apiKeys)[0]
    if (!appKey) {
      throw new Error('Expected at least one configured gateway app key')
    }

    fetchMock
      .get('https://api.openai.com')
      .intercept({ method: 'GET', path: '/v1/models' })
      .reply(200, {
        data: [
          { id: 'gpt-5.4' },
          { id: 'gpt-5-mini' },
        ],
      })

    const response = await SELF.fetch('https://example.com/metadata/models?provider=openai', {
      headers: { authorization: appKey },
    })
    expect(response.status).toBe(200)
    const payload = (await response.json()) as {
      provider: string
      upstream_provider: string
      models: Array<{ model_id: string; display_name: string; lifecycle_status: string }>
    }
    expect(payload.provider).toBe('openai')
    expect(payload.upstream_provider).toBe('openai')
    expect(payload.models.map((item) => item.model_id)).toEqual(['gpt-5.4', 'gpt-5-mini'])
    expect(payload.models[0]?.display_name).toBe('gpt-5.4')
    expect(payload.models[0]?.lifecycle_status).toBe('stable')
  })

  it('should call openai via gateway', async () => {
    const otelBatch: string[] = []
    recordOtelBatch(otelBatch)

    const client = new OpenAI({
      apiKey: 'healthy-key',
      baseURL: 'https://example.com/openai',
      fetch: SELF.fetch.bind(SELF),
    })

    const completion = await client.chat.completions.create({
      model: 'gpt-5',
      messages: [
        { role: 'developer', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is the capital of France?' },
      ],
    })

    expect(completion).toMatchSnapshot('llm')
    expect(otelBatch.length).toBe(1)
    expect(JSON.parse(otelBatch[0]!).resourceSpans?.[0].scopeSpans?.[0].spans?.[0]?.attributes).toMatchSnapshot('span')

    const response = await SELF.fetch('https://example.com/status/', { headers: { authorization: 'testing' } })
    expect(response.status).toBe(200)
    const payload = (await response.json()) as {
      keys: Array<{ spend: Array<{ raw: number }> }>
      capacity: { providers: Record<string, { currentLimit: number; inFlight: number; rejectCount: number }> }
    }
    expect(payload.keys).toHaveLength(2)
    expect(payload.capacity.providers.openai.inFlight).toBe(0)
    expect(payload.capacity.providers.openai.currentLimit).toBeGreaterThanOrEqual(1)
    expect(payload.capacity.providers.openai.rejectCount).toBe(0)
  })
})
