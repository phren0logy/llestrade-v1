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
})
