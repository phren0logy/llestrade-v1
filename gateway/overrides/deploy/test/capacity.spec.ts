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
  await env.limitsDB.prepare(RESET_SQL).run()
  fetchMock.activate()
})

afterEach(() => {
  fetchMock.assertNoPendingInterceptors()
})

describe('provider capacity', () => {
  it('rejects immediately when a provider bucket is saturated', async () => {
    const first = SELF.fetch('https://example.com/test/chat/completions', {
      method: 'POST',
      headers: {
        authorization: 'healthy-key',
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-5',
        sleep: 'sleep=250',
        messages: [{ role: 'user', content: 'sleep=250' }],
      }),
    })

    await new Promise((resolve) => setTimeout(resolve, 20))

    const second = await SELF.fetch('https://example.com/test/chat/completions', {
      method: 'POST',
      headers: {
        authorization: 'healthy-key',
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-5',
        messages: [{ role: 'user', content: 'second request' }],
      }),
    })

    expect(second.status).toBe(429)
    expect(await second.text()).toContain('openai')

    const firstResponse = await first
    expect(firstResponse.status).toBe(200)
  })

  it('keeps provider buckets independent', async () => {
    fetchMock
      .get('http://localhost:8005')
      .intercept({ method: 'POST', path: '/anthropic/v1/messages' })
      .reply(200, {
        id: 'msg_test',
        type: 'message',
        role: 'assistant',
        model: 'claude-3-5-sonnet-20241022',
        content: [{ type: 'text', text: 'hello' }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: 12, output_tokens: 9 },
      })

    const first = SELF.fetch('https://example.com/test/chat/completions', {
      method: 'POST',
      headers: {
        authorization: 'healthy-key',
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-5',
        messages: [{ role: 'user', content: 'sleep=250' }],
      }),
    })

    await new Promise((resolve) => setTimeout(resolve, 20))

    const anthropic = await SELF.fetch('https://example.com/anthropic/v1/messages', {
      method: 'POST',
      headers: {
        authorization: 'healthy-key',
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 32,
        messages: [{ role: 'user', content: 'hello' }],
      }),
    })

    expect(anthropic.status).not.toBe(429)
    await first
  })

  it('reports provider capacity state on status', async () => {
    const response = await SELF.fetch('https://example.com/status/', {
      headers: { authorization: 'testing' },
    })

    expect(response.status).toBe(200)
    const payload = (await response.json()) as {
      capacity: {
        enabled: boolean
        providers: Record<string, { currentLimit: number; inFlight: number }>
      }
    }

    expect(payload.capacity.enabled).toBe(true)
    expect(payload.capacity.providers.openai.currentLimit).toBe(1)
    expect(payload.capacity.providers.anthropic.currentLimit).toBe(2)
    expect(payload.capacity.providers.bedrock.currentLimit).toBe(4)
    expect(payload.capacity.providers['google-vertex'].currentLimit).toBe(3)
  })
})
