import type { Middleware, Next, RequestHandler } from '@pydantic/ai-gateway'
import logfire from 'logfire'

const DEFAULT_INITIAL_LIMIT = 8
const DEFAULT_MIN_LIMIT = 1
const DEFAULT_MAX_LIMIT = 32
const DEFAULT_COOLDOWN_MS = 30_000
const DEFAULT_RETRY_AFTER_SECONDS = 1
const DEFAULT_PROVIDER_IDS = ['anthropic', 'openai', 'google-vertex'] as const

type ProviderId = string

interface CapacitySettings {
  initialLimit: number
  minLimit: number
  maxLimit: number
  cooldownMs: number
}

interface CapacityBucket extends CapacitySettings {
  currentLimit: number
  inFlight: number
  successStreak: number
  cooldownUntil: number
  backoffCount: number
  rejectCount: number
}

interface CapacityConfig {
  enabled: boolean
  defaults: CapacitySettings
  overrides: Record<ProviderId, CapacitySettings>
}

export interface CapacitySnapshot {
  enabled: boolean
  providers: Record<
    string,
    {
      currentLimit: number
      inFlight: number
      cooldownUntil: string | null
      backoffCount: number
      rejectCount: number
      minLimit: number
      maxLimit: number
      initialLimit: number
      cooldownMs: number
    }
  >
}

type GatewayResult = Awaited<ReturnType<Next>>

type CompletionFeedback =
  | { kind: 'success' }
  | { kind: 'capacity'; signal: string; retryAfterMs?: number }
  | { kind: 'ignore' }

class ProviderCapacityController {
  readonly middleware: Middleware
  private readonly buckets = new Map<ProviderId, CapacityBucket>()

  constructor(private readonly config: CapacityConfig) {
    this.middleware = {
      dispatch: (next: Next): Next => {
        return async (handler: RequestHandler) => {
          if (!this.config.enabled) {
            return await next(handler)
          }

          const providerId = handler.providerId()
          const admission = this.acquire(providerId)
          if (admission) {
            return admission
          }

          try {
            const result = await next(handler)
            return this.decorateResult(providerId, result)
          } catch (error) {
            this.release(providerId, { kind: 'capacity', signal: 'transport_error' })
            throw error
          }
        }
      },
    }
  }

  snapshot(): CapacitySnapshot {
    const providers: CapacitySnapshot['providers'] = {}
    const providerIds = new Set<string>([
      ...DEFAULT_PROVIDER_IDS,
      ...Object.keys(this.config.overrides),
      ...this.buckets.keys(),
    ])

    for (const providerId of Array.from(providerIds).sort()) {
      const bucket = this.getBucket(providerId)
      providers[providerId] = {
        currentLimit: bucket.currentLimit,
        inFlight: bucket.inFlight,
        cooldownUntil: bucket.cooldownUntil > 0 ? new Date(bucket.cooldownUntil).toISOString() : null,
        backoffCount: bucket.backoffCount,
        rejectCount: bucket.rejectCount,
        minLimit: bucket.minLimit,
        maxLimit: bucket.maxLimit,
        initialLimit: bucket.initialLimit,
        cooldownMs: bucket.cooldownMs,
      }
    }

    return { enabled: this.config.enabled, providers }
  }

  resetForTest() {
    this.buckets.clear()
  }

  private decorateResult(providerId: ProviderId, result: GatewayResult): GatewayResult {
    if ('responseStream' in result) {
      const original = result.onStreamComplete
      return {
        ...result,
        onStreamComplete: original.then(
          (value) => {
            if ('error' in value) {
              this.release(providerId, { kind: 'capacity', signal: 'stream_error' })
            } else {
              this.release(providerId, { kind: 'success' })
            }
            return value
          },
          (error) => {
            this.release(providerId, { kind: 'capacity', signal: 'stream_rejection' })
            throw error
          },
        ),
      }
    }

    if ('unexpectedStatus' in result) {
      this.release(providerId, {
        kind: isCapacityStatus(result.unexpectedStatus) ? 'capacity' : 'ignore',
        signal: `status_${result.unexpectedStatus}`,
        retryAfterMs: retryAfterToMs(result.responseHeaders),
      })
      return result
    }

    if ('successStatus' in result) {
      this.release(providerId, { kind: 'success' })
      return result
    }

    if ('response' in result) {
      this.release(
        providerId,
        isCapacityStatus(result.response.status)
          ? {
              kind: 'capacity',
              signal: `status_${result.response.status}`,
              retryAfterMs: retryAfterToMs(result.response.headers),
            }
          : { kind: 'success' },
      )
      return result
    }

    this.release(providerId, { kind: 'ignore' })
    return result
  }

  private acquire(providerId: ProviderId): GatewayResult | null {
    const bucket = this.getBucket(providerId)
    if (bucket.inFlight >= bucket.currentLimit) {
      bucket.rejectCount += 1
      const retryAfterSeconds = Math.max(
        DEFAULT_RETRY_AFTER_SECONDS,
        Math.ceil(Math.max(bucket.cooldownUntil - Date.now(), 0) / 1000),
      )
      logfire.info('Provider capacity rejected request', {
        providerId,
        currentLimit: bucket.currentLimit,
        inFlight: bucket.inFlight,
        cooldownUntil: bucket.cooldownUntil || null,
        retryAfterSeconds,
      })
      return {
        response: new Response(`Provider capacity reached for ${providerId}. Retry soon.`, {
          status: 429,
          headers: {
            'content-type': 'text/plain; charset=utf-8',
            'retry-after': String(retryAfterSeconds),
          },
        }),
      }
    }
    bucket.inFlight += 1
    return null
  }

  private release(providerId: ProviderId, feedback: CompletionFeedback) {
    const bucket = this.getBucket(providerId)
    bucket.inFlight = Math.max(0, bucket.inFlight - 1)

    if (feedback.kind === 'ignore') {
      return
    }

    if (feedback.kind === 'success') {
      if (Date.now() < bucket.cooldownUntil) {
        return
      }
      bucket.successStreak += 1
      if (bucket.successStreak >= bucket.currentLimit && bucket.currentLimit < bucket.maxLimit) {
        const previousLimit = bucket.currentLimit
        bucket.currentLimit = Math.min(bucket.maxLimit, bucket.currentLimit + 1)
        bucket.successStreak = 0
        if (bucket.currentLimit !== previousLimit) {
          logfire.info('Provider capacity increased', {
            providerId,
            previousLimit,
            currentLimit: bucket.currentLimit,
          })
        }
      }
      return
    }

    const previousLimit = bucket.currentLimit
    bucket.currentLimit = Math.max(bucket.minLimit, Math.floor(bucket.currentLimit / 2))
    bucket.successStreak = 0
    bucket.backoffCount += 1
    bucket.cooldownUntil = Date.now() + (feedback.retryAfterMs ?? bucket.cooldownMs)
    logfire.info('Provider capacity backed off', {
      providerId,
      signal: feedback.signal,
      previousLimit,
      currentLimit: bucket.currentLimit,
      cooldownUntil: bucket.cooldownUntil,
    })
  }

  private getBucket(providerId: ProviderId): CapacityBucket {
    let bucket = this.buckets.get(providerId)
    if (!bucket) {
      const settings = this.config.overrides[providerId] ?? this.config.defaults
      bucket = {
        ...settings,
        currentLimit: settings.initialLimit,
        inFlight: 0,
        successStreak: 0,
        cooldownUntil: 0,
        backoffCount: 0,
        rejectCount: 0,
      }
      this.buckets.set(providerId, bucket)
    }
    return bucket
  }
}

let controller: ProviderCapacityController | undefined

export function getCapacityController(env: Env): ProviderCapacityController {
  controller ??= new ProviderCapacityController(parseCapacityConfig(env))
  return controller
}

export function resetCapacityControllerForTest() {
  controller?.resetForTest()
}

function parseCapacityConfig(env: Env): CapacityConfig {
  const bindings = env as unknown as Record<string, unknown>
  const defaults = normalizeSettings({
    initialLimit: parseInteger(bindings.ADAPTIVE_CAPACITY_DEFAULT_INITIAL_LIMIT, DEFAULT_INITIAL_LIMIT),
    minLimit: parseInteger(bindings.ADAPTIVE_CAPACITY_DEFAULT_MIN_LIMIT, DEFAULT_MIN_LIMIT),
    maxLimit: parseInteger(bindings.ADAPTIVE_CAPACITY_DEFAULT_MAX_LIMIT, DEFAULT_MAX_LIMIT),
    cooldownMs: parseInteger(bindings.ADAPTIVE_CAPACITY_DEFAULT_COOLDOWN_MS, DEFAULT_COOLDOWN_MS),
  })

  const overrides: Record<string, CapacitySettings> = {}
  for (const [providerId, prefix] of Object.entries({
    anthropic: 'ANTHROPIC',
    openai: 'OPENAI',
    'google-vertex': 'GOOGLE_VERTEX',
  })) {
    const providerSettings = normalizeSettings({
      initialLimit: parseInteger(bindings[`ADAPTIVE_CAPACITY_${prefix}_INITIAL_LIMIT`], defaults.initialLimit),
      minLimit: parseInteger(bindings[`ADAPTIVE_CAPACITY_${prefix}_MIN_LIMIT`], defaults.minLimit),
      maxLimit: parseInteger(bindings[`ADAPTIVE_CAPACITY_${prefix}_MAX_LIMIT`], defaults.maxLimit),
      cooldownMs: parseInteger(bindings[`ADAPTIVE_CAPACITY_${prefix}_COOLDOWN_MS`], defaults.cooldownMs),
    })
    overrides[providerId] = providerSettings
  }

  return {
    enabled: parseBoolean(bindings.ADAPTIVE_CAPACITY_ENABLED, true),
    defaults,
    overrides,
  }
}

function normalizeSettings(settings: CapacitySettings): CapacitySettings {
  const minLimit = Math.max(DEFAULT_MIN_LIMIT, settings.minLimit)
  const maxLimit = Math.max(minLimit, settings.maxLimit)
  return {
    minLimit,
    maxLimit,
    initialLimit: clamp(settings.initialLimit, minLimit, maxLimit),
    cooldownMs: Math.max(1_000, settings.cooldownMs),
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function parseInteger(value: unknown, fallback: number): number {
  if (value === undefined || value === null || value === '') {
    return fallback
  }
  const parsed = Number.parseInt(String(value), 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

function parseBoolean(value: unknown, fallback: boolean): boolean {
  if (value === undefined) {
    return fallback
  }
  if (typeof value === 'boolean') {
    return value
  }
  return String(value).toLowerCase() === 'true'
}

function isCapacityStatus(status: number): boolean {
  return status === 429 || status >= 500
}

function retryAfterToMs(headers: Headers): number | undefined {
  const retryAfter = headers.get('retry-after')
  if (!retryAfter) {
    return undefined
  }
  const seconds = Number.parseInt(retryAfter, 10)
  if (Number.isFinite(seconds)) {
    return Math.max(seconds, DEFAULT_RETRY_AFTER_SECONDS) * 1000
  }
  const timestamp = Date.parse(retryAfter)
  if (Number.isNaN(timestamp)) {
    return undefined
  }
  return Math.max(timestamp - Date.now(), DEFAULT_RETRY_AFTER_SECONDS * 1000)
}
