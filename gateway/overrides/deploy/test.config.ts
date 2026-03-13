import { env } from 'cloudflare:workers'
import type { Config } from '@deploy/types'

type ProviderKeys = 'openai' | 'groq' | 'anthropic' | 'google-vertex' | 'bedrock' | 'test'

export const config: Config<ProviderKeys> = {
  projects: {
    1: {
      name: 'default',
      otel: { writeToken: 'write-token', baseUrl: 'https://logfire.pydantic.dev', exporterProtocol: 'http/json' },
      users: {
        2: {
          name: 'testberto',
          spendingLimitWeekly: 2,
        },
      },
      spendingLimitDaily: 1,
    },
  },
  routingGroups: {
    anthropic: [{ key: 'anthropic' }],
    openai: [{ key: 'openai' }],
    gemini: [{ key: 'google-vertex' }],
    'google-vertex': [{ key: 'google-vertex' }],
  },
  providers: {
    openai: {
      baseUrl: 'http://localhost:8005/openai',
      providerId: 'openai',
      injectCost: true,
      credentials: env.OPENAI_API_KEY,
    },
    groq: {
      baseUrl: 'http://localhost:8005/groq',
      providerId: 'groq',
      injectCost: true,
      credentials: env.GROQ_API_KEY,
    },
    anthropic: {
      baseUrl: 'http://localhost:8005/anthropic',
      providerId: 'anthropic',
      injectCost: true,
      credentials: env.ANTHROPIC_API_KEY,
    },
    'google-vertex': {
      baseUrl: 'https://us-central1-aiplatform.googleapis.com',
      providerId: 'google-vertex',
      injectCost: true,
      credentials: env.GOOGLE_VERTEX_SERVICE_ACCOUNT_JSON,
    },
    bedrock: {
      baseUrl: 'http://localhost:8005/bedrock',
      providerId: 'bedrock',
      injectCost: true,
      credentials: env.AWS_BEARER_TOKEN_BEDROCK,
    },
    test: { baseUrl: 'http://test.example.com/test', providerId: 'test', injectCost: true, credentials: 'test' },
  },
  apiKeys: {
    'healthy-key': {
      id: 3,
      project: 1,
      user: 2,
      providers: ['openai', 'groq', 'anthropic', 'google-vertex', 'test'],
      spendingLimitMonthly: 3,
      spendingLimitTotal: 4,
    },
    'low-limit-key': { id: 4, project: 1, providers: '__all__', spendingLimitDaily: 0.01 },
  },
}
