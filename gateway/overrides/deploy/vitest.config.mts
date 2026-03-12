import { defineWorkersConfig } from '@cloudflare/vitest-pool-workers/config'

export default defineWorkersConfig({
  test: {
    testTimeout: 30000,
    resolveSnapshotPath: (testPath, snapshotExtension) => testPath + snapshotExtension,
    deps: {
      optimizer: {
        ssr: {
          enabled: true,
          include: ['@pydantic/logfire-cf-workers', '@opentelemetry/resources', 'mime-types', 'mime-db'],
        },
      },
    },
    alias: { './config': '../test.config.ts' },
    poolOptions: {
      workers: {
        singleWorker: true,
        wrangler: { configPath: './wrangler.jsonc' },
        miniflare: {
          bindings: {
            STATUS_AUTH_API_KEY: 'testing',
            OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? 'OPENAI_API_KEY-unset',
            ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY ?? 'ANTHROPIC_API_KEY-unset',
            ADAPTIVE_CAPACITY_ENABLED: 'true',
            ADAPTIVE_CAPACITY_DEFAULT_INITIAL_LIMIT: '2',
            ADAPTIVE_CAPACITY_DEFAULT_MIN_LIMIT: '1',
            ADAPTIVE_CAPACITY_DEFAULT_MAX_LIMIT: '4',
            ADAPTIVE_CAPACITY_DEFAULT_COOLDOWN_MS: '1000',
            ADAPTIVE_CAPACITY_ANTHROPIC_INITIAL_LIMIT: '2',
            ADAPTIVE_CAPACITY_OPENAI_INITIAL_LIMIT: '1',
            ADAPTIVE_CAPACITY_GOOGLE_VERTEX_INITIAL_LIMIT: '3',
          },
        },
      },
    },
  },
})
