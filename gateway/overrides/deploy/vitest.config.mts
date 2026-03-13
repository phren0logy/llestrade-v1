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
            GOOGLE_VERTEX_SERVICE_ACCOUNT_JSON:
              process.env.GOOGLE_VERTEX_SERVICE_ACCOUNT_JSON ??
              JSON.stringify({
                type: 'service_account',
                project_id: 'pydantic-ai-gateway',
                private_key_id: 'test',
                private_key:
                  '-----BEGIN PRIVATE KEY-----\\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCqbLHcKxikDFkS\\nl2RhDtUhyhjZkX0ec9UcJMdcpgmW8yIuX4e9rg6VDf4yGZY/R9s94mt50ujj1T7V\\ny6sFWyDFk8x6zGf8qrIovCMbpkuoDKdw6Pg2fT4w0ykXf7NsWiy8vS0PJ8kTXc+O\\nleKxANCeL5t1J8kEr9dEgS1nkAztlZhGFm+oDEPRFEqok6XflOnaWf0Jp8nxhDpy\\nSOrJvi1yzCxXQmZRMEfpGwHUXlZW3cZ12o+vtB8tcdqBOJNTtCcnsVQgNtJUE8wd\\nTnHhxOx4jYfz8vpVCRYz994z5earU1+aQ72swBJ4Ml1B7Q4qeebaBK9BgqLTYDMg\\nR00/44ctAgMBAAECggEABqC7Cw23A9xCE43FNtwZkFqYfE+i7s4A7fsFMNZ5PYBf\\npi+LayZLhWL5rY+erv/XqDC3ztFTwtaOfsiQsmDgpaZxl53+9k3aJ2jnsaHWQ+cY\\nFPVOf+n/gu0XZkVOYEqifkWaLMJ2mLT3hztPiWQ3eMtvr474S6EeDVk7v5ET26J7\\nManLTE8WYDlh0ZHhwUC5BgaOlp0YAEixonIetmXHTd5ruzlHpVuQWLUtulA0Wv7+\\nf+Yv4WlGTKPfzC3aaZZC0KjvQ3P8zyu1DTxN8OyPJTmw2VMM1RVPdYkNVCssMoZ+\\nKjD/H1ij3aykPRwyzQ+GtIqhK38lS7ZuJ+WF1fRlmQKBgQDht8sJetTy4lxOZ32X\\n+rDh8oC8lq1BleLen8184dfKr5Vn6fSvUGQ+aEomhCWLBq9S2ORvJ/U4yOi3Ff25\\n9/E88e366sW9ofJgjhK4kB2VzTSOx1JM4wk16fMO9Dx9XPnkX8XLfl827RSLCFbU\\n5VqO0W2xwa7TKvMSSxWS00dKBQKBgQDBSd65tVnMmENRgjo0pqT55O2BHcLNH1Ie\\nXPcbZDjgzTxIEOV4Q36AP3N6UBpkyuksL1YFQCP5G8YJhlV7YF1Av4sowc73miYE\\no0+X8xAd93k1IuUc0kiKjYOn1BBTt2RioH/FzUZ+2sp2rCWDJMwzsSwTAZdQ6kSd\\nBVaEQ3TJCQKBgCaFyKgwh4GAcoKLFRtIFMIrMh99k8o6u4KFQXvLy5lzCAu5GSKG\\nlOU1xVn3ebTVijyYebwi1K5BU75TiX8gutJM8/G7+c2YgxZJiRZoujPj1tF7YSdw\\nJBVIfUwTEPPQV3HLiqVlRvjH05a68J7bGe5//bm1tZGipeN8Xw0089jFAoGBAJx1\\nKBWY52SGR8+do3HlBpvFJD8kkP+q/7TWOavxd1z4pHgNPUIZGDfFpLr4RjUaTp5W\\nfsHnRncpdSdWlrE0sqdrpMBMCTVBkM6mRxJPTNeE75cEdQLccJ2+qThbnw+03kw7\\ncNHzNMIQZlyjQgYi7ixVmMCVxB9aUknr/Tk4xTrJAoGBANzyxuc2QKG+y/K858ib\\nR9Oow3rP+qfnzEsyftEgDxIgE1pv2voYyi2UwQDRe4dftJUtbNzJWdkeTr39LmKg\\nQ9XZPt/XuPOZ7JyyMG7s0M5Ig6ypKQlpPPk2UBC7dEv24zHmkMmmSVZAdGudGDmQ\\nL8ji+k3fimdsAXCai6QcXuKi\\n-----END PRIVATE KEY-----\\n',
                client_email: 'gateway@test.iam.gserviceaccount.com',
                client_id: '1234567890',
              }),
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
