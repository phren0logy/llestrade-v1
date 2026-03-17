/*
Copyright (C) 2025 to present Pydantic Services Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

import { env } from 'cloudflare:workers'
import { type GatewayOptions, gatewayFetch, KVCacheAdapter } from '@pydantic/ai-gateway'
import { instrument } from '@pydantic/logfire-cf-workers'
import logfire from 'logfire'
import { getCapacityController } from './capacity'
import { config } from './config'
import { ConfigDB, hash, LimitDbD1 } from './db'
import { metadataCatalog, metadataModels } from './metadata'
import { status } from './status'

const handler = {
  async fetch(request, env, ctx): Promise<Response> {
    const url = new URL(request.url)
    if (url.pathname === '/converse' || url.pathname.startsWith('/converse/')) {
      url.pathname = `/bedrock${url.pathname.slice('/converse'.length)}`
      request = new Request(url.toString(), request)
    }
    const limitDb = new LimitDbD1(env.limitsDB)
    const { pathname } = url
    const capacityController = getCapacityController(env)

    if (pathname === '/status' || pathname === '/status/') {
      return await status(request, env, limitDb, capacityController)
    }

    const gatewayEnv: GatewayOptions = {
      githubSha: env.GITHUB_SHA,
      keysDb: new ConfigDB(env.limitsDB),
      limitDb,
      cache: new KVCacheAdapter(env.KV),
      kvVersion: await hash(JSON.stringify(config)),
      subFetch: fetch,
      proxyMiddlewares: [capacityController.middleware],
    }
    if (pathname === '/metadata/catalog' || pathname === '/metadata/catalog/') {
      return await metadataCatalog(request, gatewayEnv)
    }
    if (pathname === '/metadata/models' || pathname === '/metadata/models/') {
      return await metadataModels(request, url, gatewayEnv)
    }
    try {
      return await gatewayFetch(request, url, ctx, gatewayEnv)
    } catch (error) {
      console.error('Internal Server Error:', error)
      logfire.reportError('Internal Server Error', error as Error)
      return new Response('Internal Server Error', { status: 500, headers: { 'content-type': 'text/plain' } })
    }
  },
} satisfies ExportedHandler<Env>

export default instrument(handler, {
  service: { name: 'gateway', version: env.GITHUB_SHA.substring(0, 7) },
  scrubbing: { extraPatterns: ['writeToken'] },
})
