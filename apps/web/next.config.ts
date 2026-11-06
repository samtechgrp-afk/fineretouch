import path from 'node:path';
import type { NextConfig } from 'next';

/**
 * Two build targets:
 *
 * - Server (default): `output: 'standalone'` for Docker, with the /api routes that queue
 *   cloud batches.
 * - Desktop (`FINERETOUCH_DESKTOP=1`, see `npm run build:desktop`): a fully static export in
 *   `out/` that Tauri serves from disk. Pages are .tsx and route handlers are .ts, so
 *   limiting page extensions to .tsx leaves the server-only /api routes (Postgres, Redis,
 *   S3) out of the bundle; the desktop app renders through the local engine instead.
 */
// The Tauri CLI sets TAURI_ENV_* for its beforeDev/beforeBuild commands.
const desktop = process.env.FINERETOUCH_DESKTOP === '1' || Boolean(process.env.TAURI_ENV_PLATFORM);

const nextConfig: NextConfig = desktop
  ? {
      output: 'export',
      pageExtensions: ['tsx'],
      // Tauri resolves /editor/ to editor/index.html.
      trailingSlash: true,
      images: { unoptimized: true },
      agentRules: false,
      transpilePackages: ['@fineretouch/shared'],
      env: { NEXT_PUBLIC_FINERETOUCH_DESKTOP: '1' },
    }
  : {
      output: 'standalone',
      // Monorepo: trace workspace packages (packages/shared) into the standalone bundle.
      outputFileTracingRoot: path.join(__dirname, '../..'),
      agentRules: false,
      transpilePackages: ['@fineretouch/shared'],
      // bullmq/ioredis/pg are server-only and use Node built-ins.
      serverExternalPackages: ['bullmq', 'ioredis', 'pg'],
    };

export default nextConfig;
