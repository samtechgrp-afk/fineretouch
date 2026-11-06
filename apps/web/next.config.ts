import path from 'node:path';
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'standalone',
  // Monorepo: trace workspace packages (packages/shared) into the standalone bundle.
  outputFileTracingRoot: path.join(__dirname, '../..'),
  agentRules: false,
  transpilePackages: ['@fineretouch/shared'],
  // bullmq/ioredis/pg are server-only and use Node built-ins.
  serverExternalPackages: ['bullmq', 'ioredis', 'pg'],
};

export default nextConfig;
