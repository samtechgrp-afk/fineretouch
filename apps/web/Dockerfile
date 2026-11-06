FROM node:22-alpine AS build
WORKDIR /repo
COPY package.json package-lock.json ./
COPY packages/shared/package.json packages/shared/
COPY apps/web/package.json apps/web/
RUN npm ci --workspace @fineretouch/web --workspace @fineretouch/shared --include-workspace-root
COPY packages/shared packages/shared
COPY apps/web apps/web
# Baked into the client bundle: where the browser reaches the AI service.
ARG NEXT_PUBLIC_AI_URL=http://localhost:8000
ENV NEXT_PUBLIC_AI_URL=$NEXT_PUBLIC_AI_URL NEXT_TELEMETRY_DISABLED=1
RUN npm run build -w @fineretouch/shared && npm run build -w @fineretouch/web

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /repo/apps/web/.next/standalone ./
COPY --from=build /repo/apps/web/.next/static ./apps/web/.next/static
USER node
EXPOSE 3000
CMD ["node", "apps/web/server.js"]
