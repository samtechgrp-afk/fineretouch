FROM node:22-alpine AS build
WORKDIR /repo
COPY package.json package-lock.json ./
COPY packages/shared/package.json packages/shared/
COPY services/worker/package.json services/worker/
RUN npm ci --workspace @fineretouch/worker --workspace @fineretouch/shared --include-workspace-root
COPY packages/shared packages/shared
COPY services/worker services/worker
RUN npm run build -w @fineretouch/shared && npm run build -w @fineretouch/worker \
 && npm prune --omit=dev --workspace @fineretouch/worker --workspace @fineretouch/shared

FROM node:22-alpine
WORKDIR /repo
ENV NODE_ENV=production
COPY --from=build /repo/node_modules node_modules
COPY --from=build /repo/packages/shared/package.json packages/shared/package.json
COPY --from=build /repo/packages/shared/dist packages/shared/dist
COPY --from=build /repo/services/worker/package.json services/worker/package.json
COPY --from=build /repo/services/worker/dist services/worker/dist
USER node
CMD ["node", "services/worker/dist/index.js"]
