-- FineRetouch schema. Loaded by the Postgres container on first start.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
  id          text PRIMARY KEY,               -- auth provider subject (anonymous cookie id until auth lands)
  email       text UNIQUE,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS projects (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     text NOT NULL,
  name        text NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS projects_user_idx ON projects (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS images (
  id           text PRIMARY KEY,
  project_id   uuid REFERENCES projects(id) ON DELETE CASCADE,
  user_id      text NOT NULL,
  source_key   text NOT NULL,                 -- object storage key of the original
  file_name    text NOT NULL,
  width        int,
  height       int,
  analysis     jsonb,                          -- ImageAnalysis from services/ai (faces, cull flags, colour stats)
  adjustments  jsonb NOT NULL DEFAULT '{}',    -- current Adjustments (see packages/shared)
  rejected     boolean NOT NULL DEFAULT false,
  -- 512-d image embedding for near-duplicate grouping and "find similar" culling.
  embedding    vector(512),
  created_at   timestamptz NOT NULL DEFAULT now(),
  updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS images_project_idx ON images (project_id);
CREATE INDEX IF NOT EXISTS images_embedding_idx ON images USING hnsw (embedding vector_cosine_ops);

-- Append-only edit history: lets undo survive reloads and devices.
CREATE TABLE IF NOT EXISTS image_edits (
  id           bigserial PRIMARY KEY,
  image_id     text NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  adjustments  jsonb NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS image_edits_image_idx ON image_edits (image_id, id DESC);

CREATE TABLE IF NOT EXISTS presets (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      text NOT NULL,
  name         text NOT NULL,
  adjustments  jsonb NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, name)
);

CREATE TABLE IF NOT EXISTS batches (
  id           text PRIMARY KEY,
  user_id      text NOT NULL,
  total        int NOT NULL,
  done         int NOT NULL DEFAULT 0,
  failed       int NOT NULL DEFAULT 0,
  preset       jsonb,
  state        text NOT NULL DEFAULT 'queued',
  created_at   timestamptz NOT NULL DEFAULT now(),
  finished_at  timestamptz
);
CREATE INDEX IF NOT EXISTS batches_user_idx ON batches (user_id, created_at DESC);
