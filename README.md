# FineRetouch

AI-assisted batch retouching for photographers who deliver hundreds of portraits at a time. Edit one photo, sync the look to the whole shoot, let AI handle skin, eyes, teeth and colour consistency, flag the blurry and blinking frames, and export everything at full resolution.

## What works today

- **Batch import** of JPEG, PNG, WebP and AVIF, with a filmstrip, multi-select (Ctrl/Cmd-click, Shift-click) and keyboard navigation.
- **Real-time GPU preview** (WebGL2) on a 2560 px proxy: only one shader pass runs per slider move.
- **AI face analysis** (MediaPipe 478-point face mesh) drives face-aware masks for skin, eyes, under-eyes and mouth. Without the AI service, colour-based skin detection still works.
- **Skin retouch:** frequency-separation smoothing with texture preservation, blemish removal, and even skin tone.
- **Face features:** eye brightening, dark-circle reduction, teeth whitening.
- **Colour and tone:** exposure, contrast, highlights, shadows, white balance, saturation, vibrance, sharpening, vignette, and **colour match** from a reference photo (L\*a\*b\* statistics transfer).
- **Compare:** before/after split slider, side by side, original toggle, AI mask overlay, and zoom to 800 %.
- **Undo/redo** per photo, **copy/paste settings**, **sync to all or selected**, six built-in presets plus your own.
- **AI culling:** blurry and eyes-closed flags, a manual reject key, and Keepers/Flagged filters.
- **Export** at full resolution as JPEG, PNG or WebP, with three engines: in-browser GPU, the AI service, or a **cloud batch queue** (BullMQ workers → S3) for large jobs.

## Repository layout

| Path | What it is |
| :--- | :--- |
| `apps/web` | Next.js 16 editor and API gateway (React 19, Tailwind 4, Radix, Zustand, WebGL2) |
| `packages/shared` | Edit model, presets and job contracts shared by every service |
| `services/ai` | FastAPI service: landmarks, culling, colour stats, full-resolution renderer |
| `services/worker` | BullMQ batch worker: fan-out, retries, progress, manifests |
| `db/init.sql` | PostgreSQL + pgvector schema |
| `docs/ARCHITECTURE.md` | System design, data flow, AI model map, setup and roadmap |

## Quick start

```bash
docker compose up --build
# open http://localhost:3000
```

Editor only, no backend needed:

```bash
npm install
npm run build -w @fineretouch/shared
npm run dev:web
```

Add the AI service for face-aware tools (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#7-running-it) for every option):

```bash
cd services/ai && pip install -r requirements.txt && ./scripts/download-model.sh
uvicorn app.main:app --port 8000
```

## Tests

```bash
npm test                    # shared, web and worker unit tests
cd services/ai && pytest    # renderer, colour science and HTTP API
```

## Licences

Code in this repository is proprietary to its owner unless a licence file says otherwise. The MediaPipe Face Landmarker model is downloaded at build time and is licensed by Google under Apache-2.0.
