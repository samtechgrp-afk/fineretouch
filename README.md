# FineRetouch

AI-assisted batch retouching for photographers who deliver hundreds of portraits at a time. Edit one photo, sync the look to the whole shoot, let AI handle skin, eyes, teeth and colour consistency, flag the blurry and blinking frames, and export everything at full resolution.

## What works today

- **Batch import** of JPEG, PNG, WebP, AVIF, TIFF and **camera RAW** (Canon CR2/CR3, Nikon NEF, Sony ARW, Fujifilm RAF, Adobe DNG, Olympus ORF, Panasonic RW2, Pentax PEF), with a filmstrip, multi-select (Ctrl/Cmd-click, Shift-click) and keyboard navigation.
- **Real-time GPU preview** (WebGL2) on a 2560 px proxy: only one shader pass runs per slider move.
- **RAW workflow:** the browser pulls each RAW's embedded full-size camera JPEG and EXIF in milliseconds (no demosaic), shows a format badge and camera/ISO/shutter tooltip in the filmstrip, and AI-service or cloud exports develop the original from 16-bit sensor data with LibRaw (camera or Kelvin white balance, highlight recovery, AHD/DHT/PPG demosaic, sRGB/Adobe RGB/ProPhoto).
- **AI face analysis** (MediaPipe 478-point face mesh) drives face-aware masks for skin, eyes, under-eyes and mouth. Without the AI service, colour-based skin detection still works.
- **Skin retouch:** frequency-separation smoothing with texture preservation, blemish removal, and even skin tone.
- **1x / 2x / 3x skin & blemish boost:** recursive guided-filter clean-up of stubborn dark spots, acne and redness that keeps pore texture.
- **Stray hair cleanup** around the face outline and forehead (landmark band + Gabor strand detector + inpainting).
- **Depth-aware background bokeh:** MiDaS depth + person segmentation, 100-300 % blur multiplier, circular or hexagonal aperture, highlight bloom.
- **Sync style from reference:** drop any photo; its palette, tone curve and retouch strength are mapped onto one photo or the whole shoot (exact palette or editable sliders).
- **AI assist:** one-click Auto Retouch, and Smart Sync that evens out exposure and white balance across a shoot while copying your edits.
- **Face features:** eye brightening, dark-circle reduction, teeth whitening.
- **Colour and tone:** exposure, contrast, highlights, shadows, white balance, saturation, vibrance, sharpening and vignette.
- **Compare:** before/after split slider, side by side, original toggle, a difference heat map that shows exactly what the retouch changed, AI mask overlay, and zoom to 800 %.
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
| `src-tauri` | Tauri v2 desktop shell: window, native dialogs, engine sidecar lifecycle |
| `scripts/package_desktop.py` | Freezes services/ai into the offline engine sidecar and stages the models |
| `docs/DESKTOP.md` | Offline desktop app: build, develop, security model, data folders |

## Desktop app (offline)

FineRetouch also ships as a standalone Windows, macOS and Linux app with the AI engine and
models built in, so it runs with no network at all. See [docs/DESKTOP.md](docs/DESKTOP.md).

```bash
npm install && pip install -r services/ai/requirements.txt pyinstaller
python scripts/package_desktop.py --bundle   # installer in src-tauri/target/release/bundle/
```

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
