# FineRetouch desktop (offline)

The desktop app is the same editor packaged with **Tauri v2** and the AI service frozen
into a local **engine sidecar**. Models ship inside the installer, so after installation
nothing needs the network: no account, no cloud, no model downloads.

```
┌────────────────────────── FineRetouch.app / .exe / .deb ──────────────────────────┐
│  Tauri shell (src-tauri, Rust)                                                    │
│   ├─ webview: static Next.js export (apps/web/out)  ──fetch + token──┐            │
│   │    WebGL2 preview, rawParser.ts embedded-JPEG/EXIF import        │            │
│   ├─ native Save As / folder dialogs (src/files.rs)                  ▼            │
│   └─ engine lifecycle (src/engine.rs) ──spawns──►  fineretouch-engine sidecar     │
│                                                    127.0.0.1:<random port>        │
│                                                    FastAPI app (services/ai/app)  │
│                                                    raw_engine.py (rawpy, 16-bit)  │
│                                                    retouch_engine.py, MediaPipe   │
│  resources/models/  face_landmarker.task · selfie_multiclass.tflite · midas_small │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## Build an installer

Prerequisites on the build machine: Node 22, Python 3.12+, Rust stable, and on Linux the
webview libraries (`libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev libssl-dev`).

```bash
npm install
pip install -r services/ai/requirements.txt pyinstaller
python scripts/package_desktop.py      # models + frozen engine + self-test
npm run desktop:build                  # installer in src-tauri/target/release/bundle/
```

`python scripts/package_desktop.py --bundle` does both steps. PyInstaller does not
cross-compile, so build each OS on that OS; `.github/workflows/desktop.yml` does this for
Windows, macOS (Apple Silicon and Intel) and Linux and uploads the installers.

| Output | Where |
| :--- | :--- |
| Engine sidecar | `src-tauri/binaries/fineretouch-engine-<target-triple>[.exe]` (~190 MB, one file) |
| Bundled models | `src-tauri/resources/models/` (~87 MB) → app resource `models/` |
| Installers | `src-tauri/target/release/bundle/{nsis,msi,dmg,macos,deb,rpm,appimage}` |

The Windows installer embeds the WebView2 offline installer so it also works on machines
that have never been online. That adds about 130 MB; switch
`bundle.windows.webviewInstallMode` to `embedBootstrapper` if every target PC has internet.

## Develop

```bash
# Terminal 1: the engine from source (hot reload)
cd services/ai && FINERETOUCH_ENGINE_TOKEN=dev uvicorn app.main:app --port 8000 --reload
# Terminal 2: the shell, pointed at it instead of spawning the sidecar
FINERETOUCH_ENGINE_URL=http://localhost:8000 FINERETOUCH_ENGINE_TOKEN=dev npm run desktop:dev
```

`tauri dev` still checks that a sidecar file exists, so run `package_desktop.py` once (or
create an empty placeholder named like the one above).

## How it fits together

* **Static export.** `FINERETOUCH_DESKTOP=1` (set by `npm run build:desktop`, and detected
  from the `TAURI_ENV_*` variables the Tauri CLI sets) switches `apps/web/next.config.ts` to
  `output: 'export'` with `pageExtensions: ['tsx']`. Pages are `.tsx` and the `/api` route
  handlers are `.ts`, so the server-only routes (Postgres, Redis, S3) drop out of the build.
  Cloud batch is hidden in the export dialog; the web and Docker builds are unchanged.
* **Engine start-up.** On launch the shell picks a free loopback port and a random token,
  then starts the sidecar with `--port`, `--parent-pid`, `--models-dir <resources>/models`
  and `--data-dir`. It injects `window.__FINERETOUCH_DESKTOP__ = { engineUrl, engineToken, … }`
  before the page loads; `apps/web/src/lib/ai.ts` uses it in place of `NEXT_PUBLIC_AI_URL`.
  The editor's existing health poll flips the "AI ready" pill once the engine answers
  (first launch takes a few seconds while the one-file engine unpacks).
* **Security.** The engine binds to 127.0.0.1 only, accepts only the webview origins via
  CORS, and rejects any request without the per-launch `X-FineRetouch-Token`, so other
  programs and websites open in a browser cannot drive it. The webview has no general
  file-system access: it can write only to a path the user picked in the shell's Save As or
  folder dialog in this session. `shell:allow-execute` is scoped to the engine sidecar with
  the single argument `--version`; the Rust core, not the webview, starts the real engine.
* **Shutdown.** The shell kills the sidecar on exit, and the engine also exits by itself
  within two seconds if the shell's process disappears (crash, force quit).
* **Imports** keep using `<input type="file">` and drag and drop, which already open the
  native picker and give the editor zero-copy `File` objects (large RAWs never cross IPC).
  `dragDropEnabled: false` on the window lets HTML5 drops reach the editor on Windows.
* **Exports** use native dialogs: Save As for one photo, a folder picker for several (files
  are written individually, no ZIP). "Show in folder" opens Explorer / Finder.

## Local data

| OS | Folder |
| :--- | :--- |
| Windows | `%APPDATA%\FineRetouch` |
| macOS | `~/Library/Application Support/FineRetouch` |
| Linux | `~/.local/share/FineRetouch` |

`logs/engine.log` (engine), `logs/engine-console.log` (sidecar stdout/stderr as captured by
the shell) and `cache/` (engine temp files). Presets and settings live in the webview's own
storage.

## Code signing

Unsigned builds work but trigger SmartScreen (Windows) and Gatekeeper (macOS). For release,
set the Tauri signing variables (`APPLE_CERTIFICATE`, `APPLE_SIGNING_IDENTITY`,
`APPLE_ID`/`APPLE_PASSWORD` for notarisation; a Windows `certificateThumbprint` or
`signCommand`) in the workflow secrets. On macOS the sidecar is signed together with the app.
