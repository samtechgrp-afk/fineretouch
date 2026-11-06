# Building the FineRetouch desktop installers

Two ways to get installers: let GitHub Actions build every platform, or build on your own
machine. Each OS has to build its own installer, because the AI engine is frozen with
PyInstaller and PyInstaller cannot cross-compile. A Windows PC makes the Windows
installer, a Mac makes the macOS one.

| Platform | Output | CI artifact (both formats together) |
| :--- | :--- | :--- |
| Windows x64 | `FineRetouch_0.4.0_x64-setup.exe` (NSIS) and `FineRetouch_0.4.0_x64_en-US.msi`, each with the WebView2 offline runtime | `FineRetouch-Windows-Installer`, ~850 MB |
| macOS Apple Silicon | `FineRetouch_0.4.0_aarch64.dmg` and `FineRetouch.app` | `FineRetouch-macOS-Installer`, ~430 MB |
| macOS Intel | `FineRetouch_0.4.0_x64.dmg` and `FineRetouch.app` | `FineRetouch-macOS-Intel-Installer`, ~740 MB |
| Linux x64 | `.deb`, `.rpm` and `.AppImage` | `FineRetouch-Linux-Installer`, ~920 MB |

Each artifact holds every format for its platform, so a single installer is a fraction of
the artifact size. The Intel build is larger because the last MediaPipe release for Intel
Macs (0.10.21) pulls in extra libraries (JAX, a second OpenCV).

There is no single "universal" macOS build: MediaPipe and OpenCV publish separate arm64
and x86_64 wheels, so the engine cannot be frozen as a universal2 binary. Ship the arm64
DMG to M1–M4 Macs and the Intel DMG to older ones.

## 1. GitHub Actions (recommended)

Workflow: `.github/workflows/desktop.yml`.

* **Run it:** GitHub → **Actions** → **Desktop installers** → **Run workflow** → pick the
  branch. (The button appears once the workflow is on the default branch; until then it
  also runs automatically on pull requests that change the desktop files.)
* **Download:** open the finished run and scroll to **Artifacts**:
  `FineRetouch-Windows-Installer`, `FineRetouch-macOS-Installer` (Apple Silicon),
  `FineRetouch-macOS-Intel-Installer`, `FineRetouch-Linux-Installer`.
* **Release:** push a tag such as `v0.4.0`; the same installers are attached to a draft
  GitHub release you can review and publish.

Each job: checkout → Python 3.12 → Node 22 → Rust stable → system packages (Linux) →
`npm ci` → `pip install` → `python scripts/package_desktop.py` (engine + models +
self-test) → `tauri-apps/tauri-action@v1` (`tauri build`) → upload artifact.

## 2. Build locally

### Windows (PowerShell)

Install once:

* Python 3.12 from python.org (tick "Add python.exe to PATH")
* Node.js 22 LTS
* Rust: `winget install Rustlang.Rustup`, then `rustup default stable-msvc`
* Visual Studio 2022 Build Tools with **Desktop development with C++**
  (`winget install Microsoft.VisualStudio.2022.BuildTools`, then add that workload)

Build:

```powershell
git clone https://github.com/samtechgrp-afk/fineretouch.git
cd fineretouch
git checkout claude/project-thread-omsr2h   # or the branch the desktop app is merged into

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r services\ai\requirements.txt pyinstaller
npm ci

python scripts\package_desktop.py
npx tauri build
```

CMD instead of PowerShell: use `.venv\Scripts\activate.bat`, the rest is the same
(`python scripts\package_desktop.py && npx tauri build`).

Installers: `src-tauri\target\release\bundle\nsis\*.exe` and `...\bundle\msi\*.msi`.

### macOS (Terminal)

Install once:

```bash
xcode-select --install
brew install python@3.12 node@22
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Build:

```bash
git clone https://github.com/samtechgrp-afk/fineretouch.git
cd fineretouch
git checkout claude/project-thread-omsr2h   # or the branch the desktop app is merged into

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r services/ai/requirements.txt pyinstaller
npm ci

python3 scripts/package_desktop.py && npx tauri build
```

Installers: `src-tauri/target/release/bundle/dmg/*.dmg` and `.../bundle/macos/FineRetouch.app`.
The build is for the Mac you run it on (Apple Silicon or Intel).

### Linux

```bash
sudo apt-get install -y libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev \
  libxdo-dev libssl-dev patchelf libegl1 libgles2
python3 -m pip install -r services/ai/requirements.txt pyinstaller
npm ci
python3 scripts/package_desktop.py && npx tauri build
```

### Useful script flags

```bash
python scripts/package_desktop.py --bundle                # also runs the installer build
python scripts/package_desktop.py --skip-optional-models  # face model only (~4 MB), no depth bokeh
python scripts/package_desktop.py --offline               # never download; use services/ai/models
python scripts/package_desktop.py --models-only           # just stage models
```

The first run downloads the three models (~87 MB) into `services/ai/models/`; that is the
only step that needs internet. The installed app never does.

## 3. Opening the unsigned builds

These builds are not code signed, so the OS warns the first time.

**Windows SmartScreen:** "Windows protected your PC" → **More info** → **Run anyway**.
If the browser blocks the download, choose **Keep**. A few antivirus tools flag one-file
PyInstaller programs; if `fineretouch-engine.exe` is quarantined, restore it and allow it.

**macOS Gatekeeper:** the app is ad-hoc signed (`signingIdentity: "-"`), which Apple
Silicon needs for it to run at all. Open the DMG, drag FineRetouch to Applications, then:

1. Double-click FineRetouch. macOS says it cannot verify the developer → **Done**.
2. System Settings → **Privacy & Security** → scroll down → **Open Anyway** → confirm.

Or from Terminal: `xattr -dr com.apple.quarantine /Applications/FineRetouch.app`.

For public distribution, sign and notarise instead: add an Apple Developer ID certificate
(`APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`, `APPLE_ID`,
`APPLE_PASSWORD`, `APPLE_TEAM_ID`) and a Windows code-signing certificate as repository
secrets; tauri-action picks them up. See docs/DESKTOP.md.

## 4. Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `resource path binaries/fineretouch-engine-<triple> doesn't exist` | Run `python scripts/package_desktop.py` before `tauri build`. |
| `self-test failed` | Read the printed log; usually a missing package in the venv (`pip install -r services/ai/requirements.txt pyinstaller`). |
| "AI starting" never turns into "AI ready" | Check `%APPDATA%\FineRetouch\logs\engine-console.log` (Windows) or `~/Library/Application Support/FineRetouch/logs/` (macOS). |
| DMG step times out on CI | Re-run the job; tauri-action already retries once. |
