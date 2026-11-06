'use client';

/**
 * Bridge to the Tauri desktop shell (src-tauri). In a browser every function here is a
 * no-op or returns null, so the web build keeps working unchanged.
 *
 * The shell injects `window.__FINERETOUCH_DESKTOP__` before any page script runs
 * (src-tauri/src/lib.rs), so the engine address is known synchronously at module load.
 */

export interface DesktopInfo {
  /** Base URL of the local engine sidecar, e.g. http://127.0.0.1:51234 */
  engineUrl: string;
  /** Per-launch secret the engine requires in the X-FineRetouch-Token header. */
  engineToken: string;
  /** %APPDATA%/FineRetouch, ~/Library/Application Support/FineRetouch or ~/.local/share/FineRetouch */
  dataDir: string;
  version: string;
  platform: 'windows' | 'macos' | 'linux' | string;
}

declare global {
  interface Window {
    __FINERETOUCH_DESKTOP__?: DesktopInfo;
    __TAURI_INTERNALS__?: unknown;
  }
}

export function desktopInfo(): DesktopInfo | null {
  return typeof window !== 'undefined' ? (window.__FINERETOUCH_DESKTOP__ ?? null) : null;
}

export function isDesktop(): boolean {
  return typeof window !== 'undefined' && Boolean(window.__TAURI_INTERNALS__);
}

async function invoke<T>(cmd: string, args?: Record<string, unknown> | Uint8Array, headers?: Record<string, string>): Promise<T> {
  const core = await import('@tauri-apps/api/core');
  return core.invoke<T>(cmd, args, headers ? { headers } : undefined);
}

export interface EngineStatus {
  state: 'starting' | 'ready' | 'stopped' | 'external';
  url: string;
  message?: string;
}

export function engineStatus(): Promise<EngineStatus> {
  return invoke<EngineStatus>('engine_status');
}

/** Restart the engine sidecar (after a crash, or from a "Restart engine" button). */
export function restartEngine(): Promise<EngineStatus> {
  return invoke<EngineStatus>('restart_engine');
}

/**
 * Save exports through native dialogs instead of a browser download (which webviews either
 * ignore or drop into Downloads without asking).
 *
 * One file: a Save As dialog. Several files: a folder picker, then every file is written
 * into that folder (no ZIP needed on the desktop). Returns where the files went, or null if
 * the user cancelled.
 */
export async function saveExports(files: Array<{ name: string; blob: Blob }>): Promise<string | null> {
  if (!files.length) return null;
  if (files.length === 1) {
    const target = await invoke<string | null>('choose_save_path', { defaultName: files[0].name });
    if (!target) return null;
    await writeBytes(target, '', files[0].blob);
    return target;
  }
  const dir = await invoke<string | null>('choose_export_folder');
  if (!dir) return null;
  for (const f of files) await writeBytes(dir, f.name, f.blob);
  return dir;
}

/**
 * Bytes travel as the raw IPC body (not JSON), which keeps 50 MB TIFF/PNG exports fast.
 * The shell only writes into paths the user picked in its own dialogs.
 */
async function writeBytes(target: string, name: string, blob: Blob): Promise<void> {
  const body = new Uint8Array(await blob.arrayBuffer());
  await invoke<void>('write_export', body, {
    'x-fineretouch-target': encodeURIComponent(target),
    'x-fineretouch-name': encodeURIComponent(name),
  });
}

/** Show a file or folder in Explorer / Finder / the file manager. */
export function revealInFolder(path: string): Promise<void> {
  return invoke<void>('reveal_path', { path });
}
