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
  /** <dataDir>/exports: the default export folder, created by the shell and always writable. */
  exportsDir: string;
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
 * Show a file or folder in Explorer / Finder / the file manager. Only paths the webview
 * may write (picked in a dialog, or the exports folder) can be revealed.
 * Saving itself lives in exportManager.ts (tauri-plugin-dialog + tauri-plugin-fs).
 */
export function revealInFolder(path: string): Promise<void> {
  return invoke<void>('reveal_path', { path });
}
