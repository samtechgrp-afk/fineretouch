'use client';

/**
 * Desktop export destinations: native Save As / folder dialogs (tauri-plugin-dialog) and
 * direct disk writes (tauri-plugin-fs), plus the persisted export preferences.
 *
 * Where files go
 *   - "Ask every time": one photo opens a Save As dialog, several open a folder picker.
 *     Both start in the last folder used.
 *   - "Save to folder": no dialog; files land in the chosen default folder, or in the
 *     app's own exports folder (%APPDATA%\FineRetouch\exports, ~/Library/Application
 *     Support/FineRetouch/exports, ~/.local/share/FineRetouch/exports) if none was chosen.
 *
 * Permissions: the webview can write only (a) inside the app's exports folder, declared in
 * src-tauri/capabilities/default.json, and (b) to files and folders the user picked in a
 * dialog. The dialog plugin adds each pick to the fs scope; tauri-plugin-persisted-scope
 * keeps a picked default folder writable across restarts. If that grant is ever lost, the
 * folder picker opens again instead of failing.
 *
 * In a browser none of this runs: exports download as before (see ExportDialog).
 */

import { desktopInfo } from './desktop';
import { EXT, type ExportOptions } from './export';
import { DEFAULT_WATERMARK, sanitizeWatermark, type WatermarkSettings } from './watermark';

export type ExportDestination = 'ask' | 'folder';

export interface ExportPrefs {
  format: ExportOptions['format'];
  quality: number;
  maxLongEdge: number | null;
  destination: ExportDestination;
  /** Default folder for "Save to folder"; null = the app's exports folder. */
  folder: string | null;
  /** Where the last Save As / folder dialog ended up; dialogs start there next time. */
  lastFolder: string | null;
  /** Open the folder in Explorer / Finder after saving. */
  revealAfterSave: boolean;
  watermark: WatermarkSettings;
}

export const DEFAULT_EXPORT_PREFS: Readonly<ExportPrefs> = Object.freeze({
  format: 'jpeg',
  quality: 92,
  maxLongEdge: null,
  destination: 'ask',
  folder: null,
  lastFolder: null,
  revealAfterSave: false,
  watermark: DEFAULT_WATERMARK,
});

const STORAGE_KEY = 'fineretouch.exportPrefs.v1';

export function sanitizeExportPrefs(input: unknown): ExportPrefs {
  const r = (input && typeof input === 'object' ? input : {}) as Record<string, unknown>;
  const d = DEFAULT_EXPORT_PREFS;
  const str = (v: unknown) => (typeof v === 'string' && v.length > 0 && v.length < 4096 ? v : null);
  return {
    format: r.format === 'png' || r.format === 'webp' || r.format === 'jpeg' ? r.format : d.format,
    quality: typeof r.quality === 'number' && Number.isFinite(r.quality) ? Math.min(100, Math.max(1, Math.round(r.quality))) : d.quality,
    maxLongEdge: typeof r.maxLongEdge === 'number' && r.maxLongEdge >= 64 ? Math.round(r.maxLongEdge) : null,
    destination: r.destination === 'folder' ? 'folder' : 'ask',
    folder: str(r.folder),
    lastFolder: str(r.lastFolder),
    revealAfterSave: typeof r.revealAfterSave === 'boolean' ? r.revealAfterSave : d.revealAfterSave,
    watermark: sanitizeWatermark(r.watermark),
  };
}

export function loadExportPrefs(): ExportPrefs {
  try {
    const raw = typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_KEY) : null;
    return sanitizeExportPrefs(raw ? JSON.parse(raw) : null);
  } catch {
    return { ...DEFAULT_EXPORT_PREFS };
  }
}

export function saveExportPrefs(prefs: ExportPrefs): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // Storage full or blocked (e.g. a large logo): the session keeps working with in-memory prefs.
  }
}

/** The app's own exports folder (created by the desktop shell at startup), or null in a browser. */
export function defaultExportFolder(): string | null {
  return desktopInfo()?.exportsDir ?? null;
}

/** Folder that "Save to folder" writes into right now. */
export function activeExportFolder(prefs: ExportPrefs): string | null {
  return prefs.folder ?? defaultExportFolder();
}

/** Native folder picker. The pick (recursively) becomes writable for this and later sessions. */
export async function pickExportFolder(startIn?: string | null, title = 'Choose export folder'): Promise<string | null> {
  const { open } = await import('@tauri-apps/plugin-dialog');
  const picked = await open({ directory: true, multiple: false, recursive: true, title, defaultPath: startIn ?? defaultExportFolder() ?? undefined });
  return typeof picked === 'string' ? picked : null;
}

export interface ExportFile {
  name: string;
  blob: Blob;
}

/** Where a run of exports goes, decided before anything is rendered. */
export type ExportTarget =
  | { kind: 'file'; path: string; folder: string }
  | { kind: 'folder'; folder: string };

const FILTER_LABEL: Record<ExportOptions['format'], string> = { jpeg: 'JPEG image', png: 'PNG image', webp: 'WebP image' };

/**
 * Resolve the destination first (dialogs open before rendering, so a 500-photo batch never
 * waits on a dialog at the end, and files are written as they finish instead of piling up
 * in memory). Returns null if the user cancelled. `prefs` in the result carries the updated
 * last-used / default folder to persist.
 */
export async function chooseExportTarget(
  firstName: string,
  count: number,
  prefs: ExportPrefs,
): Promise<{ target: ExportTarget; prefs: ExportPrefs } | null> {
  const path = await import('@tauri-apps/api/path');

  if (prefs.destination === 'folder') {
    const folder = activeExportFolder(prefs);
    if (!folder) throw new Error('No export folder is set');
    try {
      await ensureFolder(folder);
      return { target: { kind: 'folder', folder }, prefs };
    } catch (e) {
      if (!isScopeError(e)) throw e;
      // The grant for this folder is gone (e.g. app data was reset): ask once more.
      const again = await pickExportFolder(folder, 'Allow FineRetouch to save into this folder');
      if (!again) return null;
      return { target: { kind: 'folder', folder: again }, prefs: { ...prefs, folder: again } };
    }
  }

  const start = prefs.lastFolder ?? activeExportFolder(prefs);
  if (count === 1) {
    const { save } = await import('@tauri-apps/plugin-dialog');
    const ext = firstName.split('.').pop() ?? EXT[prefs.format];
    const chosen = await save({
      title: 'Export photo',
      defaultPath: start ? await path.join(start, firstName) : firstName,
      filters: [{ name: FILTER_LABEL[prefs.format] ?? 'Image', extensions: ext === 'jpg' ? ['jpg', 'jpeg'] : [ext] }],
    });
    if (!chosen) return null;
    const folder = await path.dirname(chosen);
    return { target: { kind: 'file', path: chosen, folder }, prefs: { ...prefs, lastFolder: folder } };
  }

  const folder = await pickExportFolder(start, `Export ${count} photos to folder`);
  if (!folder) return null;
  return { target: { kind: 'folder', folder }, prefs: { ...prefs, lastFolder: folder } };
}

/** Writes rendered exports into a target, one at a time, never overwriting inside a folder. */
export class ExportWriter {
  private taken = new Set<string>();
  readonly paths: string[] = [];

  constructor(readonly target: ExportTarget) {}

  async write(file: ExportFile): Promise<string> {
    const { join } = await import('@tauri-apps/api/path');
    const { exists } = await import('@tauri-apps/plugin-fs');
    let dest: string;
    if (this.target.kind === 'file') {
      // The Save As dialog already confirmed any overwrite.
      dest = this.target.path;
    } else {
      const name = await freeName(this.target.folder, file.name, this.taken, (p) => exists(p), join);
      dest = await join(this.target.folder, name);
    }
    await writeBytes(dest, file.blob);
    this.paths.push(dest);
    return dest;
  }
}

async function ensureFolder(folder: string): Promise<void> {
  const { exists, mkdir } = await import('@tauri-apps/plugin-fs');
  if (!(await exists(folder))) await mkdir(folder, { recursive: true });
}

async function writeBytes(target: string, blob: Blob): Promise<void> {
  const { writeFile } = await import('@tauri-apps/plugin-fs');
  // Sent as the raw IPC body, so 50 MB PNGs do not go through JSON.
  await writeFile(target, new Uint8Array(await blob.arrayBuffer()));
}

/** First of name.ext, name_2.ext, name_3.ext, ... that is neither on disk nor taken in this batch. */
export async function freeName(
  folder: string,
  name: string,
  taken: Set<string>,
  existsFn: (path: string) => Promise<boolean>,
  join: (...parts: string[]) => Promise<string>,
): Promise<string> {
  const dot = name.lastIndexOf('.');
  const base = dot > 0 ? name.slice(0, dot) : name;
  const ext = dot > 0 ? name.slice(dot) : '';
  for (let i = 1; i < 10_000; i++) {
    const candidate = i === 1 ? name : `${base}_${i}${ext}`;
    if (taken.has(candidate.toLowerCase())) continue;
    if (await existsFn(await join(folder, candidate))) continue;
    taken.add(candidate.toLowerCase());
    return candidate;
  }
  throw new Error(`Too many files named ${name} in ${folder}`);
}

export function isScopeError(e: unknown): boolean {
  const msg = String((e as Error)?.message ?? e);
  return /forbidden|not allowed|scope/i.test(msg);
}
