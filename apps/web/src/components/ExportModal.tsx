'use client';

import * as Slider from '@radix-ui/react-slider';
import * as Switch from '@radix-ui/react-switch';
import { useEffect, useRef, useState } from 'react';
import { pool, renderOnServer } from '@/lib/ai';
import { cloudAvailable, runCloudBatch } from '@/lib/cloud';
import { MIME, downloadBlob, exportName, renderFullRes, zipBlobs, type ExportOptions } from '@/lib/export';
import {
  ExportWriter,
  activeExportFolder,
  chooseExportTarget,
  defaultExportFolder,
  loadExportPrefs,
  pickExportFolder,
  saveExportPrefs,
  type ExportPrefs,
} from '@/lib/exportManager';
import { isDesktop, revealInFolder } from '@/lib/desktop';
import { decodeOriginal, getFile, isRawPhoto } from '@/lib/images';
import { useEditor } from '@/lib/store';
import {
  WATERMARK_ANCHORS,
  WATERMARK_FONTS,
  drawWatermark,
  logoFromFile,
  stampBlob,
  watermarkActive,
  type WatermarkAnchor,
  type WatermarkSettings,
} from '@/lib/watermark';
import { isFlagged } from './Filmstrip';

type Scope = 'active' | 'selected' | 'keepers' | 'all';
type Engine = 'browser' | 'server' | 'cloud';

/**
 * Export: what to render, how to encode it, where it goes (native dialogs or a default
 * folder on the desktop, downloads in a browser) and an optional watermark burned into
 * the full-resolution pixels. Settings persist between exports.
 */
export function ExportModal({ onClose }: { onClose(): void }) {
  const state = useEditor.getState();
  const aiOnline = useEditor((s) => s.aiOnline);
  const activePhoto = useEditor((s) => (s.activeId ? s.photos[s.activeId] : null));
  const [scope, setScope] = useState<Scope>(state.selected.length > 1 ? 'selected' : 'all');
  const [engine, setEngine] = useState<Engine>('browser');
  const [prefs, setPrefs] = useState<ExportPrefs>(loadExportPrefs);
  const [cloud, setCloud] = useState(false);
  const [status, setStatus] = useState<{ label: string; done: number; total: number } | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [links, setLinks] = useState<Array<{ imageId: string; url: string }>>([]);
  const [savedTo, setSavedTo] = useState<string | null>(null);
  const desktop = isDesktop();
  const abortRef = useRef<AbortController | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    dialogRef.current?.showModal();
    void cloudAvailable().then(setCloud);
    return () => abortRef.current?.abort();
  }, []);
  useEffect(() => saveExportPrefs(prefs), [prefs]);

  const patch = (p: Partial<ExportPrefs>) => setPrefs((cur) => ({ ...cur, ...p }));
  const patchWm = (p: Partial<WatermarkSettings>) => setPrefs((cur) => ({ ...cur, watermark: { ...cur.watermark, ...p } }));
  const opts: ExportOptions = { format: prefs.format, quality: prefs.quality, maxLongEdge: prefs.maxLongEdge ?? undefined, watermark: prefs.watermark };
  const wmOn = watermarkActive(prefs.watermark);

  const ids = (() => {
    const s = useEditor.getState();
    if (scope === 'active') return s.activeId ? [s.activeId] : [];
    if (scope === 'selected') return s.selected;
    if (scope === 'keepers') return s.order.filter((id) => !isFlagged(s.photos[id]));
    return s.order;
  })();

  /** Render one photo to an encoded blob, watermark included. */
  const renderOne = async (id: string): Promise<Blob> => {
    const p = useEditor.getState().photos[id];
    if (engine === 'server') {
      // Lossless from the engine when stamping, so the file is encoded only once more.
      const fmt = wmOn ? 'png' : opts.format;
      const blob = await renderOnServer(getFile(id)!, p.adjustments, { format: fmt, quality: opts.quality, maxLongEdge: opts.maxLongEdge });
      return wmOn ? stampBlob(blob, prefs.watermark, MIME[opts.format], opts.quality) : blob;
    }
    return renderFullRes(() => decodeOriginal(id), p.adjustments, { id, analysis: p.analysis, sourceStats: p.analysis?.colorStats ?? p.localStats }, opts);
  };

  const run = async () => {
    setError(null);
    setResult(null);
    setSavedTo(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const photos = useEditor.getState().photos;
    const used = new Set<string>();
    const names = new Map(ids.map((id) => [id, exportName(photos[id].name, opts.format, used)]));
    try {
      if (engine === 'cloud') {
        const items = ids.map((id) => ({ id, file: getFile(id)!, adjustments: photos[id].adjustments }));
        const { progress, files } = await runCloudBatch(
          items,
          { format: opts.format, quality: opts.quality, maxLongEdge: opts.maxLongEdge },
          (phase, done, total) => setStatus({ label: phase === 'upload' ? 'Uploading originals' : 'Rendering on GPU workers', done, total }),
          ctrl.signal,
        );
        setLinks(files);
        setResult(`${progress.done} rendered${progress.failed ? `, ${progress.failed} failed` : ''}.`);
        return;
      }

      if (desktop) {
        // Destination first: dialogs open now, then every photo is written as soon as it renders.
        const chosen = await chooseExportTarget(names.get(ids[0])!, ids.length, prefs);
        if (!chosen) {
          setResult('Export cancelled; nothing was saved.');
          return;
        }
        setPrefs(chosen.prefs);
        const writer = new ExportWriter(chosen.target);
        let done = 0;
        setStatus({ label: 'Rendering and saving', done: 0, total: ids.length });
        // The GPU path is serial (one big texture set at a time); the engine path runs a few in parallel.
        await pool(ids, engine === 'server' ? 3 : 1, async (id) => {
          if (ctrl.signal.aborted) return;
          await writer.write({ name: names.get(id)!, blob: await renderOne(id) });
          setStatus({ label: 'Rendering and saving', done: ++done, total: ids.length });
        });
        const where = chosen.target.kind === 'file' ? chosen.target.path : chosen.target.folder;
        setSavedTo(where);
        const n = writer.paths.length;
        setResult(`${ctrl.signal.aborted ? 'Stopped after' : 'Saved'} ${n} photo${n === 1 ? '' : 's'} to ${chosen.target.kind === 'file' ? where : chosen.target.folder}`);
        if (chosen.prefs.revealAfterSave && n) void revealInFolder(where).catch(() => undefined);
        return;
      }

      const out: Array<{ name: string; blob: Blob }> = [];
      let done = 0;
      setStatus({ label: 'Rendering full resolution', done: 0, total: ids.length });
      await pool(ids, engine === 'server' ? 3 : 1, async (id) => {
        if (ctrl.signal.aborted) return;
        out.push({ name: names.get(id)!, blob: await renderOne(id) });
        setStatus({ label: 'Rendering full resolution', done: ++done, total: ids.length });
      });
      if (ctrl.signal.aborted) return;
      if (out.length === 1) downloadBlob(out[0].blob, out[0].name);
      else {
        setStatus({ label: 'Packing ZIP', done: ids.length, total: ids.length });
        downloadBlob(await zipBlobs(out), `fineretouch_${new Date().toISOString().slice(0, 10)}.zip`);
      }
      setResult(`Exported ${out.length} photo${out.length === 1 ? '' : 's'}.`);
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError(String((e as Error)?.message ?? e));
    } finally {
      setStatus(null);
      abortRef.current = null;
    }
  };

  const busy = status !== null;
  const folder = activeExportFolder(prefs);

  return (
    <dialog
      ref={dialogRef}
      onClose={onClose}
      onCancel={(e) => busy && e.preventDefault()}
      className="m-auto max-h-[calc(100dvh-32px)] w-[min(760px,calc(100vw-32px))] overflow-hidden rounded-xl border border-ink-700 bg-ink-900 p-0 text-ink-100 backdrop:bg-black/60"
    >
      <div className="flex max-h-[calc(100dvh-34px)] flex-col">
        <h2 className="border-b border-ink-800 px-5 py-3 text-sm font-semibold">Export</h2>
        <div className="grid min-h-0 flex-1 gap-5 overflow-y-auto p-5 md:grid-cols-2">
          <div className="space-y-4">
            <Field label="Photos">
              <Seg<Scope> value={scope} onChange={setScope} options={[['active', 'Current'], ['selected', `Selected (${state.selected.length})`], ['keepers', 'Unflagged'], ['all', `All (${state.order.length})`]]} />
            </Field>
            <Field label="Format">
              <Seg<ExportOptions['format']> value={prefs.format} onChange={(format) => patch({ format })} options={[['jpeg', 'JPEG'], ['png', 'PNG'], ['webp', 'WebP']]} />
            </Field>
            {prefs.format !== 'png' && (
              <Field label={`Quality ${prefs.quality}`}>
                <input type="range" min={50} max={100} value={prefs.quality} onChange={(e) => patch({ quality: Number(e.target.value) })} className="w-full accent-apricot-400" aria-label="Quality" />
              </Field>
            )}
            <Field label="Long edge">
              <select className="field" value={prefs.maxLongEdge ?? ''} onChange={(e) => patch({ maxLongEdge: e.target.value ? Number(e.target.value) : null })} aria-label="Long edge">
                <option value="">Original size</option>
                <option value="4096">4096 px</option>
                <option value="2048">2048 px (web)</option>
                <option value="1080">1080 px (social)</option>
              </select>
            </Field>
            <Field label="Processing">
              <Seg<Engine>
                value={engine}
                onChange={setEngine}
                options={
                  desktop
                    ? [
                        ['browser', 'GPU preview engine'],
                        ['server', aiOnline ? 'Local AI engine' : 'Local AI engine (starting)'],
                      ]
                    : [
                        ['browser', 'This device (GPU)'],
                        ['server', aiOnline ? 'AI service' : 'AI service (offline)'],
                        ['cloud', cloud ? 'Cloud batch' : 'Cloud (not set up)'],
                      ]
                }
                disabled={{ server: !aiOnline, cloud: !cloud }}
              />
            </Field>
            {engine === 'browser' && ids.some(isRawPhoto) && (
              <p className="text-xs text-ink-400">
                {desktop
                  ? 'RAW files export from the camera’s embedded JPEG with the GPU engine. Choose the local AI engine to develop them from the full 16-bit sensor data.'
                  : 'RAW files export from the camera’s embedded JPEG on this device. Choose the AI service or Cloud batch to develop them from the full 16-bit sensor data.'}
              </p>
            )}
            {engine !== 'browser' && (
              <p className="text-xs text-ink-400">Spot healing, stray hair and bokeh run at full strength on the {desktop ? 'local AI engine' : 'AI service'}; the GPU engine uses the faster preview versions.</p>
            )}

            <Field label="Save to">
              {desktop ? (
                <div className="space-y-2">
                  <Seg<ExportPrefs['destination']>
                    value={prefs.destination}
                    onChange={(destination) => patch({ destination })}
                    options={[['ask', 'Ask every time'], ['folder', 'Default folder']]}
                  />
                  <div className="flex items-center gap-2 rounded-md border border-ink-800 bg-ink-850 px-2 py-1.5">
                    <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-ink-300" title={folder ?? ''} dir="rtl">
                      {folder ?? '—'}
                    </span>
                    <button className="btn-ghost shrink-0 px-1.5 py-0.5" onClick={async () => {
                      const picked = await pickExportFolder(folder).catch((e) => (setError(String(e)), null));
                      if (picked) patch({ folder: picked, destination: 'folder' });
                    }}>
                      Change…
                    </button>
                    {folder && (
                      <button className="btn-ghost shrink-0 px-1.5 py-0.5" onClick={() => void revealInFolder(folder).catch((e) => setError(String(e)))}>
                        Show
                      </button>
                    )}
                  </div>
                  <p className="text-[11px] leading-snug text-ink-400">
                    {prefs.destination === 'folder'
                      ? 'Photos save straight into this folder with no dialog; existing files are never overwritten.'
                      : 'One photo opens Save As; several open a folder picker. This folder is used by Default folder.'}
                    {prefs.folder && (
                      <>
                        {' '}
                        <button className="text-apricot-300 hover:underline" onClick={() => patch({ folder: null })} title={defaultExportFolder() ?? ''}>
                          Reset to app folder
                        </button>
                      </>
                    )}
                  </p>
                  <label className="flex items-center gap-2 text-xs text-ink-300">
                    <input type="checkbox" className="accent-apricot-400" checked={prefs.revealAfterSave} onChange={(e) => patch({ revealAfterSave: e.target.checked })} />
                    Open the folder when the export finishes
                  </label>
                </div>
              ) : (
                <p className="text-xs text-ink-400">Your browser&apos;s downloads folder (several photos download as one ZIP).</p>
              )}
            </Field>
          </div>

          <WatermarkEditor wm={prefs.watermark} onChange={patchWm} thumbUrl={activePhoto?.thumbUrl ?? null} cloud={engine === 'cloud'} onError={setError} />
        </div>

        <div className="space-y-2 border-t border-ink-800 px-5 py-3">
          {status && (
            <div aria-live="polite">
              <div className="mb-1 flex justify-between text-xs text-ink-300">
                <span>{status.label}</span>
                <span className="font-mono">
                  {status.done}/{status.total}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded bg-ink-700">
                <div className="h-full bg-apricot-400 transition-[width]" style={{ width: `${(status.done / Math.max(1, status.total)) * 100}%` }} />
              </div>
            </div>
          )}
          {result && <p className="break-all text-xs text-mint-400">{result}</p>}
          {links.length > 0 && (
            <ul className="max-h-32 space-y-0.5 overflow-y-auto text-xs">
              {links.map((l) => (
                <li key={l.imageId}>
                  <a className="text-apricot-300 underline" href={l.url} download>
                    {useEditor.getState().photos[l.imageId]?.name ?? l.imageId}
                  </a>
                </li>
              ))}
            </ul>
          )}
          {error && <p className="break-all text-xs text-rose-400">{error}</p>}
          <div className="flex items-center justify-end gap-2">
            {savedTo && (
              <button className="mr-auto text-xs text-apricot-300 underline-offset-2 hover:underline" onClick={() => void revealInFolder(savedTo).catch((e) => setError(String(e)))}>
                Show in folder
              </button>
            )}
            {busy ? (
              <button className="btn" onClick={() => abortRef.current?.abort()}>
                Cancel
              </button>
            ) : (
              <button className="btn-ghost" onClick={() => dialogRef.current?.close()}>
                Close
              </button>
            )}
            <button className="btn-primary" onClick={run} disabled={busy || !ids.length}>
              Export {ids.length} photo{ids.length === 1 ? '' : 's'}
            </button>
          </div>
        </div>
      </div>
    </dialog>
  );
}

// --- watermark ---------------------------------------------------------------

const ANCHOR_LABEL: Record<WatermarkAnchor, string> = {
  'top-left': 'Top left', 'top-center': 'Top', 'top-right': 'Top right',
  'middle-left': 'Left', center: 'Centre', 'middle-right': 'Right',
  'bottom-left': 'Bottom left', 'bottom-center': 'Bottom', 'bottom-right': 'Bottom right',
};

function WatermarkEditor(props: { wm: WatermarkSettings; onChange(p: Partial<WatermarkSettings>): void; thumbUrl: string | null; cloud: boolean; onError(e: string): void }) {
  const { wm, onChange } = props;
  const logoInput = useRef<HTMLInputElement>(null);
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-[11px] uppercase tracking-wider text-ink-400">Watermark</div>
        <Switch.Root
          checked={wm.enabled}
          onCheckedChange={(enabled) => onChange({ enabled })}
          aria-label="Watermark"
          className="relative h-[18px] w-8 shrink-0 rounded-full bg-ink-700 transition-colors data-[state=checked]:bg-apricot-400"
        >
          <Switch.Thumb className="block h-3.5 w-3.5 translate-x-0.5 rounded-full bg-ink-100 shadow transition-transform data-[state=checked]:translate-x-[15px] data-[state=checked]:bg-ink-950" />
        </Switch.Root>
      </div>
      <WatermarkPreview wm={wm} thumbUrl={props.thumbUrl} />
      <fieldset disabled={!wm.enabled} className="space-y-3 disabled:opacity-50">
        <input className="field" value={wm.text} maxLength={200} placeholder="© 2026 Your Name" onChange={(e) => onChange({ text: e.target.value })} aria-label="Watermark text" />
        <div className="grid grid-cols-[1fr_auto_auto] items-center gap-2">
          <select className="field" value={wm.font} onChange={(e) => onChange({ font: e.target.value })} aria-label="Font">
            {WATERMARK_FONTS.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
          <button className="seg border border-ink-700 font-bold" aria-pressed={wm.bold} onClick={() => onChange({ bold: !wm.bold })} title="Bold">
            B
          </button>
          <label className="flex items-center gap-1.5 text-xs text-ink-300" title="Text colour">
            <input type="color" value={wm.color} onChange={(e) => onChange({ color: e.target.value })} className="h-6 w-7 cursor-pointer rounded border border-ink-700 bg-transparent p-0" aria-label="Text colour" />
          </label>
        </div>
        <div className="grid grid-cols-[auto_1fr] gap-3">
          <div>
            <div className="mb-1 text-[11px] text-ink-400">Position</div>
            <div className="grid w-[72px] grid-cols-3 gap-1" role="radiogroup" aria-label="Position">
              {WATERMARK_ANCHORS.map((a) => (
                <button
                  key={a}
                  role="radio"
                  aria-checked={wm.anchor === a}
                  aria-label={ANCHOR_LABEL[a]}
                  title={ANCHOR_LABEL[a]}
                  onClick={() => onChange({ anchor: a })}
                  className={`h-5 rounded-sm border ${wm.anchor === a ? 'border-apricot-400 bg-apricot-400' : 'border-ink-700 bg-ink-850 hover:border-ink-500'}`}
                />
              ))}
            </div>
          </div>
          <div className="space-y-2">
            <MiniSlider label="Size" value={wm.size} min={0.5} max={15} step={0.1} unit="%" onChange={(size) => onChange({ size })} />
            <MiniSlider label="Opacity" value={wm.opacity} min={0} max={100} step={1} unit="%" onChange={(opacity) => onChange({ opacity })} />
            <MiniSlider label="Margin" value={wm.margin} min={0} max={20} step={0.5} unit="%" onChange={(margin) => onChange({ margin })} />
          </div>
        </div>
        <label className="flex items-center gap-2 text-xs text-ink-300">
          <input type="checkbox" className="accent-apricot-400" checked={wm.shadow} onChange={(e) => onChange({ shadow: e.target.checked })} />
          Soft shadow for contrast
        </label>
        <div className="flex items-center gap-2">
          <input
            ref={logoInput}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            hidden
            onChange={async (e) => {
              const f = e.target.files?.[0];
              e.target.value = '';
              if (!f) return;
              try {
                onChange({ logo: await logoFromFile(f) });
              } catch (err) {
                props.onError((err as Error).message);
              }
            }}
          />
          <button className="btn" onClick={() => logoInput.current?.click()}>
            {wm.logo ? 'Replace logo' : 'Add logo'}
          </button>
          {wm.logo && (
            <>
              <button className="btn-ghost" onClick={() => onChange({ logo: null })}>
                Remove
              </button>
              <div className="min-w-0 flex-1">
                <MiniSlider label="Logo" value={wm.logoSize} min={1} max={40} step={0.5} unit="%" onChange={(logoSize) => onChange({ logoSize })} />
              </div>
            </>
          )}
        </div>
        <p className="text-[11px] leading-snug text-ink-400">
          Sizes are a share of each photo&apos;s short edge, so the mark looks the same on a 1080 px proof and a full-size print.
          {props.cloud && ' Cloud batch exports are not watermarked.'}
        </p>
      </fieldset>
    </div>
  );
}

/** The active photo's thumbnail with the watermark drawn by the same code the export uses. */
function WatermarkPreview({ wm, thumbUrl }: { wm: WatermarkSettings; thumbUrl: string | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!thumbUrl) return setImg(null);
    const i = new Image();
    i.onload = () => setImg(i);
    i.src = thumbUrl;
  }, [thumbUrl]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    let cancelled = false;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const box = 320 * dpr;
    const w = img ? img.naturalWidth : 3;
    const h = img ? img.naturalHeight : 2;
    const s = box / Math.max(w, h);
    c.width = Math.round(w * s);
    c.height = Math.round(h * s);
    const ctx = c.getContext('2d')!;
    if (img) ctx.drawImage(img, 0, 0, c.width, c.height);
    else {
      const g = ctx.createLinearGradient(0, 0, c.width, c.height);
      g.addColorStop(0, '#3a3d44');
      g.addColorStop(1, '#16171a');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, c.width, c.height);
    }
    void drawWatermark(ctx, { ...wm, enabled: true }, c.width, c.height).then(() => {
      if (cancelled) return;
      if (!wm.enabled) {
        // Show a dimmed hint of the mark while it is switched off.
        ctx.fillStyle = 'rgba(13,14,16,0.55)';
        ctx.fillRect(0, 0, c.width, c.height);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [img, wm]);

  return (
    <div className="grid place-items-center rounded-md border border-ink-800 bg-ink-950 p-2">
      <canvas ref={canvas} className="max-h-48 max-w-full rounded-sm" aria-label="Watermark preview" />
    </div>
  );
}

function MiniSlider(props: { label: string; value: number; min: number; max: number; step: number; unit?: string; onChange(v: number): void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-12 shrink-0 text-[11px] text-ink-400">{props.label}</span>
      <Slider.Root
        className="relative flex h-4 min-w-0 flex-1 touch-none select-none items-center"
        min={props.min}
        max={props.max}
        step={props.step}
        value={[props.value]}
        onValueChange={([v]) => props.onChange(v)}
        aria-label={props.label}
      >
        <Slider.Track className="relative h-[3px] grow rounded-full bg-ink-700">
          <Slider.Range className="absolute h-full rounded-full bg-apricot-400" />
        </Slider.Track>
        <Slider.Thumb aria-label={props.label} className="block h-3 w-3 rounded-full border-2 border-ink-900 bg-ink-100 shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-apricot-400" />
      </Slider.Root>
      <span className="w-10 shrink-0 text-right font-mono text-[11px] tabular-nums text-ink-300">
        {Number.isInteger(props.step) ? Math.round(props.value) : props.value.toFixed(1)}
        {props.unit}
      </span>
    </div>
  );
}

// --- shared bits -------------------------------------------------------------

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-[11px] uppercase tracking-wider text-ink-400">{label}</div>
      {children}
    </div>
  );
}

function Seg<T extends string>(props: { value: T; onChange(v: T): void; options: Array<[T, string]>; disabled?: Partial<Record<T, boolean>> }) {
  return (
    <div className="flex flex-wrap gap-1 rounded-md bg-ink-850 p-1" role="group">
      {props.options.map(([v, label]) => (
        <button key={v} className="seg disabled:opacity-40" aria-pressed={props.value === v} onClick={() => props.onChange(v)} disabled={props.disabled?.[v]}>
          {label}
        </button>
      ))}
    </div>
  );
}
