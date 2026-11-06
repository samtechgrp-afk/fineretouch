'use client';

import { useEffect, useRef, useState } from 'react';
import { pool, renderOnServer } from '@/lib/ai';
import { cloudAvailable, runCloudBatch } from '@/lib/cloud';
import { downloadBlob, exportName, renderFullRes, zipBlobs, type ExportOptions } from '@/lib/export';
import { decodeOriginal, getFile, isRawPhoto } from '@/lib/images';
import { useEditor } from '@/lib/store';
import { isFlagged } from './Filmstrip';

type Scope = 'active' | 'selected' | 'keepers' | 'all';
type Engine = 'browser' | 'server' | 'cloud';

export function ExportDialog({ onClose }: { onClose(): void }) {
  const state = useEditor.getState();
  const aiOnline = useEditor((s) => s.aiOnline);
  const [scope, setScope] = useState<Scope>(state.selected.length > 1 ? 'selected' : 'all');
  const [engine, setEngine] = useState<Engine>('browser');
  const [opts, setOpts] = useState<ExportOptions>({ format: 'jpeg', quality: 92 });
  const [cloud, setCloud] = useState(false);
  const [status, setStatus] = useState<{ label: string; done: number; total: number } | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [links, setLinks] = useState<Array<{ imageId: string; url: string }>>([]);
  const abortRef = useRef<AbortController | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    dialogRef.current?.showModal();
    void cloudAvailable().then(setCloud);
    return () => abortRef.current?.abort();
  }, []);

  const ids = (() => {
    const s = useEditor.getState();
    if (scope === 'active') return s.activeId ? [s.activeId] : [];
    if (scope === 'selected') return s.selected;
    if (scope === 'keepers') return s.order.filter((id) => !isFlagged(s.photos[id]));
    return s.order;
  })();

  const run = async () => {
    setError(null);
    setResult(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const photos = useEditor.getState().photos;
    try {
      if (engine === 'cloud') {
        const items = ids.map((id) => ({ id, file: getFile(id)!, adjustments: photos[id].adjustments }));
        const { progress, files } = await runCloudBatch(
          items,
          { format: opts.format, quality: opts.quality, maxLongEdge: opts.maxLongEdge },
          (phase, done, total) => setStatus({ label: phase === 'upload' ? 'Uploading originals' : 'Rendering on GPU workers', done, total }),
          ctrl.signal,
        );
        // Downloads come straight from object storage via short-lived signed links.
        setLinks(files);
        setResult(`${progress.done} rendered${progress.failed ? `, ${progress.failed} failed` : ''}.`);
        return;
      }
      const used = new Set<string>();
      const out: Array<{ name: string; blob: Blob }> = [];
      let done = 0;
      setStatus({ label: 'Rendering full resolution', done: 0, total: ids.length });
      // The GPU path is serial (one big texture set at a time); the server path runs a few in parallel.
      await pool(ids, engine === 'server' ? 3 : 1, async (id) => {
        if (ctrl.signal.aborted) return;
        const p = photos[id];
        const file = getFile(id)!;
        const blob =
          engine === 'server'
            ? await renderOnServer(file, p.adjustments, opts)
            : await renderFullRes(() => decodeOriginal(id), p.adjustments, { id, analysis: p.analysis, sourceStats: p.analysis?.colorStats ?? p.localStats }, opts);
        out.push({ name: exportName(p.name, opts.format, used), blob });
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
      if ((e as Error).name !== 'AbortError') setError((e as Error).message);
    } finally {
      setStatus(null);
      abortRef.current = null;
    }
  };

  const busy = status !== null;

  return (
    <dialog
      ref={dialogRef}
      onClose={onClose}
      onCancel={(e) => busy && e.preventDefault()}
      className="m-auto w-[min(440px,calc(100vw-32px))] rounded-xl border border-ink-700 bg-ink-900 p-0 text-ink-100 backdrop:bg-black/60"
    >
      <div className="space-y-4 p-5">
        <h2 className="text-sm font-semibold">Export</h2>
        <Field label="Photos">
          <Seg<Scope> value={scope} onChange={setScope} options={[['active', 'Current'], ['selected', `Selected (${state.selected.length})`], ['keepers', 'Unflagged'], ['all', `All (${state.order.length})`]]} />
        </Field>
        <Field label="Format">
          <Seg<ExportOptions['format']> value={opts.format} onChange={(format) => setOpts({ ...opts, format })} options={[['jpeg', 'JPEG'], ['png', 'PNG'], ['webp', 'WebP']]} />
        </Field>
        {opts.format !== 'png' && (
          <Field label={`Quality ${opts.quality}`}>
            <input type="range" min={50} max={100} value={opts.quality} onChange={(e) => setOpts({ ...opts, quality: Number(e.target.value) })} className="w-full accent-apricot-400" />
          </Field>
        )}
        <Field label="Long edge">
          <select className="field" value={opts.maxLongEdge ?? ''} onChange={(e) => setOpts({ ...opts, maxLongEdge: e.target.value ? Number(e.target.value) : undefined })}>
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
            options={[
              ['browser', 'This device (GPU)'],
              ['server', aiOnline ? 'AI service' : 'AI service (offline)'],
              ['cloud', cloud ? 'Cloud batch' : 'Cloud (not set up)'],
            ]}
            disabled={{ server: !aiOnline, cloud: !cloud }}
          />
        </Field>
        {engine === 'browser' && ids.some(isRawPhoto) && (
          <p className="text-xs text-ink-400">
            RAW files export from the camera&apos;s embedded JPEG on this device. Choose the AI service or Cloud batch to develop them from the full 16-bit sensor data.
          </p>
        )}
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
        {result && <p className="text-xs text-mint-400">{result}</p>}
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
        {error && <p className="text-xs text-rose-400">{error}</p>}
        <div className="flex justify-end gap-2 pt-1">
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
    </dialog>
  );
}

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
