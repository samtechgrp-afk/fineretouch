'use client';

import { lookFromCube, lutToCube, decodeLut, paletteSyncAdjustments, sanitizeAdjustments, type LookLut } from '@fineretouch/shared';
import { useCallback, useEffect, useRef, useState } from 'react';
import { fitLook } from '@/lib/ai';
import {
  batchAction,
  captureReferenceLook,
  createBatchJob,
  formatEta,
  listBatchJobs,
  pickSourceFolders,
  removeBatchJob,
  setThermalProfile,
  type BatchJob,
  type BatchSyncMode,
  type ThermalProfile,
} from '@/lib/batchSync';
import { desktopInfo, isDesktop, revealInFolder } from '@/lib/desktop';
import { downloadBlob } from '@/lib/export';
import { pickExportFolder } from '@/lib/exportManager';
import { useEditor, type Photo } from '@/lib/store';
import { Heading, RangeRow, Segmented } from './controls';

/**
 * Batch Sync: grade one reference photo, then carry that look across the shoot.
 *
 *  Sync mode   Smart (your edits plus per-photo exposure / white-balance compensation),
 *              Exact copy, or Palette match (colour and tone matched to the finished reference).
 *  Look LUT    capture the reference's grade as a 3D LUT (or import a .cube), apply it to
 *              other photos, save it as .cube for other software.
 *  Folder queue (desktop) the engine's local SQLite queue: whole folders of RAW / JPEG
 *              files rendered in the background with a thermal profile, progress, pause,
 *              resume and retry. Interrupted jobs pick up where they left off.
 */
export function BatchSyncPanel({ photo }: { photo: Photo }) {
  const [mode, setMode] = useState<BatchSyncMode>('smart');
  const [amount, setAmount] = useState(100);
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const selected = useEditor((s) => s.selected);
  const order = useEditor((s) => s.order);
  const targets = (selected.length > 1 ? selected : order).filter((id) => id !== photo.id);
  const heroStats = photo.analysis?.colorStats ?? photo.localStats;

  const syncInEditor = async () => {
    setBusy(true);
    setNote(null);
    try {
      const st = useEditor.getState();
      if (mode === 'smart') {
        const n = st.smartSync();
        setNote(`Smart-synced ${n} photo${n === 1 ? '' : 's'} to ${photo.name}.`);
      } else if (mode === 'copy') {
        st.updateAdjustments(sanitizeAdjustments(photo.adjustments), targets);
        setNote(`Copied the edits to ${targets.length} photo${targets.length === 1 ? '' : 's'}.`);
      } else {
        const look = await captureReferenceLook(photo);
        st.updateAdjustments(paletteSyncAdjustments(photo.adjustments, look.gradedStats, amount), targets);
        setNote(`Matched ${targets.length} photo${targets.length === 1 ? '' : 's'} to the finished reference.`);
      }
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-5">
      <section>
        <Heading
          title="Reference"
          hint={`Grade this photo the way you want the set to look, then sync it. Reference: ${photo.name}.`}
        />
        <Segmented<BatchSyncMode>
          label="Batch sync mode"
          value={mode}
          options={[
            ['smart', 'Smart'],
            ['copy', 'Exact copy'],
            ['palette', 'Palette'],
          ]}
          onChange={setMode}
        />
        <p className="mt-1.5 text-[11px] text-ink-400">
          {mode === 'smart'
            ? 'Copies every edit and evens out exposure and white balance per photo, for sets shot in changing light.'
            : mode === 'copy'
              ? 'The same settings on every photo, number for number.'
              : 'Keeps your retouch and matches each photo’s colours and tone curve to the finished reference.'}
        </p>
        {mode === 'palette' && (
          <div className="mt-2">
            <RangeRow label="Match strength" value={amount} min={0} max={100} onStart={() => undefined} onChange={setAmount} />
          </div>
        )}
        <button className="btn-primary mt-2 w-full" disabled={busy || targets.length === 0 || (mode === 'smart' && !heroStats)} onClick={syncInEditor}>
          {busy ? 'Syncing…' : `Sync to ${selected.length > 1 ? 'selected' : 'all'} (${targets.length})`}
        </button>
        {note && <p className="mt-1.5 text-[11px] text-mint-400">{note}</p>}
      </section>
      <LookLutSection photo={photo} targets={targets} />
      {isDesktop() ? <FolderQueue photo={photo} mode={mode} amount={amount} /> : (
        <p className="text-[11px] text-ink-400">Folder queues for hundreds of RAW files run in the FineRetouch desktop app. In the browser, sync here and use Export.</p>
      )}
    </div>
  );
}

function LookLutSection({ photo, targets }: { photo: Photo; targets: string[] }) {
  const aiOnline = useEditor((s) => s.aiOnline);
  const updateAdjustments = useEditor((s) => s.updateAdjustments);
  const [look, setLook] = useState<LookLut | null>(photo.adjustments.lut);
  // A look captured from this photo already is this photo's grade: apply it to the others only.
  const [captured, setCaptured] = useState(false);
  const applyTo = captured ? targets : [photo.id, ...targets];
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const capture = async () => {
    setBusy(true);
    setNote(null);
    try {
      const ref = await captureReferenceLook(photo);
      const fitted = await fitLook(ref.before, ref.after, `${photo.name.replace(/\.[^.]+$/, '')} look`);
      setLook({ name: fitted.name, size: fitted.size, data: fitted.data, amount: 100 });
      setCaptured(true);
      setNote('Look captured. Apply it to other photos or save it as a .cube.');
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const importCube = async (file: File) => {
    try {
      setLook(lookFromCube(await file.text()));
      setCaptured(false);
      setNote(`Loaded ${file.name}.`);
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    }
  };

  const save = () => {
    if (!look) return;
    const text = lutToCube(decodeLut(look), look.size, look.name);
    downloadBlob(new Blob([text], { type: 'text/plain' }), `${look.name.replace(/[^\w.-]+/g, '_') || 'look'}.cube`);
  };

  return (
    <section>
      <Heading
        title="Look LUT"
        hint="Capture the reference's colour grade as a 3D LUT, or load a .cube from other software. The LUT goes on top of each photo's own edits, so don't also copy the reference's colour sliders."
      />
      <input ref={fileRef} type="file" accept=".cube" hidden onChange={(e) => e.target.files?.[0] && void importCube(e.target.files[0])} />
      <div className="grid grid-cols-2 gap-2">
        <button className="btn" disabled={busy || !aiOnline} onClick={capture} title={aiOnline ? undefined : 'Needs the AI engine'}>
          {busy ? 'Capturing…' : 'Capture look'}
        </button>
        <button className="btn" onClick={() => fileRef.current?.click()}>
          Import .cube
        </button>
      </div>
      {look && (
        <div className="mt-2 space-y-2">
          <p className="truncate text-xs text-ink-100" title={look.name}>
            {look.name} <span className="text-ink-400">({look.size}³)</span>
          </p>
          <RangeRow label="LUT strength" value={look.amount} min={0} max={100} onStart={() => undefined} onChange={(v) => setLook({ ...look, amount: v })} />
          <div className="grid grid-cols-2 gap-2">
            <button className="btn" disabled={applyTo.length === 0} onClick={() => updateAdjustments({ lut: look }, applyTo)}>
              Apply to {applyTo.length}
            </button>
            <button className="btn" onClick={save}>
              Save .cube
            </button>
          </div>
          <button className="btn-ghost w-full" onClick={() => updateAdjustments({ lut: null }, applyTo)}>
            Remove LUT from these photos
          </button>
        </div>
      )}
      {note && <p className="mt-1.5 text-[11px] text-mint-400">{note}</p>}
    </section>
  );
}

const PROFILE_HINT: Record<ThermalProfile, string> = {
  quiet: 'One photo at a time with pauses: cool and quiet, good next to a tethered shoot.',
  balanced: 'Leaves a core free so the editor stays smooth.',
  performance: 'Every core, no pauses. Best overnight on a well-cooled desktop.',
};

function FolderQueue({ photo, mode, amount }: { photo: Photo; mode: BatchSyncMode; amount: number }) {
  const [sources, setSources] = useState<string[]>([]);
  const [outputDir, setOutputDir] = useState<string | null>(null);
  const [profile, setProfile] = useState<ThermalProfile>('balanced');
  const [format, setFormat] = useState<'jpeg' | 'png' | 'webp'>('jpeg');
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const r = await listBatchJobs();
      setJobs(r.jobs);
      setProfile(r.profile);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);
  const active = jobs.some((j) => j.status === 'queued' || j.status === 'running');
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => void refresh(), 1500);
    return () => clearInterval(t);
  }, [active, refresh]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const name = sources.length === 1 ? (sources[0].split(/[\\/]/).filter(Boolean).pop() ?? 'Batch') : `${sources.length} folders`;
      const out = outputDir ?? `${desktopInfo()?.exportsDir ?? ''}/${name.replace(/[^\w.-]+/g, '_')}-retouched`;
      const settings = { ...photo.adjustments };
      let referenceLookStats = null;
      if (mode === 'palette') referenceLookStats = (await captureReferenceLook(photo)).gradedStats;
      await createBatchJob({
        name,
        sources,
        recursive: true,
        outputDir: out,
        profile,
        settings: {
          adjustments: settings,
          sync: { mode, referenceStats: photo.analysis?.colorStats ?? photo.localStats, referenceLookStats, amount },
          output: { format, quality: 92 },
        },
      });
      setSources([]);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const act = async (id: string, action: 'pause' | 'resume' | 'cancel' | 'retry' | 'remove') => {
    try {
      if (action === 'remove') await removeBatchJob(id);
      else await batchAction(id, action);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <section>
      <Heading title="Folder queue" hint="Process whole folders of RAW or JPEG files with this photo's look, in the background and fully offline. Sub-folders are mirrored in the output." />
      <div className="space-y-2">
        <button className="btn w-full" onClick={async () => setSources([...new Set([...sources, ...(await pickSourceFolders())])])}>
          {sources.length ? `${sources.length} folder${sources.length > 1 ? 's' : ''} chosen · add more` : 'Choose photo folders…'}
        </button>
        {sources.map((s) => (
          <p key={s} className="truncate text-[11px] text-ink-300" title={s}>
            {s}
          </p>
        ))}
        <button className="btn-ghost w-full truncate" onClick={async () => setOutputDir((await pickExportFolder(outputDir, 'Choose where retouched photos go')) ?? outputDir)}>
          Output: {outputDir ?? 'exports folder (default)'}
        </button>
        <Segmented<'jpeg' | 'png' | 'webp'>
          label="Batch output format"
          value={format}
          options={[
            ['jpeg', 'JPEG'],
            ['png', 'PNG'],
            ['webp', 'WebP'],
          ]}
          onChange={setFormat}
        />
        <div>
          <Segmented<ThermalProfile>
            label="Thermal profile"
            value={profile}
            options={[
              ['quiet', 'Quiet'],
              ['balanced', 'Balanced'],
              ['performance', 'Fast'],
            ]}
            onChange={(v) => {
              setProfile(v);
              void setThermalProfile(v).catch((e) => setError(String(e)));
            }}
          />
          <p className="mt-1 text-[11px] text-ink-400">{PROFILE_HINT[profile]}</p>
        </div>
        <button className="btn-primary w-full" disabled={busy || sources.length === 0} onClick={start}>
          {busy ? 'Queueing…' : 'Start batch'}
        </button>
      </div>
      {error && <p className="mt-1.5 text-[11px] text-rose-400">{error}</p>}
      <ul className="mt-3 space-y-2">
        {jobs.map((j) => (
          <li key={j.id} className="rounded-md bg-ink-850 p-2 text-[11px]">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-ink-100" title={j.outputDir}>
                {j.name}
              </span>
              <span className="shrink-0 text-ink-400">
                {j.done + j.failed}/{j.total} · {j.status}
              </span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded bg-ink-700" role="progressbar" aria-valuenow={j.percent} aria-valuemin={0} aria-valuemax={100} aria-label={`${j.name} progress`}>
              <div className="h-full bg-apricot-400 transition-[width]" style={{ width: `${j.percent}%` }} />
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-1 text-ink-400">
              {j.status === 'running' && <span>{formatEta(j.etaSeconds)}</span>}
              {j.failed > 0 && <span className="text-rose-400">{j.failed} failed</span>}
              <span className="ml-auto flex gap-1">
                {(j.status === 'running' || j.status === 'queued') && <button className="btn-ghost px-1.5 py-0.5" onClick={() => act(j.id, 'pause')}>Pause</button>}
                {j.status === 'paused' && <button className="btn-ghost px-1.5 py-0.5" onClick={() => act(j.id, 'resume')}>Resume</button>}
                {j.status !== 'done' && j.status !== 'cancelled' && <button className="btn-ghost px-1.5 py-0.5" onClick={() => act(j.id, 'cancel')}>Cancel</button>}
                {(j.failed > 0 || j.status === 'cancelled') && <button className="btn-ghost px-1.5 py-0.5" onClick={() => act(j.id, 'retry')}>Retry</button>}
                {j.done > 0 && <button className="btn-ghost px-1.5 py-0.5" onClick={() => void revealInFolder(j.outputDir).catch(() => undefined)}>Show</button>}
                {(j.status === 'done' || j.status === 'cancelled') && <button className="btn-ghost px-1.5 py-0.5" onClick={() => act(j.id, 'remove')}>Remove</button>}
              </span>
            </div>
            {j.errors.slice(0, 2).map((e) => (
              <p key={e.source} className="mt-0.5 truncate text-rose-400" title={`${e.source}: ${e.error}`}>
                {e.source.split(/[\\/]/).pop()}: {e.error}
              </p>
            ))}
          </li>
        ))}
      </ul>
    </section>
  );
}
