'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { profileFromRgba } from '@/engine/style';
import { aiHealth, analyze, pool } from '@/lib/ai';
import { proxyJpeg } from '@/lib/depth';
import { bitmapPixels, clearCachedMask, fitSize, forgetImage, isImportable, loadImage, newId } from '@/lib/images';
import { IMPORT_ACCEPT } from '@/lib/rawParser';
import { fileFingerprint } from '@/lib/editPersistence';
import { selectActivePhoto, useEditor, type CompareMode } from '@/lib/store';
import { AdjustmentPanel } from './AdjustmentPanel';
import { ExportModal } from './ExportModal';
import { Filmstrip, isFlagged, type CullFilter } from './Filmstrip';
import { PhotoEditorViewport, requestZoom } from './PhotoEditorViewport';
import { PresetPanel } from './PresetPanel';

const PROFILE_LONG_EDGE = 512;

async function runAnalysis(ids: string[]) {
  const st = useEditor.getState();
  await pool(ids, 2, async (id) => {
    if (!useEditor.getState().photos[id]) return;
    if (useEditor.getState().aiOnline === false) {
      st.setAnalysis(id, 'offline');
      return;
    }
    st.setAnalysis(id, 'pending');
    try {
      const analysis = await analyze(await proxyJpeg(id));
      clearCachedMask(id);
      st.setAnalysis(id, 'done', analysis);
    } catch (e) {
      console.warn('analysis failed', e);
      st.setAnalysis(id, useEditor.getState().aiOnline ? 'error' : 'offline');
    }
  });
}

export function Editor() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [importing, setImporting] = useState<{ done: number; total: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  /** Files the last import could not open (e.g. a RAW without a usable preview). */
  const [importFailures, setImportFailures] = useState<string[]>([]);
  const [exportOpen, setExportOpen] = useState(false);
  const [filter, setFilter] = useState<CullFilter>('all');
  const aiOnline = useEditor((s) => s.aiOnline);
  const count = useEditor((s) => s.order.length);
  const flagged = useEditor((s) => s.order.filter((id) => isFlagged(s.photos[id])).length);
  const view = useEditor((s) => s.view);
  const photo = useEditor(selectActivePhoto);
  const { setView, undo, redo } = useEditor.getState();

  // AI service health: poll gently so the UI recovers when the service starts.
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      const h = await aiHealth();
      if (stop) return;
      const was = useEditor.getState().aiOnline;
      useEditor.getState().setAiOnline(Boolean(h));
      if (h && was === false) {
        const s = useEditor.getState();
        void runAnalysis(s.order.filter((id) => ['offline', 'error'].includes(s.photos[id].analysisState)));
      }
    };
    void tick();
    const t = setInterval(tick, 10_000);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, []);

  const importFiles = useCallback(async (list: FileList | File[]) => {
    const files = Array.from(list).filter(isImportable);
    if (!files.length) return;
    setImporting({ done: 0, total: files.length });
    const failures: string[] = [];
    const ids = files.map(() => newId());
    const loaded = new Set<string>();
    let done = 0;
    await pool(files, 3, async (file, i) => {
      const id = ids[i];
      try {
        const img = await loadImage(id, file);
        useEditor.getState().addPhotos([
          { id, name: file.name, width: img.width, height: img.height, thumbUrl: img.thumbUrl, raw: img.raw, fingerprint: fileFingerprint(file, img.width, img.height) },
        ]);
        const [pw, ph] = fitSize(img.proxy.width, img.proxy.height, PROFILE_LONG_EDGE);
        const profile = profileFromRgba(bitmapPixels(img.proxy, PROFILE_LONG_EDGE), pw, ph);
        useEditor.getState().setLocalStats(id, profile.stats);
        useEditor.getState().setProfile(id, profile);
        loaded.add(id);
      } catch (e) {
        console.warn(`Could not open ${file.name}`, e);
        failures.push(`${file.name}: ${(e as Error).message}`);
      } finally {
        setImporting({ done: ++done, total: files.length });
      }
    });
    setImporting(null);
    setImportFailures(failures);
    const ok = ids.filter((id) => loaded.has(id));
    useEditor.getState().sortPhotos(ok); // parallel decoding finishes out of order
    void runAnalysis(ok);
  }, []);

  // Keyboard shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest('input, textarea, select, [role="slider"], dialog')) return;
      const mod = e.metaKey || e.ctrlKey;
      const s = useEditor.getState();
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) s.redo();
        else s.undo();
      } else if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        s.redo();
      } else if (mod && e.shiftKey && e.key.toLowerCase() === 'c') {
        e.preventDefault();
        s.copySettings();
      } else if (mod && e.shiftKey && e.key.toLowerCase() === 'v') {
        e.preventDefault();
        s.pasteSettings();
      } else if (mod && e.key.toLowerCase() === 'a') {
        e.preventDefault();
        s.selectAll();
      } else if (e.key === 'ArrowRight') s.navigate(1);
      else if (e.key === 'ArrowLeft') s.navigate(-1);
      else if (e.key === '\\') s.setView({ mode: s.view.mode === 'original' ? 'edited' : 'original' });
      else if (e.key.toLowerCase() === 'y') s.setView({ mode: s.view.mode === 'split' ? 'edited' : 'split' });
      else if (e.key.toLowerCase() === 'm') s.setView({ showMask: !s.view.showMask });
      else if (e.key.toLowerCase() === 'd') s.setView({ mode: s.view.mode === 'diff' ? 'edited' : 'diff' });
      else if (e.key.toLowerCase() === 'x' && s.activeId) s.toggleRejected(s.activeId);
      else if (e.key === '0') requestZoom('fit');
      else if (e.key === '1') requestZoom('actual');
      else if (e.key === '+' || e.key === '=') requestZoom('in');
      else if (e.key === '-') requestZoom('out');
      else if ((e.key === 'Delete' || e.key === 'Backspace') && s.activeId) {
        const ids = s.selected.length ? s.selected : [s.activeId];
        ids.forEach(forgetImage);
        s.removePhotos(ids);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div
      className="grid h-dvh grid-rows-[48px_1fr] overflow-hidden"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => e.currentTarget === e.target && setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        void importFiles(e.dataTransfer.files);
      }}
    >
      <header className="flex items-center gap-3 border-b border-ink-800 bg-ink-900 px-4">
        <Logo />
        <span className="hidden text-xs text-ink-400 sm:inline">{count ? `${count} photo${count === 1 ? '' : 's'}` : 'No photos'}</span>
        <AiPill online={aiOnline} />
        {importFailures.length > 0 && (
          <button className="text-xs text-rose-400 underline-offset-2 hover:underline" title={importFailures.join('\n')} onClick={() => setImportFailures([])}>
            Couldn&apos;t open {importFailures.length} file{importFailures.length === 1 ? '' : 's'}
          </button>
        )}
        <div className="ml-auto flex items-center gap-2">
          <input ref={fileInput} type="file" accept={IMPORT_ACCEPT} multiple hidden onChange={(e) => e.target.files && importFiles(e.target.files)} />
          <button className="btn" onClick={() => fileInput.current?.click()} disabled={Boolean(importing)}>
            {importing ? `Importing ${importing.done}/${importing.total}` : 'Import'}
          </button>
          <button className="btn-primary" onClick={() => setExportOpen(true)} disabled={!count}>
            Export
          </button>
        </div>
      </header>

      <div className="grid min-h-0 grid-cols-1 md:grid-cols-[240px_1fr_300px]">
        <div className="hidden min-h-0 border-r border-ink-800 bg-ink-900 md:block">
          <PresetPanel />
        </div>

        <main className="grid min-h-0 grid-rows-[40px_1fr_112px]">
          <div className="flex items-center gap-1 overflow-x-auto border-b border-ink-800 bg-ink-900 px-2">
            <div className="flex gap-0.5 rounded-md bg-ink-850 p-0.5" role="group" aria-label="Compare">
              {(
                [
                  ['edited', 'Edited'],
                  ['original', 'Original'],
                  ['split', 'Split'],
                  ['side', 'Side by side'],
                  ['diff', 'Difference'],
                ] as Array<[CompareMode, string]>
              ).map(([m, label]) => (
                <button key={m} className="seg" aria-pressed={view.mode === m} onClick={() => setView({ mode: m })}>
                  {label}
                </button>
              ))}
            </div>
            <span className="mx-1 h-5 w-px bg-ink-700" />
            <button className="btn-ghost" onClick={() => requestZoom('out')} aria-label="Zoom out">
              −
            </button>
            <button className="btn-ghost" onClick={() => requestZoom('fit')}>
              Fit
            </button>
            <button className="btn-ghost" onClick={() => requestZoom('actual')}>
              100%
            </button>
            <button className="btn-ghost" onClick={() => requestZoom('in')} aria-label="Zoom in">
              +
            </button>
            <span className="mx-1 h-5 w-px bg-ink-700" />
            <button className="btn-ghost" onClick={() => undo()} disabled={!photo?.past.length} title="Undo (Ctrl/Cmd+Z)">
              Undo
            </button>
            <button className="btn-ghost" onClick={() => redo()} disabled={!photo?.future.length} title="Redo (Ctrl/Cmd+Shift+Z)">
              Redo
            </button>
            <label className="ml-auto flex shrink-0 items-center gap-1.5 px-2 text-xs text-ink-300">
              <input type="checkbox" checked={view.showMask} onChange={(e) => setView({ showMask: e.target.checked })} className="accent-apricot-400" />
              Show AI masks
            </label>
          </div>
          <div className="relative min-h-0">
            <PhotoEditorViewport />
            {dragging && (
              <div className="pointer-events-none absolute inset-3 grid place-items-center rounded-xl border-2 border-dashed border-apricot-400 bg-black/50 text-sm">
                Drop to import
              </div>
            )}
          </div>
          <div className="grid min-h-0 grid-rows-[28px_1fr] border-t border-ink-800 bg-ink-900">
            <div className="flex items-center gap-1 px-3 text-[11px] text-ink-400">
              <span className="mr-1">Culling:</span>
              {(
                [
                  ['all', `All ${count}`],
                  ['keep', `Keepers ${count - flagged}`],
                  ['flagged', `Flagged ${flagged}`],
                ] as Array<[CullFilter, string]>
              ).map(([f, label]) => (
                <button key={f} className="seg py-0.5" aria-pressed={filter === f} onClick={() => setFilter(f)}>
                  {label}
                </button>
              ))}
              <span className="ml-auto hidden lg:inline">Ctrl/Cmd-click to multi-select · X reject · Y split · \ before</span>
            </div>
            <Filmstrip filter={filter} />
          </div>
        </main>

        <div className="hidden min-h-0 border-l border-ink-800 bg-ink-900 md:block">
          <AdjustmentPanel />
        </div>
      </div>

      {exportOpen && <ExportModal onClose={() => setExportOpen(false)} />}
    </div>
  );
}

function Logo() {
  return (
    <div className="flex items-center gap-2">
      <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
        <defs>
          <linearGradient id="fr-g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#ffcf8a" />
            <stop offset="1" stopColor="#ec9a32" />
          </linearGradient>
        </defs>
        <rect x="2" y="2" width="20" height="20" rx="6" fill="url(#fr-g)" />
        <path d="M8 16.5V7.5h6.2M8 12h4.6" stroke="#0d0e10" strokeWidth="2.2" strokeLinecap="round" fill="none" />
        <circle cx="16.5" cy="15.5" r="1.8" fill="#0d0e10" />
      </svg>
      <span className="text-sm font-semibold tracking-tight">FineRetouch</span>
    </div>
  );
}

function AiPill({ online }: { online: boolean | null }) {
  const label = online === null ? 'Checking AI…' : online ? 'AI ready' : 'AI offline';
  const dot = online === null ? 'bg-ink-400' : online ? 'bg-mint-400' : 'bg-rose-400';
  return (
    <span
      className="flex items-center gap-1.5 rounded-full border border-ink-700 px-2 py-0.5 text-[11px] text-ink-300"
      title={online ? 'Face detection and server rendering available' : 'Start services/ai to enable face-aware tools; colour-based skin retouch still works'}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  );
}
