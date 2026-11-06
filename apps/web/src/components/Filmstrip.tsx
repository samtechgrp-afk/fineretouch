'use client';

import { isDefault } from '@fineretouch/shared';
import { useEffect, useRef } from 'react';
import { describeExif } from '@/lib/rawParser';
import { useEditor, type Photo } from '@/lib/store';

export type CullFilter = 'all' | 'keep' | 'flagged';

export function isFlagged(p: Photo): boolean {
  const c = p.analysis?.cull;
  return Boolean(p.rejected || c?.blurry || c?.eyesClosed);
}

export function Filmstrip({ filter }: { filter: CullFilter }) {
  const order = useEditor((s) => s.order);
  const photos = useEditor((s) => s.photos);
  const activeId = useEditor((s) => s.activeId);
  const selected = useEditor((s) => s.selected);
  const toggleSelected = useEditor((s) => s.toggleSelected);
  const activeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [activeId]);

  const visible = order.filter((id) => {
    const p = photos[id];
    if (filter === 'keep') return !isFlagged(p);
    if (filter === 'flagged') return isFlagged(p);
    return true;
  });

  return (
    <div className="flex h-full items-center gap-2 overflow-x-auto px-3" role="listbox" aria-label="Photos" aria-multiselectable>
      {visible.map((id) => {
        const p = photos[id];
        const isActive = id === activeId;
        const isSel = selected.includes(id);
        return (
          <button
            key={id}
            ref={isActive ? activeRef : undefined}
            role="option"
            aria-selected={isActive || isSel}
            onClick={(e) => toggleSelected(id, e.metaKey || e.ctrlKey, e.shiftKey)}
            className={`relative h-[84px] shrink-0 overflow-hidden rounded-md border-2 transition ${
              isActive ? 'border-apricot-400' : isSel ? 'border-apricot-500/60' : 'border-transparent hover:border-ink-600'
            } ${p.rejected ? 'opacity-40' : ''}`}
            title={p.raw ? `${p.name}\n${describeExif(p.raw.exif)}` : p.name}
          >
            <img src={p.thumbUrl} alt={p.name} className="h-full w-auto max-w-[140px] object-cover" draggable={false} />
            <div className="absolute left-1 top-1 flex gap-0.5">
              {p.analysisState === 'pending' && <Badge tone="muted">AI…</Badge>}
              {p.analysis?.cull.blurry && <Badge tone="warn">Blurry</Badge>}
              {p.analysis?.cull.eyesClosed && <Badge tone="warn">Eyes closed</Badge>}
              {p.rejected && <Badge tone="warn">Rejected</Badge>}
            </div>
            {p.raw && (
              <span
                className="absolute bottom-1 left-1 rounded bg-black/75 px-1 py-px font-mono text-[9px] font-semibold tracking-wide text-apricot-300 ring-1 ring-apricot-400/40"
                aria-label={`${p.raw.label} camera RAW`}
              >
                {p.raw.label}
              </span>
            )}
            {!isDefault(p.adjustments) && (
              <span className="absolute bottom-1 right-1 h-2 w-2 rounded-full bg-apricot-400" title="Edited" />
            )}
          </button>
        );
      })}
      {!visible.length && <p className="px-2 text-xs text-ink-400">{order.length ? 'No photos match this filter.' : 'No photos yet.'}</p>}
    </div>
  );
}

function Badge({ children, tone }: { children: React.ReactNode; tone: 'warn' | 'muted' }) {
  return (
    <span className={`rounded px-1 py-px text-[9px] font-semibold uppercase ${tone === 'warn' ? 'bg-rose-400 text-ink-950' : 'bg-black/70 text-ink-300'}`}>
      {children}
    </span>
  );
}
