'use client';

import { BUILT_IN_PRESETS, type Preset } from '@fineretouch/shared';
import { useState } from 'react';
import { useEditor } from '@/lib/store';

export function PresetPanel() {
  const custom = useEditor((s) => s.customPresets);
  const selected = useEditor((s) => s.selected);
  const total = useEditor((s) => s.order.length);
  const activeId = useEditor((s) => s.activeId);
  const hasActive = Boolean(activeId);
  const syncCount = selected.length > 1 ? selected.filter((id) => id !== activeId).length : Math.max(0, total - 1);
  const clipboard = useEditor((s) => s.clipboard);
  const { applyPresetTo, saveCustomPreset, deleteCustomPreset, copySettings, pasteSettings, syncToSelected, resetPhotos } =
    useEditor.getState();
  const [name, setName] = useState('');

  const scope = selected.length > 1 ? `${selected.length} selected` : 'current photo';

  return (
    <aside className="flex h-full min-w-0 flex-col overflow-y-auto overflow-x-hidden" aria-label="Presets and sync">
      <Block title="Batch sync">
        <p className="mb-2 text-[11px] text-ink-400">
          Edit one photo, then push the same settings to the rest of the shoot.
        </p>
        <div className="grid grid-cols-2 gap-1.5">
          <button className="btn" onClick={copySettings} disabled={!hasActive} title="Ctrl/Cmd+Shift+C">
            Copy
          </button>
          <button className="btn" onClick={() => pasteSettings()} disabled={!clipboard || !hasActive} title="Ctrl/Cmd+Shift+V">
            Paste
          </button>
          <button className="btn-primary col-span-2" onClick={syncToSelected} disabled={!hasActive || syncCount === 0}>
            Sync to {selected.length > 1 ? `${syncCount} selected` : `all ${syncCount} others`}
          </button>
          <button className="btn-ghost col-span-2" onClick={() => resetPhotos()} disabled={!hasActive}>
            Reset {scope}
          </button>
        </div>
      </Block>

      <Block title="Presets">
        <p className="mb-2 text-[11px] text-ink-400">Applies to {scope}.</p>
        <ul className="space-y-1">
          {[...BUILT_IN_PRESETS, ...custom].map((p) => (
            <PresetRow key={p.id} preset={p} onApply={() => applyPresetTo(p)} onDelete={p.builtIn ? undefined : () => deleteCustomPreset(p.id)} disabled={!hasActive} />
          ))}
        </ul>
        <form
          className="mt-3 flex gap-1.5"
          onSubmit={(e) => {
            e.preventDefault();
            if (saveCustomPreset(name)) setName('');
          }}
        >
          <input className="field" placeholder="Save current as…" value={name} onChange={(e) => setName(e.target.value)} maxLength={60} aria-label="Preset name" />
          <button className="btn" type="submit" disabled={!name.trim() || !hasActive}>
            Save
          </button>
        </form>
      </Block>
    </aside>
  );
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-ink-800 px-4 py-3">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-300">{title}</h2>
      {children}
    </section>
  );
}

function PresetRow({ preset, onApply, onDelete, disabled }: { preset: Preset; onApply(): void; onDelete?: () => void; disabled: boolean }) {
  return (
    <li className="group flex items-center gap-1">
      <button
        className="min-w-0 flex-1 rounded-md px-2 py-1.5 text-left hover:bg-ink-800 disabled:opacity-40"
        onClick={onApply}
        disabled={disabled}
        title={preset.description}
      >
        <span className="block text-xs text-ink-100">{preset.name}</span>
        <span className="block truncate text-[11px] text-ink-400">{preset.description}</span>
      </button>
      {onDelete && (
        <button className="btn-ghost opacity-0 group-hover:opacity-100" onClick={onDelete} aria-label={`Delete ${preset.name}`}>
          ✕
        </button>
      )}
    </li>
  );
}
