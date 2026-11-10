import { isKeepKind, type MarkStatus, type PerfectionMark, type PerfectionReport, type QaLabel } from '@fineretouch/shared';
import type { DisplayView } from '@/engine/renderer';

/** Overlay colours per mark status (the Final quality check uses the same ones). */
export const MARK_COLOURS: Record<MarkStatus, string> = {
  corrected: 'rgba(74,222,128,0.95)',
  remaining: 'rgba(248,113,113,0.95)',
  review: 'rgba(251,191,36,0.95)',
  protected: 'rgba(96,165,250,0.95)',
  ignored: 'rgba(161,161,170,0.8)',
};

interface Frame {
  display: DisplayView;
  /** Rendered image size in image pixels. */
  width: number;
  height: number;
}

/** CSS-pixel centre and radius of a mark in the viewport (single / split view). */
function place(m: PerfectionMark, f: Frame): { x: number; y: number; r: number } {
  const d = f.display;
  const long = Math.max(f.width, f.height);
  return {
    x: (m.x * f.width * d.scale + d.offsetX) / d.dpr,
    y: (m.y * f.height * d.scale + d.offsetY) / d.dpr,
    r: Math.max(5, (m.r * long * d.scale) / d.dpr * 1.6),
  };
}

/** Draw every mark of the report as a ring coloured by status; selected marks get a white halo. */
export function drawMarks(
  canvas: HTMLCanvasElement, wrap: HTMLElement, frame: Frame | null, report: PerfectionReport | null, selected: readonly number[],
  labels?: { labels: readonly QaLabel[]; selected: string | null } | null,
) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(wrap.clientWidth * dpr);
  const h = Math.round(wrap.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const g = canvas.getContext('2d');
  if (!g) return;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, w, h);
  if (!frame || frame.display.mode === 'side') return;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (labels) drawLabels(g, frame, labels.labels, labels.selected);
  if (!report) return;
  const sel = new Set(selected);
  for (const m of report.marks) {
    const p = place(m, frame);
    if (p.x < -p.r || p.y < -p.r || p.x > w / dpr + p.r || p.y > h / dpr + p.r) continue;
    if (sel.has(m.id)) {
      g.strokeStyle = 'rgba(255,255,255,0.95)';
      g.lineWidth = 3;
      g.beginPath();
      g.arc(p.x, p.y, p.r + 3, 0, Math.PI * 2);
      g.stroke();
    }
    g.strokeStyle = MARK_COLOURS[m.status];
    g.lineWidth = m.status === 'remaining' || m.status === 'review' ? 2 : 1.25;
    if (m.status === 'ignored' || m.status === 'protected') g.setLineDash([3, 2]);
    g.beginPath();
    g.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    g.stroke();
    g.setLineDash([]);
  }
}

/** The mark under a CSS-pixel point (smallest one that contains it), if any. */
export function hitMark(x: number, y: number, frame: Frame | null, report: PerfectionReport | null): PerfectionMark | null {
  if (!frame || !report || frame.display.mode === 'side') return null;
  let best: PerfectionMark | null = null;
  let bestR = Infinity;
  for (const m of report.marks) {
    const p = place(m, frame);
    if (Math.hypot(x - p.x, y - p.y) <= p.r + 2 && p.r < bestR) {
      best = m;
      bestR = p.r;
    }
  }
  return best;
}

/** QA labels: squares (red = remove, cyan = keep; dashed until confirmed, white when selected). */
function drawLabels(g: CanvasRenderingContext2D, frame: Frame, labels: readonly QaLabel[], selected: string | null) {
  for (const l of labels) {
    const p = place({ x: l.x, y: l.y, r: l.r } as PerfectionMark, frame);
    const s = p.r * 1.1;
    g.strokeStyle = l.id === selected ? 'rgba(255,255,255,1)' : isKeepKind(l.kind) ? 'rgba(34,211,238,0.95)' : 'rgba(244,63,94,0.95)';
    g.lineWidth = l.id === selected ? 2.5 : 1.5;
    g.setLineDash(l.confirmed ? [] : [4, 3]);
    g.strokeRect(p.x - s, p.y - s, 2 * s, 2 * s);
  }
  g.setLineDash([]);
}

/** The label under a CSS-pixel point, smallest first. */
export function hitLabel(x: number, y: number, frame: Frame | null, labels: readonly QaLabel[]): QaLabel | null {
  if (!frame || frame.display.mode === 'side') return null;
  let best: QaLabel | null = null;
  let bestR = Infinity;
  for (const l of labels) {
    const p = place({ x: l.x, y: l.y, r: l.r } as PerfectionMark, frame);
    const s = p.r * 1.1 + 2;
    if (Math.abs(x - p.x) <= s && Math.abs(y - p.y) <= s && p.r < bestR) {
      best = l;
      bestR = p.r;
    }
  }
  return best;
}
