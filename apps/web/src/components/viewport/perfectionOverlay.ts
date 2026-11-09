import type { MarkStatus, PerfectionMark, PerfectionReport } from '@fineretouch/shared';
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
export function drawMarks(canvas: HTMLCanvasElement, wrap: HTMLElement, frame: Frame | null, report: PerfectionReport | null, selected: readonly number[]) {
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
  if (!frame || !report || frame.display.mode === 'side') return;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
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
