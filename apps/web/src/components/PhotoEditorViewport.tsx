'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { LayerKind, LayerStroke, RetouchLayer } from '@fineretouch/shared';
import { Renderer, type DisplayView } from '@/engine/renderer';
import { ensureDepth, hairRoiFor } from '@/lib/depth';
import { ensureParse, ensureSegmentation, lensRoiFor } from '@/lib/studio';
import { editMaskFor, faceMaskFor } from '@/lib/export';
import { ensureProxy, fitSize, getDepth, getProxy, getSegmentation } from '@/lib/images';
import { LAYER_LONG_EDGE, healedPixels, pixelsOf, syncLayerMasks } from '@/lib/localRetouch';
import { perfectedPixels, perfectionKeyOf, schedulePerfection, usePerfection } from '@/lib/perfection';
import { drawMarks, hitMark } from './viewport/perfectionOverlay';
import { autoHealSource } from '@/engine/heal';
import { noiseLevelOf } from '@/lib/noise';
import { displayAdjustments, selectActivePhoto, useEditor } from '@/lib/store';

const BG: [number, number, number] = [0.051, 0.055, 0.063];
const MAX_ZOOM_PERCENT = 800;

interface Geometry {
  dpr: number;
  cw: number; // device px
  ch: number;
  paneW: number;
  fit: number; // device px per proxy px when zoom = 1
}

/**
 * WebGL2 viewport. The proxy (<= 2560px) is uploaded once per photo; slider
 * moves only re-run the retouch pass, so feedback stays within a frame.
 */
export function PhotoEditorViewport() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const loadedIdRef = useRef<string | null>(null);
  const maskKeyRef = useRef<string>('');
  const hairKeyRef = useRef<string>('');
  const depthKeyRef = useRef<string>('');
  const lensKeyRef = useRef<string>('');
  const segKeyRef = useRef<string>('');
  const editKeyRef = useRef<unknown>(null);
  const healRef = useRef<Uint8ClampedArray | null>(null);
  const displayRef = useRef<DisplayView | null>(null);
  const brushCanvasRef = useRef<HTMLCanvasElement>(null);
  const marksCanvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const [renderMs, setRenderMs] = useState<number | null>(null);
  // Render stages the last frame ran (0 = served from the stage cache, e.g. while panning).
  const [stagesRun, setStagesRun] = useState<number | null>(null);

  const photo = useEditor(selectActivePhoto);
  const view = useEditor((s) => s.view);
  const pending = useEditor((s) => s.assistant.pending);
  const setView = useEditor((s) => s.setView);

  // Create / destroy the GL context.
  useEffect(() => {
    try {
      rendererRef.current = new Renderer(canvasRef.current!);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    return () => {
      rendererRef.current?.dispose();
      rendererRef.current = null;
      loadedIdRef.current = null;
    };
  }, []);

  const geometry = useCallback((): Geometry | null => {
    const r = rendererRef.current;
    const canvas = canvasRef.current;
    if (!r || !canvas || !r.width) return null;
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.width;
    const ch = canvas.height;
    const paneW = view.mode === 'side' ? cw / 2 : cw;
    const fit = Math.min(paneW / r.width, ch / r.height) * 0.96;
    return { dpr, cw, ch, paneW, fit };
  }, [view.mode]);

  /** Skin Perfection marks over the photo (Final quality check overlay). */
  const perfFrame = () => {
    const d = displayRef.current;
    const r = rendererRef.current;
    return d && r && r.width ? { display: d, width: r.width, height: r.height } : null;
  };
  const perfReport = () => {
    const st = useEditor.getState();
    const p = selectActivePhoto(st);
    const ps = usePerfection.getState();
    if (!p || !p.adjustments.skinPerfection.enabled || !ps.overlay) return null;
    return ps.byPhoto[p.id]?.report ?? null;
  };
  const drawPerfectionMarks = () => {
    const c = marksCanvasRef.current;
    const wrap = wrapRef.current;
    if (c && wrap) drawMarks(c, wrap, perfFrame(), perfReport(), usePerfection.getState().selected);
  };

  const draw = useCallback(() => {
    cancelAnimationFrame(frameRef.current);
    frameRef.current = requestAnimationFrame(() => {
      const r = rendererRef.current;
      const st = useEditor.getState();
      const p = selectActivePhoto(st);
      if (!r || !p) return;
      // The pending assistant plan is previewed here only (never saved or exported until approved).
      const adj = displayAdjustments(st, p);
      const proxy = getProxy(p.id);
      if (!proxy) {
        // Evicted from the preview cache: decode it again, then draw.
        void ensureProxy(p.id).then((b) => b && draw());
        return;
      }
      const t0 = performance.now();
      if (loadedIdRef.current !== p.id) {
        r.setSource(proxy, proxy.width, proxy.height);
        loadedIdRef.current = p.id;
        maskKeyRef.current = hairKeyRef.current = depthKeyRef.current = lensKeyRef.current = segKeyRef.current = '';
        editKeyRef.current = null;
        healRef.current = null;
      }
      // Pro tools: heal / clone strokes replayed on the proxy pixels (stage 0b of the engine).
      const lr = adj.localRetouch;
      // Skin Perfection (stage 0c, engine only): its perfected preview is the base the heal
      // strokes and every shader stage work on. While a re-run is pending the last result shows.
      const perf = adj.skinPerfection.enabled ? perfectedPixels(p.id, perfectionKeyOf(adj), proxy.width, proxy.height) : null;
      const base = perf ? perf.pixels : null;
      const healed = lr.heal.length ? healedPixels(`${p.id}:proxy`, base ?? pixelsOf(`${p.id}:proxy`, proxy), proxy.width, proxy.height, lr.heal) : base;
      if (healRef.current !== healed) {
        r.setHealedSource(healed);
        healRef.current = healed;
      }
      const sm = adj.skinMask;
      const faceKey = `${p.id}:${p.analysisState}:${p.analysis?.faces.length ?? 0}`;
      const maskKey = `${faceKey}:${p.parseState}:${sm.useSegmentation}:${sm.includeBodySkin}`;
      const mask = faceMaskFor(p.id, p.analysis, proxy.width, proxy.height, sm);
      if (maskKeyRef.current !== maskKey) {
        r.setMask(mask?.data ?? null, mask?.width, mask?.height);
        maskKeyRef.current = maskKey;
      }
      const edit = editMaskFor(p.id, sm, proxy.width, proxy.height);
      if (editKeyRef.current !== (edit?.data ?? null)) {
        r.setEdit(edit?.data ?? null, edit?.width, edit?.height);
        editKeyRef.current = edit?.data ?? null;
      }
      syncLayerMasks(r, p.id, lr, fitSize(proxy.width, proxy.height, LAYER_LONG_EDGE), mask, edit, `${maskKey}:${sm.strokes.length}:${sm.feather}`);
      const hairOn = adj.strayHairRemoval.enabled;
      const hairKey = `${faceKey}:${hairOn}`;
      if (hairKeyRef.current !== hairKey) {
        const roi = hairOn ? hairRoiFor(p.id, p.analysis, proxy.width, proxy.height) : null;
        r.setHairRoi(roi?.data ?? null, roi?.width, roi?.height);
        hairKeyRef.current = hairKey;
      }
      const depthKey = `${p.id}:${p.depthState}`;
      if (depthKeyRef.current !== depthKey) {
        const d = getDepth(p.id);
        r.setDepth(d?.source ?? null, d?.width, d?.height);
        depthKeyRef.current = depthKey;
      }
      const glareOn = adj.glareReduction > 0 || st.view.overlay === 'glare';
      const lensKey = `${faceKey}:${glareOn}`;
      if (lensKeyRef.current !== lensKey) {
        const roi = glareOn ? lensRoiFor(p.id, p.analysis, proxy.width, proxy.height) : null;
        r.setLensRoi(roi?.data ?? null, roi?.width, roi?.height);
        lensKeyRef.current = lensKey;
      }
      const segKey = `${p.id}:${p.segmentState}`;
      if (segKeyRef.current !== segKey) {
        const seg = getSegmentation(p.id);
        r.setSegmentation(seg ?? null, seg?.width, seg?.height);
        segKeyRef.current = segKey;
      }
      const noiseLevel = adj.skinNoise > 0 ? noiseLevelOf(`${p.id}:proxy`, proxy) : undefined;
      r.render(adj, { hasFace: Boolean(mask), sourceStats: p.analysis?.colorStats ?? p.localStats, noiseLevel });
      const g = geometry();
      if (!g) return;
      const v = st.view;
      const scale = g.fit * v.zoom;
      const display: DisplayView = {
        mode: v.mode,
        scale,
        offsetX: (g.paneW - r.width * scale) / 2 + v.panX * g.dpr,
        offsetY: (g.ch - r.height * scale) / 2 + v.panY * g.dpr,
        split: v.split,
        showMask: v.showMask,
        overlay: v.overlay,
        layerMask: v.proTool === 'layer' && v.activeLayerId ? layerView(lr.layers, v.activeLayerId) : null,
        dpr: g.dpr,
        background: BG,
      };
      r.display(display, adj.glareReduction / 100);
      displayRef.current = display;
      drawPerfectionMarks();
      setRenderMs(performance.now() - t0);
      setStagesRun(r.lastStages.length);
    });
  }, [geometry]);

  // Redraw whenever the photo, its edits/analysis or the view change.
  useEffect(() => {
    draw();
  }, [photo?.id, photo?.adjustments, photo?.analysis, photo?.analysisState, photo?.localStats, photo?.depthState, photo?.segmentState, photo?.parseState, pending, view, draw]);

  // Skin Perfection: re-run the engine (debounced) when its settings change; redraw on results.
  const perfOn = Boolean(photo?.adjustments.skinPerfection.enabled);
  const perfKey = photo && perfOn ? perfectionKeyOf(photo.adjustments) : '';
  const perfAuto = usePerfection((s) => s.auto);
  const perfOnline = useEditor((s) => s.aiOnline);
  useEffect(() => {
    if (photo && perfOn && perfAuto && perfOnline) schedulePerfection(photo.id);
  }, [photo?.id, perfKey, perfOn, perfAuto, perfOnline]); // eslint-disable-line react-hooks/exhaustive-deps
  const perfState = usePerfection((s) => (photo ? s.byPhoto[photo.id] : undefined));
  const perfSelected = usePerfection((s) => s.selected);
  const perfOverlay = usePerfection((s) => s.overlay);
  useEffect(() => {
    draw();
  }, [perfState?.key, perfState?.report, draw]);
  useEffect(() => {
    drawPerfectionMarks();
  }, [perfSelected, perfOverlay]); // eslint-disable-line react-hooks/exhaustive-deps

  // Edit as shown (with a pending assistant plan previewed), so previewed tools get their maps too.
  const shown = useEditor((s) => (photo ? displayAdjustments(s, photo) : null));
  // AI face parsing for the semantic skin mask, once the photo has a face and the engine is up.
  const parseWanted = Boolean(photo?.adjustments.skinMask.useSegmentation && photo?.analysis?.faces.length);
  useEffect(() => {
    if (photo && parseWanted && photo.parseState === 'none') void ensureParse(photo.id);
  }, [photo?.id, parseWanted, photo?.parseState]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch (or upgrade) the depth map once bokeh is switched on for this photo.
  const bokehOn = Boolean(shown?.depthBokeh.enabled);
  const aiOnline = useEditor((s) => s.aiOnline);
  useEffect(() => {
    if (photo && bokehOn && photo.depthState !== 'pending' && photo.depthState !== 'model') void ensureDepth(photo.id);
  }, [photo?.id, bokehOn, photo?.depthState, photo?.analysisState, aiOnline]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch the garment mask once a clothing tool (or its overlay) is switched on.
  const clothingOn = Boolean(shown?.clothing.enabled) || view.overlay === 'clothing';
  useEffect(() => {
    if (photo && clothingOn && photo.segmentState === 'none') void ensureSegmentation(photo.id);
  }, [photo?.id, clothingOn, photo?.segmentState]); // eslint-disable-line react-hooks/exhaustive-deps
  // The engine came back: try again for a photo that had no segmentation.
  useEffect(() => {
    const p = selectActivePhoto(useEditor.getState());
    if (aiOnline && p?.segmentState === 'unavailable') useEditor.getState().setSegmentState(p.id, 'none');
    if (aiOnline && p?.parseState === 'unavailable') useEditor.getState().setParseState(p.id, 'none');
  }, [aiOnline]);

  // Keep the drawing buffer at device resolution.
  useEffect(() => {
    const wrap = wrapRef.current!;
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      const canvas = canvasRef.current!;
      canvas.width = Math.max(1, Math.round(wrap.clientWidth * dpr));
      canvas.height = Math.max(1, Math.round(wrap.clientHeight * dpr));
      draw();
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [draw]);

  const zoomPercent = useCallback(
    (zoom: number) => {
      const g = geometry();
      const r = rendererRef.current;
      if (!g || !r || !photo) return 100;
      // 100% = one original pixel per CSS pixel.
      const devicePerOriginal = (g.fit * zoom * r.width) / photo.width;
      return (devicePerOriginal / g.dpr) * 100;
    },
    [geometry, photo],
  );

  const maxZoom = useCallback(() => {
    const pct = zoomPercent(1);
    return Math.max(1, MAX_ZOOM_PERCENT / pct);
  }, [zoomPercent]);

  /** Zoom keeping the image point under (cx, cy) CSS px fixed. */
  const zoomAt = useCallback(
    (nextZoom: number, cx?: number, cy?: number) => {
      const wrap = wrapRef.current;
      if (!wrap) return;
      const v = useEditor.getState().view;
      const z = Math.min(maxZoom(), Math.max(0.25, nextZoom));
      const paneW = v.mode === 'side' ? wrap.clientWidth / 2 : wrap.clientWidth;
      const px = (cx ?? paneW / 2) % paneW;
      const py = cy ?? wrap.clientHeight / 2;
      // Point relative to the pane centre, before and after zoom.
      const k = z / v.zoom;
      const dx = px - paneW / 2;
      const dy = py - wrap.clientHeight / 2;
      setView({ zoom: z, panX: dx - (dx - v.panX) * k, panY: dy - (dy - v.panY) * k });
    },
    [maxZoom, setView],
  );

  useEffect(() => {
    const wrap = wrapRef.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = wrap.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
      zoomAt(useEditor.getState().view.zoom * factor, e.clientX - rect.left, e.clientY - rect.top);
    };
    wrap.addEventListener('wheel', onWheel, { passive: false });
    return () => wrap.removeEventListener('wheel', onWheel);
  }, [zoomAt]);

  // Expose zoom helpers to the toolbar via a custom event (keeps the store free of DOM).
  useEffect(() => {
    const onZoom = (e: Event) => {
      const d = (e as CustomEvent<{ action: 'fit' | 'in' | 'out' | 'actual' }>).detail;
      const v = useEditor.getState().view;
      if (d.action === 'fit') setView({ zoom: 1, panX: 0, panY: 0 });
      else if (d.action === 'in') zoomAt(v.zoom * 1.5);
      else if (d.action === 'out') zoomAt(v.zoom / 1.5);
      else if (d.action === 'actual') zoomAt(100 / zoomPercent(1));
    };
    window.addEventListener('fineretouch:zoom', onZoom);
    return () => window.removeEventListener('fineretouch:zoom', onZoom);
  }, [setView, zoomAt, zoomPercent]);

  const drag = useRef<{ kind: 'pan' | 'split'; x: number; y: number; panX: number; panY: number } | null>(null);
  /** Stroke being painted: normalised image points (committed as one undo step on release). */
  const stroke = useRef<{ points: number[]; offset: [number, number] | null } | null>(null);
  const activeLayer = photo && view.activeLayerId ? photo.adjustments.localRetouch.layers.find((l) => l.id === view.activeLayerId) : undefined;
  /** Which brush the pointer paints with, if any. */
  const brush: BrushKind | null = view.maskBrush
    ? view.maskBrush
    : view.proTool === 'heal' || view.proTool === 'clone'
      ? view.proTool
      : view.proTool === 'layer' && activeLayer
        ? layerStrokeMode(activeLayer.kind, view.layerBrush)
        : null;
  /** Source offset for the next heal / clone stroke starting at pt (aligned once set). */
  const sourceOffsetFor = (pt: [number, number]): [number, number] | null => {
    if (view.cloneOffset) return view.cloneOffset;
    if (view.cloneSource) return [view.cloneSource[0] - pt[0], view.cloneSource[1] - pt[1]];
    return null;
  };

  /** CSS px in the viewport -> normalised image coordinates (null outside the photo). */
  const toImage = (cssX: number, cssY: number): [number, number] | null => {
    const d = displayRef.current;
    const r = rendererRef.current;
    if (!d || !r || !r.width) return null;
    const canvas = canvasRef.current!;
    let px = cssX * d.dpr;
    if (d.mode === 'side') px %= canvas.width / 2;
    const u = (px - d.offsetX) / d.scale / r.width;
    const v = (cssY * d.dpr - d.offsetY) / d.scale / r.height;
    return [u, v];
  };
  /** Brush radius in CSS px at the current zoom. */
  const brushCssRadius = () => {
    const d = displayRef.current;
    const r = rendererRef.current;
    if (!d || !r) return 0;
    return (view.brushSize * Math.max(r.width, r.height) * d.scale) / d.dpr;
  };
  const drawBrush = (x: number, y: number, painting: boolean) => {
    const c = brushCanvasRef.current;
    const wrap = wrapRef.current;
    if (!c || !wrap) return;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(wrap.clientWidth * dpr) || c.height !== Math.round(wrap.clientHeight * dpr)) {
      c.width = Math.round(wrap.clientWidth * dpr);
      c.height = Math.round(wrap.clientHeight * dpr);
    }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!painting) g.clearRect(0, 0, c.width, c.height);
    const rad = brushCssRadius();
    const colour = BRUSH_COLOURS[brush ?? 'add'];
    // Heal / clone: where the source is sampled from (follows the brush once aligned).
    const d = displayRef.current;
    const r = rendererRef.current;
    if ((brush === 'heal' || brush === 'clone') && d && r && !painting) {
      const pt = toImage(x, y);
      const off = pt ? (stroke.current?.offset ?? sourceOffsetFor(pt)) : null;
      g.strokeStyle = 'rgba(120,255,170,0.9)';
      g.setLineDash([4, 3]);
      g.beginPath();
      if (off) g.arc(x + (off[0] * r.width * d.scale) / d.dpr, y + (off[1] * r.height * d.scale) / d.dpr, rad, 0, Math.PI * 2);
      else if (view.cloneSource) {
        const sx = (view.cloneSource[0] * r.width * d.scale + d.offsetX) / d.dpr;
        const sy = (view.cloneSource[1] * r.height * d.scale + d.offsetY) / d.dpr;
        g.arc(sx, sy, rad, 0, Math.PI * 2);
      }
      g.stroke();
      g.setLineDash([]);
    }
    if (painting) {
      g.fillStyle = colour;
      g.beginPath();
      g.arc(x, y, rad, 0, Math.PI * 2);
      g.fill();
    } else {
      g.strokeStyle = 'rgba(255,255,255,0.9)';
      g.lineWidth = 1;
      g.beginPath();
      g.arc(x, y, rad, 0, Math.PI * 2);
      g.stroke();
    }
  };
  const clearBrush = () => {
    const c = brushCanvasRef.current;
    c?.getContext('2d')?.clearRect(0, 0, c.width, c.height);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const rect = wrapRef.current!.getBoundingClientRect();
    if (brush && photo && e.button === 0) {
      const pt = toImage(e.clientX - rect.left, e.clientY - rect.top);
      if (!pt) return;
      if ((brush === 'heal' || brush === 'clone') && e.altKey) {
        // Alt-click sets the source; the first stroke after it fixes the (aligned) offset.
        setView({ cloneSource: pt, cloneOffset: null });
        return;
      }
      let offset: [number, number] | null = null;
      if (brush === 'heal' || brush === 'clone') {
        offset = sourceOffsetFor(pt);
        if (!offset && brush === 'clone') return; // clone needs a source (Alt-click)
        if (offset && !view.cloneOffset) setView({ cloneOffset: offset });
      }
      stroke.current = { points: [pt[0], pt[1]], offset };
      (e.target as Element).setPointerCapture(e.pointerId);
      clearBrush();
      drawBrush(e.clientX - rect.left, e.clientY - rect.top, true);
      return;
    }
    if (!brush && e.button === 0) {
      // Click a Skin Perfection mark to select it (Shift adds / removes); drag still pans.
      const m = hitMark(e.clientX - rect.left, e.clientY - rect.top, perfFrame(), perfReport());
      if (m) {
        const ps = usePerfection.getState();
        const cur = ps.selected;
        ps.select(e.shiftKey ? (cur.includes(m.id) ? cur.filter((i) => i !== m.id) : [...cur, m.id]) : [m.id]);
        return;
      }
    }
    const v = useEditor.getState().view;
    const x = e.clientX - rect.left;
    const nearSplit = v.mode === 'split' && Math.abs(x - v.split * rect.width) < 24;
    drag.current = { kind: nearSplit || (v.mode === 'split' && v.zoom <= 1) ? 'split' : 'pan', x: e.clientX, y: e.clientY, panX: v.panX, panY: v.panY };
    (e.target as Element).setPointerCapture(e.pointerId);
    if (drag.current.kind === 'split') setView({ split: Math.min(1, Math.max(0, x / rect.width)) });
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const rect = wrapRef.current!.getBoundingClientRect();
    if (brush) {
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const st = stroke.current;
      if (st) {
        const pt = toImage(x, y);
        const n = st.points.length;
        // Skip points closer than a fifth of the brush radius (keeps strokes small).
        if (pt && Math.hypot(pt[0] - st.points[n - 2], pt[1] - st.points[n - 1]) > view.brushSize * 0.2 && n < 7998) {
          st.points.push(pt[0], pt[1]);
          drawBrush(x, y, true);
        }
      } else drawBrush(x, y, false);
      return;
    }
    const d = drag.current;
    if (!d) return;
    if (d.kind === 'split') setView({ split: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)) });
    else setView({ panX: d.panX + e.clientX - d.x, panY: d.panY + e.clientY - d.y });
  };
  const onPointerUp = () => {
    drag.current = null;
    const st = stroke.current;
    stroke.current = null;
    if (!st || !brush || !photo) return;
    const points = st.points.map((v) => Math.round(v * 1e4) / 1e4);
    const store = useEditor.getState();
    if (brush === 'heal' || brush === 'clone') {
      let offset = st.offset;
      if (!offset) {
        // Spot healing: no source set, so pick one around the stroke on the current pixels.
        const proxy = getProxy(photo.id);
        if (proxy) {
          const adj0 = photo.adjustments;
          const perf = adj0.skinPerfection.enabled ? perfectedPixels(photo.id, perfectionKeyOf(adj0), proxy.width, proxy.height) : null;
          const base = perf?.pixels ?? pixelsOf(`${photo.id}:proxy`, proxy);
          const cur = healedPixels(`${photo.id}:proxy`, base, proxy.width, proxy.height, photo.adjustments.localRetouch.heal) ?? base;
          offset = autoHealSource(cur, proxy.width, proxy.height, view.brushSize, points);
        }
      }
      if (offset) {
        const o: [number, number] = [Math.round(offset[0] * 1e5) / 1e5, Math.round(offset[1] * 1e5) / 1e5];
        store.addHealStroke({ mode: brush, size: view.brushSize, hardness: view.brushHardness, opacity: view.brushOpacity, offset: o, points });
      }
    } else if (view.maskBrush) {
      store.addMaskStroke({ mode: view.maskBrush, size: view.brushSize, points });
    } else if (activeLayer) {
      store.addLayerStroke(activeLayer.id, { mode: brush, size: view.brushSize, flow: view.brushFlow, points });
    }
    clearBrush();
  };

  const pct = photo ? Math.round(zoomPercent(view.zoom)) : null;

  return (
    <div
      ref={wrapRef}
      className="relative h-full w-full touch-none select-none overflow-hidden bg-ink-950"
      style={{ cursor: brush ? 'crosshair' : view.mode === 'split' ? 'ew-resize' : view.zoom > 1 ? 'grab' : 'default' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onPointerLeave={() => !stroke.current && clearBrush()}
      onDoubleClick={() => !brush && (view.zoom === 1 ? zoomAt(100 / zoomPercent(1)) : setView({ zoom: 1, panX: 0, panY: 0 }))}
      data-testid="viewport"
    >
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-label="Photo preview" />
      <canvas ref={marksCanvasRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden data-testid="perfection-marks" />
      <canvas ref={brushCanvasRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden data-testid="brush-layer" />
      {!photo && !error && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center text-sm text-ink-400">
          Drop photos anywhere or use Import to start.
        </div>
      )}
      {error && <div className="absolute inset-0 grid place-items-center p-6 text-center text-rose-400">{error}</div>}
      {photo && (view.mode === 'split' || view.mode === 'side') && (
        <>
          <Tag className="left-3">Before</Tag>
          <Tag className="right-3">After</Tag>
        </>
      )}
      {photo && (
        <div className="pointer-events-none absolute bottom-3 right-3 rounded bg-black/60 px-2 py-1 font-mono text-[11px] text-ink-300">
          {pct}% {renderMs !== null && `· ${renderMs.toFixed(1)} ms`}{stagesRun === 0 ? ' · cached' : stagesRun ? ` · ${stagesRun}/4 stages` : ''}
        </div>
      )}
    </div>
  );
}

type BrushKind = 'add' | 'erase' | 'dodge' | 'burn' | 'heal' | 'clone';

const BRUSH_COLOURS: Record<BrushKind, string> = {
  add: 'rgba(255,64,90,0.35)',
  erase: 'rgba(80,160,255,0.35)',
  dodge: 'rgba(255,220,90,0.35)',
  burn: 'rgba(60,80,200,0.4)',
  heal: 'rgba(120,255,170,0.3)',
  clone: 'rgba(190,120,255,0.3)',
};

/** A dodge & burn layer paints dodge / burn (or erases); other layers paint their mask. */
function layerStrokeMode(kind: LayerKind, mode: LayerStroke['mode']): LayerStroke['mode'] {
  if (kind === 'dodgeBurn') return mode === 'add' ? 'dodge' : mode;
  return mode === 'dodge' || mode === 'burn' ? 'add' : mode;
}

function layerView(layers: readonly RetouchLayer[], id: string): DisplayView['layerMask'] {
  const l = layers.find((x) => x.id === id);
  return l ? { id: l.id, dodgeBurn: l.kind === 'dodgeBurn' } : null;
}

function Tag({ children, className }: { children: React.ReactNode; className: string }) {
  return (
    <span className={`pointer-events-none absolute top-3 rounded bg-black/60 px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-100 ${className}`}>
      {children}
    </span>
  );
}

export function requestZoom(action: 'fit' | 'in' | 'out' | 'actual') {
  window.dispatchEvent(new CustomEvent('fineretouch:zoom', { detail: { action } }));
}
