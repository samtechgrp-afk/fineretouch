'use client';

/**
 * CPU / GPU selection (v0.12). Two export engines render the same edit:
 *
 *   GPU  the WebGL2 shader chain of the live preview, run at full size (tiled when needed).
 *        Fast on any real GPU, including the Ryzen iGPU. No CUDA, no vendor driver needed.
 *   CPU  the local AI engine (services/ai), the reference implementation: full-strength spot
 *        healing, stray hair and bokeh, and 16-bit camera RAW development. Slower.
 *
 * "Auto" uses the GPU when the browser has a hardware WebGL2 device and the CPU engine when it
 * only has a software rasteriser (SwiftShader, llvmpipe, Microsoft Basic Render), which would
 * be far slower than the CPU engine itself.
 */

import type { GpuBudget } from './export';

export type EnginePref = 'auto' | 'gpu' | 'cpu';
export type ExportEngine = 'browser' | 'server';

export interface GpuInfo {
  available: boolean;
  /** Unmasked renderer string when the browser exposes it, else the generic one. */
  renderer: string;
  maxTexture: number;
  software: boolean;
}

export interface ProcessingPrefs {
  engine: EnginePref;
  gpuBudget: GpuBudget;
}

const KEY = 'fineretouch.processing.v1';
export const DEFAULT_PROCESSING: Readonly<ProcessingPrefs> = Object.freeze({ engine: 'auto', gpuBudget: 'balanced' });

const SOFTWARE = /swiftshader|llvmpipe|softpipe|software|basic render|microsoft basic/i;

export function isSoftwareRenderer(name: string): boolean {
  return SOFTWARE.test(name);
}

let cached: GpuInfo | null = null;

/** WebGL2 device facts (cached; one short-lived context). */
export function detectGpu(): GpuInfo {
  if (cached) return cached;
  try {
    const gl = new OffscreenCanvas(1, 1).getContext('webgl2');
    if (!gl) return (cached = { available: false, renderer: 'WebGL2 not available', maxTexture: 0, software: true });
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = String(dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    const maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return (cached = { available: true, renderer, maxTexture, software: isSoftwareRenderer(renderer) });
  } catch {
    return (cached = { available: false, renderer: 'WebGL2 not available', maxTexture: 0, software: true });
  }
}

/** The export engine a preference resolves to, with the reason (shown in the export dialog). */
export function resolveEngine(pref: EnginePref, gpu: GpuInfo, cpuOnline: boolean): { engine: ExportEngine; reason: string } {
  if (pref === 'gpu') return { engine: 'browser', reason: 'GPU chosen in settings.' };
  if (pref === 'cpu') return cpuOnline ? { engine: 'server', reason: 'CPU engine chosen in settings.' } : { engine: 'browser', reason: 'The CPU engine is not running, so the GPU is used.' };
  if (!gpu.available) return cpuOnline ? { engine: 'server', reason: 'No WebGL2 GPU here, so the CPU engine renders.' } : { engine: 'browser', reason: 'No WebGL2 GPU and no CPU engine.' };
  if (gpu.software && cpuOnline) return { engine: 'server', reason: `Only a software GPU (${gpu.renderer.slice(0, 40)}) is available, so the CPU engine is faster.` };
  return { engine: 'browser', reason: `GPU: ${gpu.renderer.slice(0, 60)}.` };
}

export function loadProcessingPrefs(): ProcessingPrefs {
  try {
    const raw = typeof window !== 'undefined' ? window.localStorage.getItem(KEY) : null;
    const v = raw ? (JSON.parse(raw) as Partial<ProcessingPrefs>) : {};
    return {
      engine: v.engine === 'gpu' || v.engine === 'cpu' ? v.engine : 'auto',
      gpuBudget: v.gpuBudget === 'low' || v.gpuBudget === 'high' ? v.gpuBudget : 'balanced',
    };
  } catch {
    return { ...DEFAULT_PROCESSING };
  }
}

export function saveProcessingPrefs(p: ProcessingPrefs): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    /* storage blocked: the session keeps the choice in memory */
  }
}
