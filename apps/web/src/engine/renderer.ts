import type { Adjustments, ColorStats } from '@fineretouch/shared';
import { gradingIsDefault, hslIsDefault, layerIsActive, resolveAdjustments, splitSigma } from '@fineretouch/shared';
import { matchMatrix, toneCurveKnots } from './color';
import { gradeBounds, gradeUniform, hslUniform } from './colorMixer';
import {
  BLUR_FRAG,
  UNDER_REF_FRAG,
  UNDER_REF_SIGMA,
  BLUR_SIGMAS,
  BOKEH_FRAG,
  BOKEH_MAX_WORKING_COC,
  BOKEH_RADIUS_SHARE,
  BOKEH_TAPS,
  COMPOSITE_FRAG,
  DISPLAY_FRAG,
  DOWNSAMPLE_FRAG,
  FS_SMOOTH_RATIO,
  LOCAL_FRAG,
  RETOUCH_FRAG,
  VERT,
  apertureKernel,
} from './shaders';
import { decodeLut } from '@fineretouch/shared';
import { OVERLAY_FRAG, OVERLAY_MODES, STUDIO_FRAG, type StudioOverlay } from '@/components/viewport/shaders';

type GL = WebGL2RenderingContext;
type TexSource = ImageBitmap | ImageData | HTMLCanvasElement | OffscreenCanvas | HTMLImageElement;

interface Target {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
}

export type CompareMode = 'edited' | 'original' | 'split' | 'side' | 'diff';

export interface DisplayView {
  mode: CompareMode;
  /** Device pixels per image pixel. */
  scale: number;
  /** Device-pixel position of the image's top-left corner inside its pane. */
  offsetX: number;
  offsetY: number;
  /** Split line position as a fraction of the canvas width. */
  split: number;
  showMask: boolean;
  /** Pro studio QA overlay (glare veil, texture retention, garment mask); display only. */
  overlay?: StudioOverlay;
  /** With showMask: show this local retouch layer's mask instead of the skin mask. */
  layerMask?: { id: string; dodgeBurn: boolean } | null;
  dpr: number;
  background: [number, number, number];
}

/** Where a tile sits in the whole photo (tiled export, v0.12). */
export interface TileFrame {
  /** Tile origin in whole-photo pixels. */
  x: number;
  y: number;
  /** Whole-photo size in pixels. */
  fullW: number;
  fullH: number;
}

export interface RenderInfo {
  hasFace: boolean;
  /** L*a*b* stats of the unedited image; needed for colour matching. */
  sourceStats: ColorStats | null;
  /** Measured luma noise level of the source (engine/noise.ts), for skin noise reduction. */
  noiseLevel?: number;
  /** Widest face as a share of the whole photo's width (analysis box), for face-scaled stages. */
  faceShare?: number;
}

/**
 * GPU retouch pipeline. Blur bands depend only on the source image, so they are
 * computed once per photo; a slider change only re-runs the single retouch pass
 * plus the display pass, which keeps interaction well under a frame on a proxy.
 */
export class Renderer {
  readonly gl: GL;
  /** The whole photo this source belongs to (itself unless it is an export tile). */
  private frame: TileFrame = { x: 0, y: 0, fullW: 1, fullH: 1 };
  /**
   * Bumped when a stage's input textures change (source and masks: retouch; lens and garment
   * masks: studio; layer masks: local; depth: bokeh). Invalidates that stage and the ones after it.
   */
  private versions = { retouch: 0, studio: 0, local: 0, bokeh: 0 };
  /** Stage cache keys of the last render (v0.12 preview caching). */
  private stageKeys = { retouch: '', studio: '', local: '', bokeh: '' };
  /** Stages the last render() actually ran (for the viewport's timing readout and tests). */
  lastStages: Array<'retouch' | 'studio' | 'local' | 'bokeh'> = [];
  private blurProg: WebGLProgram;
  private underRefProg: WebGLProgram;
  /** Dark circles reference: blurred premultiplied skin colour and blurred weight (S2). */
  private underRef: Target[] = [];
  private retouchProg: WebGLProgram;
  private displayProg: WebGLProgram;
  private downProg: WebGLProgram;
  private bokehProg: WebGLProgram;
  private compositeProg: WebGLProgram;
  private studioProg: WebGLProgram;
  private overlayProg: WebGLProgram;
  private localProg: WebGLProgram;
  /** The photo as loaded; `src` is the same texture, or the photo with heal / clone strokes. */
  private orig: WebGLTexture | null = null;
  /** Local retouch layer masks by layer id (r coverage, g dodge, b burn) and the data uploaded. */
  private layerTex = new Map<string, { tex: WebGLTexture; data: Uint8Array }>();
  /** Ping-pong targets for the layer passes and the frequency-separation blurs. */
  private localTargets: Target[] = [];
  /** Lens region (single channel) for glare reduction, from the eye landmarks. */
  private lens: WebGLTexture | null = null;
  /** Engine segmentation (r clothes, g hair, b person) for the clothing tools. */
  private seg: WebGLTexture | null = null;
  private lut: WebGLTexture | null = null;
  private lutKey = '';
  private lutSize = 0;
  /** STUDIO pass output and the QA overlay (never exported). */
  private studio: Target | null = null;
  private overlayTarget: Target | null = null;
  /** Retouch output after the studio pass: what bokeh and export read. */
  private base: Target | null = null;
  /** Output of the local layers (the base when there are none); bokeh input. */
  private local: Target | null = null;
  private hairRoi: WebGLTexture | null = null;
  private depth: WebGLTexture | null = null;
  /** Bokeh working targets at reduced size: [downsampled, pass 1, pass 2]. */
  private bokehTargets: Target[] = [];
  private bokehSize: [number, number] = [0, 0];
  private final: Target | null = null;
  /** Texture shown by display() and read by readPixels(): `final` when bokeh ran, else `out`. */
  private shown: Target | null = null;
  private kernels = { circle: apertureKernel(BOKEH_TAPS, 'circle'), hex: apertureKernel(BOKEH_TAPS, 'hex') };
  private vao: WebGLVertexArrayObject;
  private src: WebGLTexture | null = null;
  private mask: WebGLTexture | null = null;
  private edit: WebGLTexture | null = null;
  private tmp: Target | null = null;
  private bands: Target[] = [];
  private out: Target | null = null;
  private uniformCache = new Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>();
  width = 0;
  height = 0;

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas) {
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: false }) as GL | null;
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    this.gl = gl;
    this.blurProg = this.program(VERT, BLUR_FRAG);
    this.underRefProg = this.program(VERT, UNDER_REF_FRAG);
    this.retouchProg = this.program(VERT, RETOUCH_FRAG);
    this.displayProg = this.program(VERT, DISPLAY_FRAG);
    this.downProg = this.program(VERT, DOWNSAMPLE_FRAG);
    this.bokehProg = this.program(VERT, BOKEH_FRAG);
    this.compositeProg = this.program(VERT, COMPOSITE_FRAG);
    this.studioProg = this.program(VERT, STUDIO_FRAG);
    this.overlayProg = this.program(VERT, OVERLAY_FRAG);
    this.localProg = this.program(VERT, LOCAL_FRAG);
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    for (const p of this.programs()) {
      const loc = gl.getAttribLocation(p, 'aPos');
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      }
    }
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  }

  /**
   * Long edge of the whole photo: every radius that scales with the photo (blur bands,
   * frequency separation, bokeh) uses it, so an export tile gets the same radii as the whole.
   */
  get longEdge(): number {
    return Math.max(this.frame.fullW, this.frame.fullH);
  }

  /** useProgram plus the tile frame uniform (programs without it ignore it). */
  private use(p: WebGLProgram): void {
    const gl = this.gl;
    gl.useProgram(p);
    const f = this.frame;
    const loc = this.loc(p, 'uFrame');
    if (loc) gl.uniform4f(loc, f.x / f.fullW, f.y / f.fullH, this.width / f.fullW, this.height / f.fullH);
  }

  get maxTextureSize(): number {
    return this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE) as number;
  }

  private programs(): WebGLProgram[] {
    return [this.blurProg, this.underRefProg, this.retouchProg, this.displayProg, this.downProg, this.bokehProg, this.compositeProg, this.studioProg, this.overlayProg, this.localProg];
  }

  /** Upload a new photo and precompute its four Gaussian bands. */
  setSource(image: TexSource, width: number, height: number, frame?: TileFrame): void {
    this.disposeImageResources();
    this.width = width;
    this.height = height;
    this.frame = frame ?? { x: 0, y: 0, fullW: width, fullH: height };
    this.src = this.orig = this.texture(width, height, image);
    this.tmp = this.target(width, height);
    this.bands = [0, 1, 2, 3].map(() => this.target(width, height));
    this.underRef = [];
    this.out = this.target(width, height);
    this.final = this.target(width, height);
    this.studio = this.target(width, height);
    this.overlayTarget = null;
    this.shown = this.base = this.local = this.out;
    this.setMask(null);
    this.setEdit(null);
    this.setHairRoi(null);
    this.setDepth(null);
    this.setLensRoi(null);
    this.setSegmentation(null);
    this.computeBands();
    this.versions.retouch++;
  }

  /**
   * The photo with its heal / clone strokes (RGBA, same size as the source), or null for the
   * photo as loaded. Every retouch stage then works on it; 'Original' still shows the photo.
   */
  setHealedSource(pixels: Uint8Array | Uint8ClampedArray | null): void {
    if (!this.orig) return;
    if (this.src && this.src !== this.orig) this.gl.deleteTexture(this.src);
    this.versions.retouch++;
    this.src = pixels ? this.texture(this.width, this.height, null, new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength)) : this.orig;
    this.computeBands();
  }

  private computeBands(): void {
    const gl = this.gl;
    const { width, height } = this;
    const L = this.longEdge;
    const sigmas = [BLUR_SIGMAS.s0, BLUR_SIGMAS.s1, BLUR_SIGMAS.s2, BLUR_SIGMAS.s3].map((s) => Math.max(s * L, 0.6));
    this.use(this.blurProg);
    gl.viewport(0, 0, width, height);
    sigmas.forEach((sigma, i) => {
      this.blurPass(this.src!, this.tmp!, [1 / width, 0], sigma);
      this.blurPass(this.tmp!.tex, this.bands[i], [0, 1 / height], sigma);
    });
  }

  /** Upload a local retouch layer's mask (re-uploaded only when the data changed). */
  setLayerMask(id: string, data: Uint8Array, width: number, height: number): void {
    const hit = this.layerTex.get(id);
    if (hit?.data === data) return;
    if (hit) this.gl.deleteTexture(hit.tex);
    this.layerTex.set(id, { tex: this.texture(width, height, null, data), data });
    this.versions.local++;
  }

  /** Drop the masks of layers that are gone or inactive. */
  pruneLayerMasks(keep: Set<string>): void {
    for (const [id, t] of this.layerTex) {
      if (!keep.has(id)) {
        this.versions.local++;
        this.gl.deleteTexture(t.tex);
        this.layerTex.delete(id);
      }
    }
  }

  /**
   * RGBA mask (r skin, g eyes, b under-eye, a mouth), or null to clear. It is
   * sampled by UV, so it may be smaller than the source (masks are soft anyway).
   */
  setMask(data: Uint8Array | null, width = 1, height = 1): void {
    const gl = this.gl;
    this.versions.retouch++;
    if (this.mask) gl.deleteTexture(this.mask);
    this.mask = data ? this.texture(width, height, null, data) : this.texture(1, 1, null, new Uint8Array(4));
  }

  /** User mask strokes as RGBA (r = add, g = erase), sampled by UV; null when there are none. */
  setEdit(data: Uint8Array | null, width = 1, height = 1): void {
    this.versions.retouch++;
    if (this.edit) this.gl.deleteTexture(this.edit);
    this.edit = data ? this.texture(width, height, null, data) : null;
  }

  /** Single-channel (0..255) stray-hair region, or null to disable the hair pass. */
  setHairRoi(data: Uint8Array | null, width = 1, height = 1): void {
    this.versions.retouch++;
    if (this.hairRoi) this.gl.deleteTexture(this.hairRoi);
    this.hairRoi = data ? this.texture(width, height, null, expandGrey(data)) : null;
  }

  /** Depth map (0 = subject, 255 = far): a decoded image (red channel used) or single-channel bytes. */
  setDepth(source: TexSource | Uint8Array | null, width = 1, height = 1): void {
    this.versions.bokeh++;
    if (this.depth) this.gl.deleteTexture(this.depth);
    if (!source) this.depth = null;
    else if (source instanceof Uint8Array) this.depth = this.texture(width, height, null, expandGrey(source));
    else this.depth = this.texture(width, height, source);
  }

  /** Single-channel (0..255) eyeglass lens region, or null (glare reduction off). */
  setLensRoi(data: Uint8Array | null, width = 1, height = 1): void {
    this.versions.studio++;
    if (this.lens) this.gl.deleteTexture(this.lens);
    this.lens = data ? this.texture(width, height, null, expandGrey(data)) : null;
  }

  /** Segmentation from the engine (PNG: r clothes, g hair, b person), or null. */
  setSegmentation(source: TexSource | null, width = 1, height = 1): void {
    this.versions.studio++;
    if (this.seg) this.gl.deleteTexture(this.seg);
    this.seg = source ? this.texture(width, height, source) : null;
  }

  get hasSegmentation(): boolean {
    return this.seg !== null;
  }

  /** Upload a look LUT as an RGBA16F 3D texture (x = red, y = green, z = blue). Cached by content. */
  private ensureLut(look: NonNullable<Adjustments['lut']>): boolean {
    const key = `${look.size}:${look.data.length}:${look.data.slice(0, 64)}:${look.data.slice(-64)}`;
    if (key === this.lutKey && this.lut) return true;
    const gl = this.gl;
    let table: Float32Array;
    try {
      table = decodeLut(look);
    } catch {
      return false;
    }
    const n = look.size;
    const rgba = new Float32Array(n * n * n * 4);
    for (let i = 0; i < n * n * n; i++) {
      rgba[i * 4] = table[i * 3];
      rgba[i * 4 + 1] = table[i * 3 + 1];
      rgba[i * 4 + 2] = table[i * 3 + 2];
      rgba[i * 4 + 3] = 1;
    }
    if (this.lut) gl.deleteTexture(this.lut);
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA16F, n, n, n, 0, gl.RGBA, gl.FLOAT, rgba);
    this.lut = tex;
    this.lutKey = key;
    this.lutSize = n;
    return true;
  }

  get hasDepth(): boolean {
    return this.depth !== null;
  }

  /** Run the retouch pass (and the bokeh chain when enabled) into the offscreen output. */
  render(edit: Adjustments, info: RenderInfo): void {
    if (!this.src || !this.out) return;
    // Per-effect enable / 1x-2x-3x / opacity and global strength -> effective engine values
    // (same function as services/ai/app/effects.py, so preview and export agree).
    const adj = resolveAdjustments(edit);
    // Preview caching (v0.12): each stage re-runs only when its own settings or anything
    // before it changed. Panning, zooming or a bokeh / layer change reuse the earlier stages.
    const k = stageKeys(adj, info, this.versions);
    this.lastStages = [];
    if (k.retouch !== this.stageKeys.retouch) {
      this.retouchPass(adj, info);
      this.lastStages.push('retouch');
    }
    if (k.studio !== this.stageKeys.studio) {
      this.base = this.studioPass(adj, info.hasFace) ? this.studio! : this.out;
      this.lastStages.push('studio');
    }
    if (k.local !== this.stageKeys.local) {
      this.local = this.localPass(adj, this.base!);
      this.lastStages.push('local');
    }
    if (k.bokeh !== this.stageKeys.bokeh) {
      const b = adj.depthBokeh;
      const maxCoc = (b.blurRadius / 100) * BOKEH_RADIUS_SHARE * this.longEdge * (b.bokehMultiplier / 100);
      this.shown = b.enabled && this.depth && maxCoc >= 0.75 ? this.bokeh(maxCoc, b.apertureShape, b.highlightBoost / 100, this.local!) : this.local;
      this.lastStages.push('bokeh');
    }
    this.stageKeys = k;
  }

  /** Forget the stage cache (the next render runs every stage). */
  invalidate(): void {
    this.versions.retouch++;
  }

  private retouchPass(adj: ReturnType<typeof resolveAdjustments>, info: RenderInfo): void {
    if (!this.src || !this.out) return;
    const gl = this.gl;
    const dark = adj.darkCircles > 0 && info.hasFace;
    if (dark) this.computeUnderRef(info.faceShare ?? 0.25);
    const p = this.retouchProg;
    this.use(p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.out.fbo);
    gl.viewport(0, 0, this.width, this.height);
    this.bindTex(p, 'uSrc', this.src, 0);
    this.bindTex(p, 'uLow0', this.bands[0].tex, 1);
    this.bindTex(p, 'uLow1', this.bands[1].tex, 2);
    this.bindTex(p, 'uLow2', this.bands[2].tex, 3);
    this.bindTex(p, 'uMask', this.mask!, 4);
    this.bindTex(p, 'uLow3', this.bands[3].tex, 5);
    this.bindTex(p, 'uHairRoi', this.hairRoi ?? this.mask!, 6);
    this.bindTex(p, 'uEdit', this.edit ?? this.mask!, 7);
    this.bindTex(p, 'uUnderRefRgb', dark ? this.underRef[1].tex : this.mask!, 8);
    this.bindTex(p, 'uUnderRefW', dark ? this.underRef[2].tex : this.mask!, 9);
    gl.uniform1i(this.loc(p, 'uHasEdit'), this.edit ? 1 : 0);
    const u = (name: string, v: number) => gl.uniform1f(this.loc(p, name), v);
    gl.uniform1i(this.loc(p, 'uHasFace'), info.hasFace ? 1 : 0);
    u('uExposure', adj.exposure / 100);
    u('uContrast', adj.contrast / 100);
    u('uHighlights', adj.highlights / 100);
    u('uShadows', adj.shadows / 100);
    u('uTemp', adj.temperature / 100);
    u('uTint', adj.tint / 100);
    u('uSat', adj.saturation / 100);
    u('uVib', adj.vibrance / 100);
    u('uSmooth', adj.skinSmoothing / 100);
    u('uTexture', adj.skinTexture / 100);
    u('uBlemish', adj.blemishRemoval / 100);
    u('uUnify', adj.skinToneUnify / 100);
    u('uRedness', adj.rednessReduction / 100);
    u('uNoise', adj.skinNoise / 100);
    u('uNoiseLevel', info.noiseLevel ?? 0.004);
    u('uUnderEyeTex', adj.underEyeTexture / 100);
    u('uEye', adj.eyeBrighten / 100);
    u('uSkinBright', adj.skinBrighten / 100);
    u('uDark', adj.darkCircles / 100);
    u('uTeeth', adj.teethWhiten / 100);
    u('uSharpen', adj.sharpen / 100);
    u('uVignette', adj.vignette / 100);
    u('uMultiplier', adj.skinMultiplier);
    u('uSpotMultiplier', adj.spotMultiplier);
    u('uAcne', adj.acneRemoval / 100);
    u('uDarkSpot', adj.darkSpotRemoval / 100);
    u('uAcneMultiplier', adj.acneMultiplier);
    u('uDarkMultiplier', adj.darkSpotMultiplier);
    gl.uniform1i(this.loc(p, 'uKeepFreckles'), adj.keepFreckles ? 1 : 0);
    gl.uniform1i(this.loc(p, 'uProtectMoles'), adj.protectBeautyMarks ? 1 : 0);
    const skinWork =
      adj.skinSmoothing > 0 || adj.blemishRemoval > 0 || adj.skinToneUnify > 0 || adj.skinMultiplier > 1 || adj.acneRemoval > 0 || adj.darkSpotRemoval > 0;
    gl.uniform1i(this.loc(p, 'uToneGuard'), adj.skinToneGuard && skinWork ? 1 : 0);
    const hsl = !hslIsDefault(adj.hsl);
    gl.uniform1i(this.loc(p, 'uHsl'), hsl ? 1 : 0);
    if (hsl) gl.uniform3fv(this.loc(p, 'uHslK'), hslUniform(adj.hsl));
    const grade = !gradingIsDefault(adj.colorGrading);
    gl.uniform1i(this.loc(p, 'uGrade'), grade ? 1 : 0);
    if (grade) {
      gl.uniform3fv(this.loc(p, 'uGradeOff'), gradeUniform(adj.colorGrading));
      gl.uniform3fv(this.loc(p, 'uGradeBounds'), gradeBounds(adj.colorGrading));
    }
    const hair = adj.strayHairRemoval.enabled && this.hairRoi !== null;
    gl.uniform1i(this.loc(p, 'uHair'), hair ? 1 : 0);
    u('uHairStrength', adj.strayHairRemoval.strength / 100);
    const match = adj.colorMatch && info.sourceStats ? adj.colorMatch : null;
    gl.uniform1i(this.loc(p, 'uMatch'), match ? 1 : 0);
    if (match && info.sourceStats) {
      u('uMatchAmount', match.amount / 100);
      gl.uniform3fv(this.loc(p, 'uSrcMean'), info.sourceStats.mean);
      gl.uniform3fv(this.loc(p, 'uRefMean'), match.reference.mean);
      // Row-major matrix; WebGL2 transposes on upload so GLSL sees M * v.
      gl.uniformMatrix3fv(this.loc(p, 'uMatchM'), true, matchMatrix(info.sourceStats, match.reference));
      const curve = Boolean(info.sourceStats.lQuantiles && match.reference.lQuantiles);
      gl.uniform1i(this.loc(p, 'uCurve'), curve ? 1 : 0);
      if (curve) {
        const [qs, qr] = toneCurveKnots(info.sourceStats.lQuantiles!, match.reference.lQuantiles!);
        gl.uniform1fv(this.loc(p, 'uSrcQ'), qs);
        gl.uniform1fv(this.loc(p, 'uRefQ'), qr);
      }
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Glasses glare, clothing iron / recolour and the look LUT. Returns false when nothing is on. */
  private studioPass(adj: Adjustments, hasFace: boolean): boolean {
    const gl = this.gl;
    const glare = adj.glareReduction > 0 && hasFace && this.lens !== null;
    const cl = adj.clothing;
    const iron = cl.enabled && cl.wrinkleStrength > 0 && this.seg !== null;
    const recolour = cl.enabled && (cl.hueShift !== 0 || cl.saturation !== 0) && this.seg !== null;
    const lut = adj.lut !== null && adj.lut.amount > 0 && this.ensureLut(adj.lut);
    if (!glare && !iron && !recolour && !lut) return false;
    const p = this.studioProg;
    this.use(p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.studio!.fbo);
    gl.viewport(0, 0, this.width, this.height);
    this.bindTex(p, 'uImg', this.out!.tex, 0);
    this.bindTex(p, 'uLow0', this.bands[0].tex, 1);
    this.bindTex(p, 'uLow1', this.bands[1].tex, 2);
    this.bindTex(p, 'uLow3', this.bands[3].tex, 3);
    this.bindTex(p, 'uLens', this.lens ?? this.mask!, 4);
    this.bindTex(p, 'uSeg', this.seg ?? this.mask!, 5);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_3D, lut ? this.lut : null);
    gl.uniform1i(this.loc(p, 'uLut'), 6);
    gl.uniform1i(this.loc(p, 'uGlare'), glare ? 1 : 0);
    gl.uniform1i(this.loc(p, 'uIron'), iron ? 1 : 0);
    gl.uniform1i(this.loc(p, 'uRecolour'), recolour ? 1 : 0);
    gl.uniform1i(this.loc(p, 'uLutOn'), lut ? 1 : 0);
    gl.uniform1f(this.loc(p, 'uGlareStrength'), adj.glareReduction / 100);
    gl.uniform1f(this.loc(p, 'uIronStrength'), cl.wrinkleStrength / 100);
    gl.uniform1f(this.loc(p, 'uHue'), cl.hueShift);
    gl.uniform1f(this.loc(p, 'uSatShift'), cl.saturation / 100);
    gl.uniform1f(this.loc(p, 'uLutSize'), this.lutSize || 2);
    gl.uniform1f(this.loc(p, 'uLutAmount'), (adj.lut?.amount ?? 0) / 100);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_3D, null);
    return true;
  }

  /**
   * Local retouch layers (v0.10), bottom first, one pass each (ping-pong). Frequency separation
   * blurs the current image at the layer's split radius (and FS_SMOOTH_RATIO times it) first.
   */
  private localPass(adj: Adjustments, input: Target): Target {
    const layers = adj.localRetouch.layers.filter((l) => layerIsActive(l) && this.layerTex.has(l.id));
    if (!layers.length) return input;
    const gl = this.gl;
    const { width, height } = this;
    if (!this.localTargets.length) this.localTargets = [0, 1, 2, 3].map(() => this.target(width, height));
    const [a, b, lowA, lowB] = this.localTargets;
    let cur = input;
    for (const l of layers) {
      const dodgeBurn = l.kind === 'dodgeBurn';
      const fs = !dodgeBurn && (l.toneSmoothing > 0 || l.texture !== 100);
      const corr = !dodgeBurn && (l.exposure !== 0 || l.contrast !== 0 || l.saturation !== 0 || l.temperature !== 0 || l.tint !== 0);
      if (fs) {
        const sigma = splitSigma(l.splitRadius, this.longEdge);
        this.use(this.blurProg);
        gl.viewport(0, 0, width, height);
        this.blurPass(cur.tex, this.tmp!, [1 / width, 0], sigma);
        this.blurPass(this.tmp!.tex, lowA, [0, 1 / height], sigma);
        this.blurPass(cur.tex, this.tmp!, [1 / width, 0], sigma * FS_SMOOTH_RATIO);
        this.blurPass(this.tmp!.tex, lowB, [0, 1 / height], sigma * FS_SMOOTH_RATIO);
      }
      const out = cur === a ? b : a;
      const p = this.localProg;
      this.use(p);
      gl.bindFramebuffer(gl.FRAMEBUFFER, out.fbo);
      gl.viewport(0, 0, width, height);
      this.bindTex(p, 'uImg', cur.tex, 0);
      this.bindTex(p, 'uLowA', lowA.tex, 1);
      this.bindTex(p, 'uLowB', lowB.tex, 2);
      this.bindTex(p, 'uLayer', this.layerTex.get(l.id)!.tex, 3);
      gl.uniform1i(this.loc(p, 'uDodgeBurn'), dodgeBurn ? 1 : 0);
      gl.uniform1i(this.loc(p, 'uFs'), fs ? 1 : 0);
      gl.uniform1i(this.loc(p, 'uCorr'), corr ? 1 : 0);
      gl.uniform1f(this.loc(p, 'uGain'), (l.mask.density / 100) * (l.opacity / 100));
      gl.uniform3f(this.loc(p, 'uLuma'), l.mask.lumaMin / 100, l.mask.lumaMax / 100, l.mask.lumaSoftness / 100);
      gl.uniform1f(this.loc(p, 'uSmooth'), l.toneSmoothing / 100);
      gl.uniform1f(this.loc(p, 'uTexture'), l.texture / 100);
      gl.uniform1f(this.loc(p, 'uDb'), l.dodgeBurnStrength / 100);
      gl.uniform1i(this.loc(p, 'uRange'), { all: 0, shadows: 1, midtones: 2, highlights: 3 }[l.dodgeBurnRange]);
      gl.uniform1f(this.loc(p, 'uExposure'), l.exposure / 100);
      gl.uniform1f(this.loc(p, 'uContrast'), l.contrast / 100);
      gl.uniform1f(this.loc(p, 'uSat'), l.saturation / 100);
      gl.uniform1f(this.loc(p, 'uTemp'), l.temperature / 100);
      gl.uniform1f(this.loc(p, 'uTint'), l.tint / 100);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      cur = out;
    }
    return cur;
  }

  /** Render the QA overlay into its own target (display only), or drop it. */
  private overlayPass(mode: StudioOverlay, glareStrength: number): WebGLTexture | null {
    if (mode === 'none' || !this.src) return null;
    if (mode === 'glare' && !this.lens) return null;
    if (mode === 'clothing' && !this.seg) return null;
    const gl = this.gl;
    this.overlayTarget ??= this.target(this.width, this.height);
    const p = this.overlayProg;
    this.use(p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.overlayTarget.fbo);
    gl.viewport(0, 0, this.width, this.height);
    this.bindTex(p, 'uSrc', this.src, 0);
    this.bindTex(p, 'uEdited', (this.local ?? this.base ?? this.out)!.tex, 1);
    this.bindTex(p, 'uLow1', this.bands[1].tex, 2);
    this.bindTex(p, 'uLow3', this.bands[3].tex, 3);
    this.bindTex(p, 'uLens', this.lens ?? this.mask!, 4);
    this.bindTex(p, 'uSeg', this.seg ?? this.mask!, 5);
    this.bindTex(p, 'uMask', this.mask!, 6);
    gl.uniform1i(this.loc(p, 'uMode'), OVERLAY_MODES[mode]);
    gl.uniform1f(this.loc(p, 'uGlareStrength'), glareStrength);
    const ring = Math.max(1.5, BLUR_SIGMAS.s0 * this.longEdge);
    gl.uniform2f(this.loc(p, 'uRing'), ring / this.width, ring / this.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return this.overlayTarget.tex;
  }

  /** Downsample, two gather passes, composite. Returns the target holding the result. */
  private bokeh(maxCoc: number, shape: 'circle' | 'hex', gain: number, input: Target): Target {
    const gl = this.gl;
    const s = Math.min(1, BOKEH_MAX_WORKING_COC / maxCoc);
    const bw = Math.max(1, Math.round(this.width * s));
    const bh = Math.max(1, Math.round(this.height * s));
    if (this.bokehSize[0] !== bw || this.bokehSize[1] !== bh) {
      this.disposeTargets(this.bokehTargets);
      this.bokehTargets = [0, 1, 2].map(() => this.target(bw, bh));
      this.bokehSize = [bw, bh];
    }
    const [small, a, bt] = this.bokehTargets;
    gl.viewport(0, 0, bw, bh);

    this.use(this.downProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, small.fbo);
    this.bindTex(this.downProg, 'uTex', input.tex, 0);
    gl.uniform2f(this.loc(this.downProg, 'uFootprint'), 1 / bw, 1 / bh);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const p = this.bokehProg;
    this.use(p);
    this.bindTex(p, 'uDepth', this.depth!, 1);
    gl.uniform2f(this.loc(p, 'uTexel'), 1 / bw, 1 / bh);
    gl.uniform1f(this.loc(p, 'uMaxCoc'), maxCoc * s);
    gl.uniform2fv(this.loc(p, 'uKernel'), this.kernels[shape]);
    const pass = (input: Target, output: Target, radiusScale: number, g: number) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, output.fbo);
      this.bindTex(p, 'uImg', input.tex, 0);
      gl.uniform1f(this.loc(p, 'uRadiusScale'), radiusScale);
      gl.uniform1f(this.loc(p, 'uGain'), g);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    pass(small, a, 1, gain);
    pass(a, bt, 0.45, 0);

    const c = this.compositeProg;
    this.use(c);
    gl.viewport(0, 0, this.width, this.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.final!.fbo);
    this.bindTex(c, 'uSharp', input.tex, 0);
    this.bindTex(c, 'uBlur', bt.tex, 1);
    this.bindTex(c, 'uDepth', this.depth!, 2);
    gl.uniform1f(this.loc(c, 'uMaxCocFull'), maxCoc);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return this.final!;
  }

  /** Composite into the visible canvas. */
  display(view: DisplayView, glareStrength = 0.6): void {
    if (!this.src || !this.out) return;
    const gl = this.gl;
    const overlay = this.overlayPass(view.overlay ?? 'none', glareStrength);
    const p = this.displayProg;
    const cw = gl.drawingBufferWidth;
    const ch = gl.drawingBufferHeight;
    this.use(p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, cw, ch);
    this.bindTex(p, 'uProcessed', (this.shown ?? this.out).tex, 0);
    this.bindTex(p, 'uOriginal', this.orig ?? this.src, 1);
    this.bindTex(p, 'uMask', this.mask!, 2);
    const lm = view.layerMask ? this.layerTex.get(view.layerMask.id) : undefined;
    gl.uniform1i(this.loc(p, 'uLayerShow'), lm ? (view.layerMask!.dodgeBurn ? 2 : 1) : 0);
    this.bindTex(p, 'uLayerMask', lm?.tex ?? this.mask!, 5);
    gl.uniform2f(this.loc(p, 'uCanvas'), cw, ch);
    gl.uniform2f(this.loc(p, 'uImage'), this.width, this.height);
    gl.uniform1f(this.loc(p, 'uScale'), view.scale);
    gl.uniform2f(this.loc(p, 'uOffset'), view.offsetX, view.offsetY);
    const mode = { edited: 0, original: 1, split: 2, side: 3, diff: 4 }[view.mode];
    gl.uniform1i(this.loc(p, 'uMode'), mode);
    gl.uniform1f(this.loc(p, 'uSplit'), view.split * cw);
    gl.uniform1i(this.loc(p, 'uShowMask'), view.showMask ? 1 : 0);
    gl.uniform1i(this.loc(p, 'uOverlayOn'), overlay ? 1 : 0);
    this.bindTex(p, 'uOverlay', overlay ?? this.mask!, 3);
    this.bindTex(p, 'uEdit', this.edit ?? this.mask!, 4);
    gl.uniform1i(this.loc(p, 'uHasEdit'), this.edit ? 1 : 0);
    gl.uniform3fv(this.loc(p, 'uBg'), view.background);
    gl.uniform1f(this.loc(p, 'uDpr'), view.dpr);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Read the retouched image back (top row first), e.g. for export; `rect` reads part of it. */
  readPixels(rect?: { x: number; y: number; width: number; height: number }): Uint8ClampedArray<ArrayBuffer> {
    const gl = this.gl;
    const r = rect ?? { x: 0, y: 0, width: this.width, height: this.height };
    const px = new Uint8ClampedArray(new ArrayBuffer(r.width * r.height * 4));
    gl.bindFramebuffer(gl.FRAMEBUFFER, (this.shown ?? this.out)!.fbo);
    // Texture row 0 is the top image row, so y counts from the top here too.
    gl.readPixels(r.x, r.y, r.width, r.height, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return px;
  }

  /** `loseContext` frees the GPU context immediately; only for canvases that will not be reused. */
  dispose(loseContext = false): void {
    this.disposeImageResources();
    const gl = this.gl;
    this.programs().forEach((p) => gl.deleteProgram(p));
    if (this.lut) gl.deleteTexture(this.lut);
    this.lut = null;
    this.lutKey = '';
    gl.deleteVertexArray(this.vao);
    // A canvas keeps one context for its lifetime, so losing it would break a
    // remount of the viewport (e.g. React StrictMode); export canvases opt in.
    if (loseContext) gl.getExtension('WEBGL_lose_context')?.loseContext();
  }

  // --- internals -------------------------------------------------------------

  /**
   * Dark circles reference (shaders UNDER_REF_FRAG): skin * colour and skin weight, each blurred
   * at the S2 scale. Built on demand (only while Dark circles is on) into the bands' size.
   */
  private computeUnderRef(faceShare: number): void {
    const gl = this.gl;
    const { width, height } = this;
    if (!this.underRef.length) this.underRef = [this.target(width, height), this.target(width, height), this.target(width, height)];
    const [scratch, rgb, wgt] = this.underRef;
    // UNDER_REF_SIGMA of the face width (engines/under_eye.UNDER_REF_SIGMA), in this source's pixels.
    const sigma = Math.max(UNDER_REF_SIGMA * faceShare * this.frame.fullW, 0.6);
    gl.viewport(0, 0, width, height);
    for (const [part, out] of [[0, rgb], [1, wgt]] as const) {
      this.use(this.underRefProg);
      gl.bindFramebuffer(gl.FRAMEBUFFER, scratch.fbo);
      this.bindTex(this.underRefProg, 'uSrc', this.src!, 0);
      this.bindTex(this.underRefProg, 'uMask', this.mask!, 1);
      gl.uniform1i(this.loc(this.underRefProg, 'uPart'), part);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.use(this.blurProg);
      this.blurPass(scratch.tex, this.tmp!, [1 / width, 0], sigma);
      this.blurPass(this.tmp!.tex, out, [0, 1 / height], sigma);
    }
  }

  private blurPass(input: WebGLTexture, output: Target, dir: [number, number], sigma: number) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, output.fbo);
    this.bindTex(this.blurProg, 'uTex', input, 0);
    gl.uniform2f(this.loc(this.blurProg, 'uDir'), dir[0], dir[1]);
    gl.uniform1f(this.loc(this.blurProg, 'uSigma'), sigma);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private disposeTargets(list: Array<Target | null>) {
    for (const t of list) {
      if (t) {
        this.gl.deleteTexture(t.tex);
        this.gl.deleteFramebuffer(t.fbo);
      }
    }
  }

  private disposeImageResources() {
    const gl = this.gl;
    if (this.orig && this.orig !== this.src) gl.deleteTexture(this.orig);
    for (const t of [this.src, this.mask, this.edit, this.hairRoi, this.depth, this.lens, this.seg]) if (t) gl.deleteTexture(t);
    for (const t of this.layerTex.values()) gl.deleteTexture(t.tex);
    this.layerTex.clear();
    this.disposeTargets([this.tmp, this.out, this.final, this.studio, this.overlayTarget, ...this.bands, ...this.underRef, ...this.bokehTargets, ...this.localTargets]);
    this.localTargets = [];
    this.orig = null;
    this.src = this.mask = this.edit = this.hairRoi = this.depth = this.lens = this.seg = null;
    this.tmp = this.out = this.final = this.shown = this.studio = this.overlayTarget = this.base = this.local = null;
    this.bands = [];
    this.underRef = [];
    this.bokehTargets = [];
    this.bokehSize = [0, 0];
  }

  private texture(w: number, h: number, image: TexSource | null, pixels?: Uint8Array): WebGLTexture {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (image) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, image as TexImageSource);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels ?? null);
    }
    return tex;
  }

  private target(w: number, h: number): Target {
    const gl = this.gl;
    const tex = this.texture(w, h, null);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fbo };
  }

  private bindTex(p: WebGLProgram, name: string, tex: WebGLTexture, unit: number) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(this.loc(p, name), unit);
  }

  private loc(p: WebGLProgram, name: string): WebGLUniformLocation | null {
    let m = this.uniformCache.get(p);
    if (!m) this.uniformCache.set(p, (m = new Map()));
    if (!m.has(name)) m.set(name, this.gl.getUniformLocation(p, name));
    return m.get(name)!;
  }

  private program(vs: string, fs: string): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`Shader compile failed: ${gl.getShaderInfoLog(s)}`);
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, 'aPos');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`Program link failed: ${gl.getProgramInfoLog(p)}`);
    return p;
  }
}

/** Cheap string hash (FNV-1a) for big values in cache keys, e.g. LUT data. */
function hash(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return `${str.length}:${(h >>> 0).toString(36)}`;
}

/**
 * Cache keys of the four render stages. Each key includes the one before it, so a change
 * re-runs its stage and every stage after it. Strokes are left out: they reach the GPU as
 * textures, whose uploads bump that stage's version.
 */
export function stageKeys(
  adj: ReturnType<typeof resolveAdjustments>,
  info: RenderInfo,
  v: { retouch: number; studio: number; local: number; bokeh: number },
) {
  const { localRetouch, depthBokeh, glareReduction, glareForce, clothing, lut, skinMask: _skinMask, ...rest } = adj;
  void _skinMask;
  const retouch = `${v.retouch}|${JSON.stringify(rest)}|${info.hasFace}|${info.noiseLevel ?? ''}|${JSON.stringify(info.sourceStats)}`;
  const studio = `${retouch}|${v.studio}|${glareReduction}|${glareForce}|${JSON.stringify(clothing)}|${lut ? `${lut.size}:${lut.amount}:${hash(lut.data)}` : ''}`;
  const layers = localRetouch.layers.map((l) => ({ ...l, mask: { ...l.mask, strokes: l.mask.strokes.length } }));
  const local = `${studio}|${v.local}|${JSON.stringify(layers)}`;
  const bokeh = `${local}|${v.bokeh}|${JSON.stringify(depthBokeh)}`;
  return { retouch, studio, local, bokeh };
}

/** Single channel -> RGBA (grey in every channel) for upload as an RGBA8 texture. */
function expandGrey(data: Uint8Array): Uint8Array {
  const out = new Uint8Array(data.length * 4);
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = v;
    out[i * 4 + 3] = 255;
  }
  return out;
}
