import type { Adjustments, ColorStats } from '@fineretouch/shared';
import { BLUR_FRAG, BLUR_SIGMAS, DISPLAY_FRAG, RETOUCH_FRAG, VERT } from './shaders';

type GL = WebGL2RenderingContext;
type TexSource = ImageBitmap | ImageData | HTMLCanvasElement | OffscreenCanvas | HTMLImageElement;

interface Target {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
}

export type CompareMode = 'edited' | 'original' | 'split' | 'side';

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
  dpr: number;
  background: [number, number, number];
}

export interface RenderInfo {
  hasFace: boolean;
  /** L*a*b* stats of the unedited image; needed for colour matching. */
  sourceStats: ColorStats | null;
}

/**
 * GPU retouch pipeline. Blur bands depend only on the source image, so they are
 * computed once per photo; a slider change only re-runs the single retouch pass
 * plus the display pass, which keeps interaction well under a frame on a proxy.
 */
export class Renderer {
  readonly gl: GL;
  private blurProg: WebGLProgram;
  private retouchProg: WebGLProgram;
  private displayProg: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private src: WebGLTexture | null = null;
  private mask: WebGLTexture | null = null;
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
    this.retouchProg = this.program(VERT, RETOUCH_FRAG);
    this.displayProg = this.program(VERT, DISPLAY_FRAG);
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    for (const p of [this.blurProg, this.retouchProg, this.displayProg]) {
      const loc = gl.getAttribLocation(p, 'aPos');
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      }
    }
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  }

  get maxTextureSize(): number {
    return this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE) as number;
  }

  /** Upload a new photo and precompute its three Gaussian bands. */
  setSource(image: TexSource, width: number, height: number): void {
    const gl = this.gl;
    this.disposeImageResources();
    this.width = width;
    this.height = height;
    this.src = this.texture(width, height, image);
    this.tmp = this.target(width, height);
    this.bands = [this.target(width, height), this.target(width, height), this.target(width, height)];
    this.out = this.target(width, height);
    this.setMask(null);

    const L = Math.max(width, height);
    const sigmas = [BLUR_SIGMAS.s0, BLUR_SIGMAS.s1, BLUR_SIGMAS.s2].map((s) => Math.max(s * L, 0.6));
    gl.useProgram(this.blurProg);
    gl.viewport(0, 0, width, height);
    sigmas.forEach((sigma, i) => {
      this.blurPass(this.src!, this.tmp!, [1 / width, 0], sigma);
      this.blurPass(this.tmp!.tex, this.bands[i], [0, 1 / height], sigma);
    });
  }

  /**
   * RGBA mask (r skin, g eyes, b under-eye, a mouth), or null to clear. It is
   * sampled by UV, so it may be smaller than the source (masks are soft anyway).
   */
  setMask(data: Uint8Array | null, width = 1, height = 1): void {
    const gl = this.gl;
    if (this.mask) gl.deleteTexture(this.mask);
    this.mask = data ? this.texture(width, height, null, data) : this.texture(1, 1, null, new Uint8Array(4));
  }

  /** Run the retouch pass into the offscreen output texture. */
  render(adj: Adjustments, info: RenderInfo): void {
    if (!this.src || !this.out) return;
    const gl = this.gl;
    const p = this.retouchProg;
    gl.useProgram(p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.out.fbo);
    gl.viewport(0, 0, this.width, this.height);
    this.bindTex(p, 'uSrc', this.src, 0);
    this.bindTex(p, 'uLow0', this.bands[0].tex, 1);
    this.bindTex(p, 'uLow1', this.bands[1].tex, 2);
    this.bindTex(p, 'uLow2', this.bands[2].tex, 3);
    this.bindTex(p, 'uMask', this.mask!, 4);
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
    u('uEye', adj.eyeBrighten / 100);
    u('uDark', adj.darkCircles / 100);
    u('uTeeth', adj.teethWhiten / 100);
    u('uSharpen', adj.sharpen / 100);
    u('uVignette', adj.vignette / 100);
    const match = adj.colorMatch && info.sourceStats ? adj.colorMatch : null;
    gl.uniform1i(this.loc(p, 'uMatch'), match ? 1 : 0);
    if (match && info.sourceStats) {
      u('uMatchAmount', match.amount / 100);
      gl.uniform3fv(this.loc(p, 'uSrcMean'), info.sourceStats.mean);
      gl.uniform3fv(this.loc(p, 'uSrcStd'), info.sourceStats.std);
      gl.uniform3fv(this.loc(p, 'uRefMean'), match.reference.mean);
      gl.uniform3fv(this.loc(p, 'uRefStd'), match.reference.std);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Composite into the visible canvas. */
  display(view: DisplayView): void {
    if (!this.src || !this.out) return;
    const gl = this.gl;
    const p = this.displayProg;
    const cw = gl.drawingBufferWidth;
    const ch = gl.drawingBufferHeight;
    gl.useProgram(p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, cw, ch);
    this.bindTex(p, 'uProcessed', this.out.tex, 0);
    this.bindTex(p, 'uOriginal', this.src, 1);
    this.bindTex(p, 'uMask', this.mask!, 2);
    gl.uniform2f(this.loc(p, 'uCanvas'), cw, ch);
    gl.uniform2f(this.loc(p, 'uImage'), this.width, this.height);
    gl.uniform1f(this.loc(p, 'uScale'), view.scale);
    gl.uniform2f(this.loc(p, 'uOffset'), view.offsetX, view.offsetY);
    const mode = { edited: 0, original: 1, split: 2, side: 3 }[view.mode];
    gl.uniform1i(this.loc(p, 'uMode'), mode);
    gl.uniform1f(this.loc(p, 'uSplit'), view.split * cw);
    gl.uniform1i(this.loc(p, 'uShowMask'), view.showMask ? 1 : 0);
    gl.uniform3fv(this.loc(p, 'uBg'), view.background);
    gl.uniform1f(this.loc(p, 'uDpr'), view.dpr);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Read the retouched image back (top row first), e.g. for export. */
  readPixels(): Uint8ClampedArray<ArrayBuffer> {
    const gl = this.gl;
    const px = new Uint8ClampedArray(new ArrayBuffer(this.width * this.height * 4));
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.out!.fbo);
    gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return px;
  }

  /** `loseContext` frees the GPU context immediately; only for canvases that will not be reused. */
  dispose(loseContext = false): void {
    this.disposeImageResources();
    const gl = this.gl;
    [this.blurProg, this.retouchProg, this.displayProg].forEach((p) => gl.deleteProgram(p));
    gl.deleteVertexArray(this.vao);
    // A canvas keeps one context for its lifetime, so losing it would break a
    // remount of the viewport (e.g. React StrictMode); export canvases opt in.
    if (loseContext) gl.getExtension('WEBGL_lose_context')?.loseContext();
  }

  // --- internals -------------------------------------------------------------

  private blurPass(input: WebGLTexture, output: Target, dir: [number, number], sigma: number) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, output.fbo);
    this.bindTex(this.blurProg, 'uTex', input, 0);
    gl.uniform2f(this.loc(this.blurProg, 'uDir'), dir[0], dir[1]);
    gl.uniform1f(this.loc(this.blurProg, 'uSigma'), sigma);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private disposeImageResources() {
    const gl = this.gl;
    for (const t of [this.src, this.mask]) if (t) gl.deleteTexture(t);
    for (const t of [this.tmp, this.out, ...this.bands]) {
      if (t) {
        gl.deleteTexture(t.tex);
        gl.deleteFramebuffer(t.fbo);
      }
    }
    this.src = this.mask = null;
    this.tmp = this.out = null;
    this.bands = [];
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
