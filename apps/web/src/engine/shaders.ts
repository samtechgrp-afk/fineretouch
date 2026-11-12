/**
 * WebGL2 shader chain for the live proxy viewport.
 *
 * Mirror of services/ai/app/pipeline.py (the full-resolution server renderer).
 * Constants and stage order must stay identical so the preview matches the export.
 *
 * Coordinate convention: texture row 0 is the image's top row, and every
 * offscreen pass keeps that orientation. Only the display pass flips to the canvas.
 *
 * Pass graph per slider change:
 *   RETOUCH (skin, boost, stray hair, face, colour, tone) -> out
 *   [pro studio on] STUDIO out -> studio (glasses glare, clothing iron / recolour, look LUT;
 *              see components/viewport/shaders)
 *   [bokeh on] DOWNSAMPLE out -> small, BOKEH x2 ("double blur") small -> a -> b,
 *              COMPOSITE out + b by depth -> final
 *   DISPLAY final (or out) with compare modes.
 *
 * The spot healing, boost (2x/3x), stray-hair and bokeh passes are real-time
 * approximations of services/ai/app/retouch_engine.py. The HSL mixer and colour grading
 * mirror services/ai/app/color_engine.py exactly (constants from @fineretouch/shared);
 * everything else mirrors pipeline.py exactly.
 */
import {
  HSL_CENTERS,
  HSL_CHANNELS,
  HSL_HUE_RANGE,
  HSL_LUM_RANGE,
  HSL_NEUTRAL_EDGE,
  SKIN_TONE_GUARD,
} from '@fineretouch/shared';

const glslFloat = (v: number) => (Number.isInteger(v) ? `${v}.0` : `${v}`);
const HUE_CENTERS_GLSL = `float[9](${[...HSL_CHANNELS.map((c) => HSL_CENTERS[c]), 360].map(glslFloat).join(', ')})`;

/** s3 is the broad base used by the 2x/3x skin boost for tone evening. */
export const BLUR_SIGMAS = { s0: 0.0012, s1: 0.004, s2: 0.012, s3: 0.03 } as const;

/** Depth band (0 = subject) over which the circle of confusion ramps up; same as FOCUS_START/END in retouch_engine.py. */
export const FOCUS_START = 0.04;
export const FOCUS_END = 0.45;
/** blurRadius 100 => 2% of the long edge before the multiplier (BOKEH_RADIUS_SHARE). */
export const BOKEH_RADIUS_SHARE = 0.02;
export const BOKEH_TAPS = 48;
/** Kernels are kept at or below this many working pixels; bigger blurs run on a downsampled copy. */
export const BOKEH_MAX_WORKING_COC = 24;

export const VERT = /* glsl */ `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

/** Separable Gaussian: 25 taps spaced sigma/4 apart cover +/-3 sigma at any radius. */
export const BLUR_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uTex;
uniform vec2 uDir;     // (1/w, 0) or (0, 1/h)
uniform float uSigma;  // in pixels
void main() {
  float stepPx = max(uSigma * 0.25, 0.15);
  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  for (int i = -12; i <= 12; i++) {
    float x = float(i) * stepPx;
    float w = exp(-(x * x) / (2.0 * uSigma * uSigma));
    acc += texture(uTex, vUv + uDir * x).rgb * w;
    wsum += w;
  }
  outColor = vec4(acc / wsum, 1.0);
}`;

/**
 * Dark circles reference (v0.14, mirrors engines/under_eye.correct_dark_circles): face skin with
 * the under-eye area cut out, written premultiplied (uPart 0: rgb * w) or as the weight alone
 * (uPart 1). Both are then blurred at the S2 scale; their ratio is the colour of the person's own
 * skin around each eye, without the eye, lashes or background in it.
 */
/** Dark circles reference blur, as a share of the face width (engines/under_eye.UNDER_REF_SIGMA). */
export const UNDER_REF_SIGMA = 0.06;

export const UNDER_REF_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc, uMask;
uniform vec4 uFrame;
uniform int uPart;
void main() {
  vec4 m = texture(uMask, uFrame.xy + vUv * uFrame.zw);
  float w = m.r * (1.0 - smoothstep(0.05, 0.3, m.b));
  outColor = uPart == 0 ? vec4(texture(uSrc, vUv).rgb * w, 1.0) : vec4(vec3(w), 1.0);
}`;

/**
 * Tiled export (v0.12): a tile covers part of the photo. Masks, depth, segmentation and layer
 * masks are whole-photo textures, so they are sampled at fuv(vUv). uFrame = (x0, y0, w, h) of
 * the tile in whole-photo UV; (0, 0, 1, 1) when rendering the whole photo.
 */
export const FRAME = /* glsl */ `
uniform vec4 uFrame;
vec2 fuv(vec2 uv) { return uFrame.xy + uv * uFrame.zw; }
`;

const COMMON = /* glsl */ `
const vec3 LUMA = vec3(0.299, 0.587, 0.114);
float luma(vec3 c) { return dot(c, LUMA); }
float cbOf(vec3 c) { return -0.168736 * c.r - 0.331264 * c.g + 0.5 * c.b + 0.5; }
float crOf(vec3 c) { return 0.5 * c.r - 0.418688 * c.g - 0.081312 * c.b + 0.5; }
vec3 fromYCbCr(float y, float cb, float cr) {
  cb -= 0.5; cr -= 0.5;
  return vec3(y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb);
}
float skinProb(vec3 c) {
  float cb = cbOf(c), cr = crOf(c);
  return smoothstep(0.50, 0.55, cr) * (1.0 - smoothstep(0.66, 0.71, cr))
       * smoothstep(0.27, 0.32, cb) * (1.0 - smoothstep(0.48, 0.53, cb));
}
vec3 toLinear(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}
vec3 toSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
const mat3 RGB2XYZ = mat3(0.4124564, 0.2126729, 0.0193339,
                          0.3575761, 0.7151522, 0.1191920,
                          0.1804375, 0.0721750, 0.9503041);
const vec3 WHITE = vec3(0.95047, 1.0, 1.08883);
const float LAB_EPS = 216.0 / 24389.0;
const float LAB_KAPPA = 24389.0 / 27.0;
vec3 rgbToLab(vec3 c) {
  vec3 xyz = (RGB2XYZ * toLinear(c)) / WHITE;
  vec3 f = mix((LAB_KAPPA * xyz + 16.0) / 116.0, pow(max(xyz, 0.0), vec3(1.0 / 3.0)), step(vec3(LAB_EPS), xyz));
  return vec3(116.0 * f.y - 16.0, 500.0 * (f.x - f.y), 200.0 * (f.y - f.z));
}
vec3 rgb2hsv(vec3 c) {
  float v = max(c.r, max(c.g, c.b));
  float d = v - min(c.r, min(c.g, c.b));
  float s = v > 1e-6 ? d / v : 0.0;
  float dd = max(d, 1e-6);
  float h = v == c.r ? mod((c.g - c.b) / dd, 6.0) : (v == c.g ? (c.b - c.r) / dd + 2.0 : (c.r - c.g) / dd + 4.0);
  return vec3(d > 1e-6 ? h * 60.0 : 0.0, s, v);
}
vec3 hsv2rgb(float h, float s, float v) {
  vec3 k = mod(vec3(5.0, 3.0, 1.0) + mod(h, 360.0) / 60.0, 6.0);
  return v * (1.0 - s * clamp(min(k, 4.0 - k), 0.0, 1.0));
}
vec3 labToRgb(vec3 lab) {
  float fy = (lab.x + 16.0) / 116.0;
  vec3 f = vec3(fy + lab.y / 500.0, fy, fy - lab.z / 200.0);
  vec3 f3 = f * f * f;
  vec3 xyz = mix((116.0 * f - 16.0) / LAB_KAPPA, f3, step(vec3(LAB_EPS), f3));
  xyz.y = lab.x > LAB_KAPPA * LAB_EPS ? fy * fy * fy : lab.x / LAB_KAPPA;
  return toSrgb(inverse(RGB2XYZ) * (xyz * WHITE));
}
`;

export const RETOUCH_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc, uLow0, uLow1, uLow2, uLow3, uMask, uHairRoi;
uniform sampler2D uUnderRefRgb, uUnderRefW; // dark circles reference (UNDER_REF_FRAG, blurred)
uniform sampler2D uEdit;        // user mask strokes: r = add, g = erase (masking.apply_mask_edits)
uniform bool uHasEdit;
uniform bool uHasFace;
uniform float uMultiplier;      // 1..3 skin boost
uniform float uSpotMultiplier;  // 1..3 spot detection reach (max of skin boost and spot healing multiplier)
uniform float uAcne, uDarkSpot;               // 0..1 separate acne / dark spot tools (v0.7)
uniform float uAcneMultiplier, uDarkMultiplier; // 1..3 their detector reach
uniform bool uKeepFreckles;
uniform bool uHair;             // stray hair cleanup enabled and a face ROI exists
uniform float uHairStrength;    // 0..1
uniform mat3 uMatchM;           // colour transfer matrix (L*a*b*), see matchMatrix() in color.ts
uniform bool uCurve;            // histogram-matched L* tone curve available
uniform float uSrcQ[17], uRefQ[17];
// Adjustments normalised to -1..1 or 0..1 on the CPU.
uniform float uExposure, uContrast, uHighlights, uShadows;
uniform float uTemp, uTint, uSat, uVib;
uniform float uSmooth, uTexture, uBlemish, uUnify;
uniform float uRedness, uNoise, uUnderEyeTex; // v0.9 skin engine (app/engines/redness.py, noise.py, under_eye.py)
uniform float uNoiseLevel;                    // measured luma noise level of the source (lib/noise.ts)
uniform float uEye, uDark, uTeeth, uSharpen, uVignette;
uniform bool uMatch;
uniform float uMatchAmount;
uniform vec3 uSrcMean, uRefMean;
uniform bool uProtectMoles;     // spot healing keeps beauty marks
uniform bool uToneGuard;        // CIELAB skin-tone guard on and some skin retouch active
uniform bool uHsl;              // 8-channel mixer active
uniform vec3 uHslK[8];          // per channel (hue, saturation, luminance) in -1..1
uniform bool uGrade;            // colour grading active
uniform vec3 uGradeOff[3];      // shadows / midtones / highlights RGB offsets (see gradeUniforms)
uniform vec3 uGradeBounds;      // shadow pivot, highlight pivot, crossover width
${COMMON}
${FRAME}
const float HUE_C[9] = ${HUE_CENTERS_GLSL};
// Skin-tone guard (retouch_engine.skin_tone_guard): chroma floor and hue lock in CIELAB.
float wrapDeg(float d) { return mod(d + 180.0, 360.0) - 180.0; }
vec3 toneGuard(vec3 c, vec3 orig, vec3 ref) {
  vec3 lab = rgbToLab(c), lo = rgbToLab(orig), lr = rgbToLab(ref);
  float ch = length(lab.yz), cho = length(lo.yz), chr = length(lr.yz);
  float hr = chr < 1e-5 ? 0.0 : degrees(atan(lr.z, lr.y));
  float hue = ch < 1e-5 ? hr : degrees(atan(lab.z, lab.y));
  float ho = cho < 1e-5 ? hr : degrees(atan(lo.z, lo.y));
  float chn = max(ch, ${glslFloat(SKIN_TONE_GUARD.chromaKeep)} * min(cho, chr));
  float d = wrapDeg(hue - hr), dO = wrapDeg(ho - hr);
  float dn = clamp(d, min(dO, 0.0) - ${glslFloat(SKIN_TONE_GUARD.hueTolerance)}, max(dO, 0.0) + ${glslFloat(SKIN_TONE_GUARD.hueTolerance)});
  float lock = smoothstep(${glslFloat(SKIN_TONE_GUARD.minChroma[0])}, ${glslFloat(SKIN_TONE_GUARD.minChroma[1])}, min(chr, chn));
  float turn = (dn - d) * lock;
  if (chn <= ch + 1e-4 && abs(turn) <= 1e-3) return c;
  float hn = radians(hr + d + turn);
  return clamp(labToRgb(vec3(lab.x, chn * cos(hn), chn * sin(hn))), 0.0, 1.0);
}
vec3 hslMixer(vec3 c) {
  vec3 hsv = rgb2hsv(c);
  int lo = 0;
  for (int i = 1; i < 8; i++) if (hsv.x >= HUE_C[i]) lo = i;
  float t = (hsv.x - HUE_C[lo]) / (HUE_C[lo + 1] - HUE_C[lo]);
  t = t * t * (3.0 - 2.0 * t);
  vec3 k = mix(uHslK[lo], uHslK[(lo + 1) % 8], t);
  float g = smoothstep(0.0, ${glslFloat(HSL_NEUTRAL_EDGE)}, hsv.y);
  vec3 o = hsv2rgb(hsv.x + k.x * ${glslFloat(HSL_HUE_RANGE)} * g, clamp(hsv.y * (1.0 + k.y), 0.0, 1.0), hsv.z);
  float y1 = luma(o);
  float keep = y1 > 1e-4 ? luma(c) / max(y1, 1e-4) : 1.0;  // hue / saturation keep luma
  return clamp(o * keep * (1.0 + ${glslFloat(HSL_LUM_RANGE)} * k.z * g * hsv.y), 0.0, 1.0);
}
vec3 colorGrade(vec3 c) {
  float y = clamp(luma(c), 0.0, 1.0);
  float hw = uGradeBounds.z * 0.5;
  float ws = 1.0 - smoothstep(uGradeBounds.x - hw, uGradeBounds.x + hw, y);
  float wh = smoothstep(uGradeBounds.y - hw, uGradeBounds.y + hw, y);
  float wm = clamp(1.0 - ws - wh, 0.0, 1.0);
  return clamp(c + ws * uGradeOff[0] + wm * uGradeOff[1] + wh * uGradeOff[2], 0.0, 1.0);
}
float toneCurve(float L) {
  if (L <= uSrcQ[0]) return uRefQ[0];
  for (int i = 1; i < 17; i++) {
    if (L <= uSrcQ[i]) return mix(uRefQ[i - 1], uRefQ[i], (L - uSrcQ[i - 1]) / max(uSrcQ[i] - uSrcQ[i - 1], 1e-4));
  }
  return uRefQ[16];
}
void main() {
  vec3 src = texture(uSrc, vUv).rgb;
  vec3 low0 = texture(uLow0, vUv).rgb;
  vec3 low1 = texture(uLow1, vUv).rgb;
  vec3 low2 = texture(uLow2, vUv).rgb;
  vec3 low3 = texture(uLow3, vUv).rgb;
  vec4 m = texture(uMask, fuv(vUv)); // r = face skin, g = eyes, b = under-eye, a = mouth

  float p = skinProb(low0);
  float skin = uHasFace ? m.r * (0.35 + 0.65 * p) : p * 0.8;
  // Brush strokes are absolute: "add" paints skin in, "erase" protects it from every skin tool.
  vec2 e = uHasEdit ? texture(uEdit, fuv(vUv)).rg : vec2(0.0);
  skin = max(skin, e.r) * (1.0 - e.g);
  float faceArea = max(m.r, e.r) * (1.0 - e.g);

  vec3 c = src;
  // Skin noise reduction (preview of engines/noise.py): the finest luma band (3x3) is
  // soft-thresholded at 2.5x the measured noise level; chroma is averaged over 5x5 on skin.
  if (uNoise > 0.0) {
    ivec2 sz = textureSize(uSrc, 0);
    ivec2 ip = ivec2(vUv * vec2(sz));
    vec3 b3 = vec3(0.0), b5 = vec3(0.0);
    for (int dy = -2; dy <= 2; dy++)
      for (int dx = -2; dx <= 2; dx++) {
        vec3 t = texelFetch(uSrc, clamp(ip + ivec2(dx, dy), ivec2(0), sz - 1), 0).rgb;
        b5 += t;
        if (abs(dx) <= 1 && abs(dy) <= 1) b3 += t;
      }
    b3 /= 9.0;
    b5 /= 25.0;
    float k = uNoise * skin;
    float fineY = luma(src) - luma(b3);
    float thr = 2.5 * uNoiseLevel;
    float y = luma(src) - clamp(fineY, -thr, thr) * k;
    float cb = cbOf(src), cr = crOf(src);
    float edge = smoothstep(0.02, 0.05, abs(cb - cbOf(b5)) + abs(cr - crOf(b5)));
    float kc = k * (1.0 - edge);
    c = fromYCbCr(y, cb + (cbOf(b5) - cb) * kc, cr + (crOf(b5) - cr) * kc);
  }
  vec3 fine = src - low1;
  vec3 mid = low1 - low2;

  float face = uHasFace ? faceArea : skin;
  // Acne removal (preview of retouch_engine.remove_acne): red spots get the surrounding tone
  // plus their own grain, then excess redness around them is pulled back to the local skin.
  if (uAcne > 0.0) {
    float sens = 1.0 + 0.45 * (uAcneMultiplier - 1.0);
    float r = crOf(low1) - crOf(low2);
    float spot = smoothstep(0.012 / sens, 0.035 / sens, r);
    vec3 clean = low2 + fine * uTexture;
    c += (clean - c) * spot * face * min(1.0, uAcne * (0.9 + 0.1 * sens));
    float flush = smoothstep(0.004, 0.02 / sens, crOf(low2) - crOf(low3)) * face * uAcne * 0.6;
    c = fromYCbCr(luma(c), cbOf(c), crOf(c) - max(crOf(low2) - crOf(low3), 0.0) * flush);
  }
  // Dark spot removal (preview of retouch_engine.remove_dark_spots): dark, non-red marks healed,
  // then brownish pigmentation (tone darker AND lower Cb than the wider skin) evened; strong
  // dark edges (facial shading) are excluded.
  if (uDarkSpot > 0.0) {
    float sens = (1.0 + 0.45 * (uDarkMultiplier - 1.0)) / (uKeepFreckles ? 1.8 : 1.0);
    float d = luma(low2) - luma(low1);
    float r = crOf(low1) - crOf(low2);
    float spot = smoothstep(0.015 / sens, 0.05 / sens, d) * (1.0 - smoothstep(0.008, 0.02, r));
    if (uProtectMoles) spot *= 1.0 - smoothstep(0.1, 0.16, luma(low2) - luma(low0));
    vec3 clean = low2 + fine * uTexture;
    c += (clean - c) * spot * face * min(1.0, uDarkSpot * (0.9 + 0.1 * sens));
    float dp = luma(low3) - luma(low2);
    float pig = smoothstep(0.004, 0.02, dp) * smoothstep(0.002, 0.008, cbOf(low3) - cbOf(low2));
    float edge = smoothstep(0.06, 0.14, dp);
    c += (low3 - low2) * 0.9 * uDarkSpot * (uKeepFreckles ? 0.6 : 1.0) * pig * (1.0 - edge) * skin;
  }

  // Frequency separation (preview of retouch_engine.natural_skin_retouch): tone layer = low1,
  // detail layer = src - low1. The blotch band (low1 - low2) is dropped from the tone layer and
  // the detail layer comes back by the Texture retention factor. Export also runs an adaptive
  // median on the tone layer and heals spots before the split.
  if (uSmooth > 0.0) {
    float edge = smoothstep(0.08, 0.22, abs(luma(fine)) + abs(luma(mid)));
    float k = skin * uSmooth * (1.0 - edge);
    c -= k * ((1.0 - uTexture) * fine + 0.85 * mid);
  }
  // Spot healing (preview of retouch_engine.heal_spots): small spots darker or redder than
  // the surrounding tone get the surrounding tone plus their own grain. The face mask alone
  // gates it, because a red pimple fails the colour-based skin test. 2x/3x widen detection;
  // very dark, neutral spots (beauty marks) are kept unless protection is off.
  if (uBlemish > 0.0) {
    float sens = 1.0 + 0.45 * (uSpotMultiplier - 1.0);
    float d = luma(low2) - luma(low1);
    float r = crOf(low1) - crOf(low2);
    float spot = max(smoothstep(0.015 / sens, 0.05 / sens, d), smoothstep(0.012 / sens, 0.035 / sens, r));
    if (uProtectMoles) spot *= 1.0 - smoothstep(0.1, 0.16, luma(low2) - luma(low0)) * (1.0 - smoothstep(0.01, 0.025, r));
    float area = uHasFace ? faceArea : skin;
    vec3 clean = low2 + fine * uTexture;
    c += (clean - c) * spot * area * min(1.0, uBlemish * (0.9 + 0.1 * sens));
  }
  // Redness reduction (preview of engines/redness.py): excess Cr of the tone layer over the
  // surrounding skin colour pulled back, Cb toward it; luma unchanged. Export compares with a
  // skin-only average; the preview's broad bands also hold background and clothes, so the
  // reference is the redder of low2 and low3 (never a too-neutral one that over-corrects).
  if (uRedness > 0.0) {
    float excess = crOf(low1) - max(crOf(low2), crOf(low3));
    float k = uRedness * smoothstep(0.004, 0.025, excess) * skin;
    c = fromYCbCr(luma(c), cbOf(c) + 0.4 * (cbOf(low2) - cbOf(low1)) * k, crOf(c) - 0.9 * max(excess, 0.0) * k);
  }
  if (uUnify > 0.0) {
    float t = uUnify * 0.8 * skin;
    float y = luma(c);
    float cb = cbOf(c), cr = crOf(c);
    c = fromYCbCr(y, cb + (cbOf(low2) - cb) * t, cr + (crOf(low2) - cr) * t);
  }
  // 2x / 3x boost: stubborn spots in the detail layer, pigment and redness in the colour layer.
  float extra = uMultiplier - 1.0;
  if (extra > 0.0) {
    float edge = smoothstep(0.05, 0.14, abs(luma(low1) - luma(low2)));
    float sens = 1.0 + 0.6 * extra;
    float spot = smoothstep(0.03 / sens, 0.08 / sens, luma(low1) - luma(src)) * (1.0 - edge) * skin;
    vec3 patchc = low1 + (src - low0) * 0.85; // local tone + its own fine grain
    c = mix(c, patchc, spot * min(1.0, 0.5 + 0.25 * extra));
    float pig = smoothstep(0.01, 0.05, luma(low3) - luma(low2)) * (1.0 - edge);
    float redl = smoothstep(0.008, 0.03, crOf(low2) - crOf(low3));
    float lift = min(max(luma(low3) - luma(low2), 0.0), 0.08) * pig * (0.35 + 0.2 * extra);
    float t = clamp(max(pig, redl) * (0.55 + 0.25 * extra) + 0.12 * extra, 0.0, 1.0) * skin;
    float yb = luma(c) + lift * skin;
    c = fromYCbCr(yb, cbOf(c) + (cbOf(low3) - cbOf(low2)) * t, crOf(c) + (crOf(low3) - crOf(low2)) * t);
  }
  // Skin-tone guard: retouched skin never goes ashen or drifts off its undertone.
  if (uToneGuard) {
    c = clamp(c, 0.0, 1.0);
    c = mix(c, toneGuard(c, src, low2), skin);
  }
  // Stray hair: thin strands that differ from their surroundings inside the face-outline band.
  if (uHair) {
    float roi = texture(uHairRoi, fuv(vUv)).r;
    if (roi > 0.0) {
      float d = luma(low1) - luma(src);
      float thin = max(d, -0.8 * d);
      float coarse = abs(luma(low1) - luma(low2));
      float line = smoothstep(0.05 - 0.025 * uHairStrength, 0.12 - 0.04 * uHairStrength, thin)
                 * (1.0 - smoothstep(0.05, 0.12, coarse));
      c = mix(c, low1, line * roi * (0.6 + 0.4 * uHairStrength));
    }
  }
  if (uHasFace) {
    if (uEye > 0.0) {
      float e = uEye * m.g;
      c = c * (1.0 + 0.35 * e) + 0.6 * e * (src - low0);
    }
    float underEye = m.b * (1.0 - e.g); // Protect strokes apply to the under-eye tools too
    // Dark circles (v0.14): the shadow (low1) is lifted as a whole toward the surrounding face
    // skin (masked blur), so texture rides on top and nothing grey comes in.
    if (uDark > 0.0) {
      float wr = texture(uUnderRefW, vUv).r;
      float d = uDark * underEye * step(0.02, wr);
      vec3 ref = texture(uUnderRefRgb, vUv).rgb / max(wr, 0.02);
      float y = luma(c) + max(0.0, luma(ref) - luma(low1)) * 0.85 * d;
      float cb = cbOf(c) + (cbOf(ref) - cbOf(low1)) * 0.6 * d;
      float cr = crOf(c) + (crOf(ref) - crOf(low1)) * 0.6 * d;
      c = fromYCbCr(y, cb, cr);
    }
    // Under-eye texture (preview of engines/under_eye.py): the crease band (low0 - low1) faded,
    // dark lines fully at 100%, lighter ridges by 35%; pores (src - low0) kept.
    if (uUnderEyeTex > 0.0) {
      float band = luma(low0) - luma(low1);
      float k = uUnderEyeTex * underEye;
      c -= k * (min(band, 0.0) + 0.35 * max(band, 0.0));
    }
    if (uTeeth > 0.0) {
      float tp = smoothstep(0.30, 0.55, luma(c)) * m.a * uTeeth;
      float neutral = max(c.b, (c.r + c.g) * 0.5);
      c.b += (neutral - c.b) * tp;
      c *= 1.0 + 0.08 * tp;
    }
  }
  c = clamp(c, 0.0, 1.0);

  // Statistical colour transfer in L*a*b*: covariance-aligning matrix + L* tone curve.
  if (uMatch && uMatchAmount > 0.0) {
    vec3 lab = rgbToLab(c);
    vec3 matched = uMatchM * (lab - uSrcMean) + uRefMean;
    if (uCurve) matched.x = toneCurve(lab.x);
    c = mix(c, labToRgb(matched), uMatchAmount);
  }

  // Exposure and white balance in linear light.
  vec3 lin = toLinear(c) * exp2(uExposure * 2.0);
  lin *= vec3(1.0 + 0.2 * uTemp, 1.0 - 0.12 * uTint, 1.0 - 0.2 * uTemp);
  c = toSrgb(lin);

  float y = luma(c);
  c += 0.25 * uHighlights * smoothstep(0.5, 1.0, y) + 0.25 * uShadows * (1.0 - smoothstep(0.0, 0.5, y));
  c = (c - 0.5) * (1.0 + 0.5 * uContrast) + 0.5;

  y = luma(c);
  float sat = clamp(max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b)), 0.0, 1.0);
  c = y + (c - y) * (1.0 + uSat);
  c = y + (c - y) * (1.0 + uVib * (1.0 - sat));

  // Colour mixer and grading (color_engine.py, stage 8b).
  c = clamp(c, 0.0, 1.0);
  if (uHsl) c = hslMixer(c);
  if (uGrade) c = colorGrade(c);

  if (uSharpen > 0.0) c += uSharpen * 1.2 * (src - low0) * (1.0 - 0.8 * skin * uSmooth);

  if (uVignette != 0.0) {
    float d = length(fuv(vUv) - 0.5) / sqrt(0.5);
    c *= 1.0 + uVignette * 0.6 * smoothstep(0.35, 1.0, d);
  }
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

/** Draws the processed/original textures into the canvas with zoom, pan and a split line. */
export const DISPLAY_FRAG = /* glsl */ `#version 300 es
precision highp float;
out vec4 outColor;
uniform sampler2D uProcessed, uOriginal, uMask, uOverlay, uEdit, uLayerMask;
uniform bool uHasEdit;
uniform int uLayerShow;  // 0 skin mask overlay, 1 a layer's coverage, 2 a dodge & burn layer
uniform bool uOverlayOn;  // pro studio QA overlay (premultiplied by its own alpha below)
uniform vec2 uCanvas;   // canvas size in device pixels
uniform vec2 uImage;    // proxy size in pixels
uniform float uScale;   // device pixels per image pixel
uniform vec2 uOffset;   // device-pixel position of the image's top-left corner
uniform int uMode;      // 0 edited, 1 original, 2 split, 3 side by side, 4 difference
uniform float uSplit;   // split position in device pixels (mode 2)
uniform bool uShowMask;
uniform vec3 uBg;
uniform float uDpr;
const vec3 LUMA = vec3(0.299, 0.587, 0.114);
float luma(vec3 c) { return dot(c, LUMA); }

vec3 sampleTex(sampler2D t, vec2 ip) {
  // Crisp pixels when zoomed in past 2x, smooth filtering otherwise.
  if (uScale >= 2.0) return texelFetch(t, ivec2(clamp(floor(ip), vec2(0.0), uImage - 1.0)), 0).rgb;
  return texture(t, ip / uImage).rgb;
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, uCanvas.y - gl_FragCoord.y);
  bool original = uMode == 1;
  if (uMode == 3) {
    // Side by side: each half shows the same view (the CPU fits the image to a half-width pane).
    float half_ = uCanvas.x * 0.5;
    if (abs(p.x - half_) < uDpr) { outColor = vec4(uBg * 0.5, 1.0); return; }
    original = p.x < half_;
    if (!original) p.x -= half_;
  }
  if (uMode == 2) {
    if (abs(p.x - uSplit) < uDpr) { outColor = vec4(1.0); return; }
    original = p.x < uSplit;
  }
  vec2 ip = (p - uOffset) / uScale;
  if (any(lessThan(ip, vec2(0.0))) || any(greaterThanEqual(ip, uImage))) {
    outColor = vec4(uBg, 1.0);
    return;
  }
  vec3 c = original ? sampleTex(uOriginal, ip) : sampleTex(uProcessed, ip);
  if (uMode == 4) {
    // Retouch heat map: where and how strongly the edit changed the photo (QA for over-retouching).
    vec3 o = sampleTex(uOriginal, ip);
    float d = clamp(length(c - o) * 4.0, 0.0, 1.0);
    vec3 heat = d < 0.5 ? mix(vec3(0.05, 0.06, 0.1), vec3(0.93, 0.6, 0.2), d * 2.0) : mix(vec3(0.93, 0.6, 0.2), vec3(1.0), (d - 0.5) * 2.0);
    c = mix(luma(o) * vec3(0.35), heat, smoothstep(0.02, 0.1, d));
  }
  if (uShowMask && uLayerShow > 0) {
    vec4 lm = texture(uLayerMask, ip / uImage);
    if (uLayerShow == 1) c = mix(c, vec3(1.0, 0.25, 0.35), lm.r * 0.5);
    else {
      c = mix(c, vec3(1.0, 0.85, 0.3), lm.g * 0.6);
      c = mix(c, vec3(0.25, 0.4, 1.0), lm.b * 0.6);
    }
  } else if (uShowMask) {
    vec4 m = texture(uMask, ip / uImage);
    vec2 e = uHasEdit ? texture(uEdit, ip / uImage).rg : vec2(0.0);
    c = mix(c, vec3(1.0, 0.25, 0.35), max(m.r, e.r) * (1.0 - e.g) * 0.45);
    c = mix(c, vec3(0.2, 0.9, 0.4), m.g * 0.6);
    c = mix(c, vec3(0.3, 0.5, 1.0), m.b * 0.6);
    c = mix(c, vec3(1.0, 0.9, 0.2), m.a * 0.6);
  }
  if (uOverlayOn && !original) {
    vec4 o = texture(uOverlay, ip / uImage);
    c = mix(c, o.rgb, o.a);
  }
  outColor = vec4(c, 1.0);
}`;


/**
 * Local retouch layer (v0.10, Phase 5), one pass per layer: frequency separation and the
 * correction brush through the layer's coverage, or dodge & burn in linear light. A port of
 * services/ai/app/engines/local.py apply_local_layers. uLowA / uLowB are the current image
 * blurred at the split radius and at FS_SMOOTH_RATIO times it.
 */
export const FS_SMOOTH_RATIO = 5;
export const LOCAL_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uImg, uLowA, uLowB, uLayer;
uniform bool uDodgeBurn, uFs, uCorr;
uniform float uGain;          // density * opacity
uniform vec3 uLuma;           // lumaMin, lumaMax (0..1), softness (0..1)
uniform float uSmooth, uTexture;
uniform float uDb;            // dodge & burn strength 0..1
uniform int uRange;           // 0 all, 1 shadows, 2 midtones, 3 highlights
uniform float uExposure, uContrast, uSat, uTemp, uTint; // -1..1
${COMMON}
${FRAME}
float lumaGate(float y) {
  if (uLuma.x <= 0.0 && uLuma.y >= 1.0) return 1.0;
  float s = max(0.005, uLuma.z * 0.25);
  float g = 1.0;
  if (uLuma.x > 0.0) g *= smoothstep(uLuma.x - s, uLuma.x, y);
  if (uLuma.y < 1.0) g *= 1.0 - smoothstep(uLuma.y, uLuma.y + s, y);
  return g;
}
float rangeWeight(float y) {
  if (uRange == 1) return 1.0 - smoothstep(0.2, 0.6, y);
  if (uRange == 3) return smoothstep(0.4, 0.8, y);
  if (uRange == 2) return 1.0 - smoothstep(0.15, 0.5, abs(y - 0.5));
  return 1.0;
}
void main() {
  vec3 c = clamp(texture(uImg, vUv).rgb, 0.0, 1.0);
  vec4 L = texture(uLayer, fuv(vUv));
  float gate = lumaGate(luma(c)) * uGain;
  if (uDodgeBurn) {
    float stops = 1.5 * uDb * (L.g - L.b) * gate * rangeWeight(luma(c));
    outColor = vec4(toSrgb(toLinear(c) * exp2(stops)), 1.0);
    return;
  }
  float m = L.r * gate;
  if (uFs) {
    vec3 low = texture(uLowA, vUv).rgb;
    vec3 delta = texture(uLowB, vUv).rgb - low;
    float edge = 1.0 - smoothstep(0.04, 0.12, abs(luma(delta)));
    vec3 o = low + delta * (uSmooth * edge) + (c - low) * uTexture;
    c = c + (o - c) * m;
  }
  if (uCorr) {
    vec3 lin = toLinear(c) * exp2(uExposure * 2.0) * vec3(1.0 + 0.2 * uTemp, 1.0 - 0.12 * uTint, 1.0 - 0.2 * uTemp);
    vec3 o = toSrgb(lin);
    o = (o - 0.5) * (1.0 + 0.5 * uContrast) + 0.5;
    float y = luma(o);
    o = clamp(y + (o - y) * (1.0 + uSat), 0.0, 1.0);
    c = c + (o - c) * m;
  }
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

/** Box-filtered downsample (4x4 taps across the source footprint). */
export const DOWNSAMPLE_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uTex;
uniform vec2 uFootprint; // destination texel size in source UV
void main() {
  vec3 acc = vec3(0.0);
  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++)
      acc += texture(uTex, vUv + (vec2(float(x), float(y)) / 3.0 - 0.5) * uFootprint).rgb;
  outColor = vec4(acc / 16.0, 1.0);
}`;

/**
 * Depth-aware gather bokeh. Each output pixel averages aperture-shaped samples within
 * its own circle of confusion; a sample only counts if its own CoC reaches back to this
 * pixel, so the sharp subject never smears into the background. Highlights are boosted
 * in linear light so bright points bloom into disc or hexagon shapes. Run twice
 * (full radius, then 45%): the "double blur" hides the sparse-tap pattern.
 */
export const BOKEH_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uImg, uDepth;
uniform vec2 uTexel;          // 1 / working size
uniform float uMaxCoc;        // working pixels
uniform float uRadiusScale;   // 1.0 then 0.45
uniform float uGain;          // highlight boost 0..1
uniform vec2 uKernel[${BOKEH_TAPS}];  // unit disk or hexagon
${COMMON}
${FRAME}
float coc(vec2 uv) { return uMaxCoc * uRadiusScale * smoothstep(${FOCUS_START}, ${FOCUS_END}, texture(uDepth, fuv(uv)).r); }
vec3 boosted(vec3 srgb) {
  vec3 lin = toLinear(srgb);
  return lin * (1.0 + uGain * 6.0 * smoothstep(0.55, 0.95, luma(lin)));
}
void main() {
  vec3 centre = texture(uImg, vUv).rgb;
  float cc = coc(vUv);
  if (cc < 0.5) { outColor = vec4(centre, 1.0); return; }
  vec3 acc = boosted(centre);
  float wsum = 1.0;
  for (int i = 0; i < ${BOKEH_TAPS}; i++) {
    vec2 o = uKernel[i] * cc;
    vec2 uv = vUv + o * uTexel;
    float w = clamp(coc(uv) - length(o) + 1.0, 0.0, 1.0);
    acc += boosted(texture(uImg, uv).rgb) * w;
    wsum += w;
  }
  outColor = vec4(toSrgb(clamp(acc / wsum, 0.0, 1.0)), 1.0);
}`;

/** Blend the sharp retouch with the (upsampled) bokeh by each pixel's circle of confusion. */
export const COMPOSITE_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSharp, uBlur, uDepth;
uniform float uMaxCocFull; // pixels at the output resolution
${FRAME}
void main() {
  float coc = uMaxCocFull * smoothstep(${FOCUS_START}, ${FOCUS_END}, texture(uDepth, fuv(vUv)).r);
  float a = smoothstep(0.35, 1.5, coc);
  outColor = vec4(mix(texture(uSharp, vUv).rgb, texture(uBlur, vUv).rgb, a), 1.0);
}`;

/**
 * Sample positions for the bokeh gather: a Vogel (golden-angle) spiral filling the unit
 * disk, or the same spiral stretched to a hexagon (flat aperture blades) for `hex`.
 */
export function apertureKernel(n: number, shape: 'circle' | 'hex'): Float32Array {
  const out = new Float32Array(n * 2);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const r = Math.sqrt((i + 0.5) / n);
    const t = i * golden;
    let k = 1;
    if (shape === 'hex') {
      const seg = Math.PI / 3;
      const a = ((t % seg) + seg) % seg;
      k = Math.cos(Math.PI / 6) / Math.cos(a - Math.PI / 6);
    }
    out[i * 2] = Math.cos(t) * r * k;
    out[i * 2 + 1] = Math.sin(t) * r * k;
  }
  return out;
}
