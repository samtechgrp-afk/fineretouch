/**
 * WebGL2 shader chain for the live proxy viewport.
 *
 * Mirror of services/ai/app/pipeline.py (the full-resolution server renderer).
 * Constants and stage order must stay identical so the preview matches the export.
 *
 * Coordinate convention: texture row 0 is the image's top row, and every
 * offscreen pass keeps that orientation. Only the display pass flips to the canvas.
 */

export const BLUR_SIGMAS = { s0: 0.0012, s1: 0.004, s2: 0.012 } as const;

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
uniform sampler2D uSrc, uLow0, uLow1, uLow2, uMask;
uniform bool uHasFace;
// Adjustments normalised to -1..1 or 0..1 on the CPU.
uniform float uExposure, uContrast, uHighlights, uShadows;
uniform float uTemp, uTint, uSat, uVib;
uniform float uSmooth, uTexture, uBlemish, uUnify;
uniform float uEye, uDark, uTeeth, uSharpen, uVignette;
uniform bool uMatch;
uniform float uMatchAmount;
uniform vec3 uSrcMean, uSrcStd, uRefMean, uRefStd;
${COMMON}
void main() {
  vec3 src = texture(uSrc, vUv).rgb;
  vec3 low0 = texture(uLow0, vUv).rgb;
  vec3 low1 = texture(uLow1, vUv).rgb;
  vec3 low2 = texture(uLow2, vUv).rgb;
  vec4 m = texture(uMask, vUv); // r = face skin, g = eyes, b = under-eye, a = mouth

  float p = skinProb(low0);
  float skin = uHasFace ? m.r * (0.35 + 0.65 * p) : p * 0.8;

  vec3 c = src;
  vec3 fine = src - low1;
  vec3 mid = low1 - low2;

  // Frequency separation: drop the blotchy mid band, keep a share of the pores.
  if (uSmooth > 0.0) {
    float edge = smoothstep(0.08, 0.22, abs(luma(fine)) + abs(luma(mid)));
    float k = skin * uSmooth * (1.0 - edge);
    c -= k * ((1.0 - uTexture) * fine + 0.85 * mid);
  }
  // Blemishes: small spots darker or redder than the surrounding tone.
  if (uBlemish > 0.0) {
    float dark = smoothstep(0.015, 0.05, luma(low2) - luma(low1));
    float red = smoothstep(0.012, 0.035, crOf(low1) - crOf(low2));
    float spot = max(dark, red) * skin;
    vec3 clean = low2 + fine * uTexture;
    c += (clean - c) * spot * uBlemish;
  }
  if (uUnify > 0.0) {
    float t = uUnify * 0.8 * skin;
    float y = luma(c);
    float cb = cbOf(c), cr = crOf(c);
    c = fromYCbCr(y, cb + (cbOf(low2) - cb) * t, cr + (crOf(low2) - cr) * t);
  }
  if (uHasFace) {
    if (uEye > 0.0) {
      float e = uEye * m.g;
      c = c * (1.0 + 0.35 * e) + 0.6 * e * (src - low0);
    }
    if (uDark > 0.0) {
      float d = uDark * m.b;
      float y = luma(c);
      float cb = cbOf(c), cr = crOf(c);
      y += max(0.0, luma(low2) * 1.06 - y) * d;
      c = fromYCbCr(y, cb + (cbOf(low2) - cb) * 0.5 * d, cr + (crOf(low2) - cr) * 0.5 * d);
    }
    if (uTeeth > 0.0) {
      float tp = smoothstep(0.30, 0.55, luma(c)) * m.a * uTeeth;
      float neutral = max(c.b, (c.r + c.g) * 0.5);
      c.b += (neutral - c.b) * tp;
      c *= 1.0 + 0.08 * tp;
    }
  }
  c = clamp(c, 0.0, 1.0);

  // Reinhard colour transfer in L*a*b*.
  if (uMatch && uMatchAmount > 0.0) {
    vec3 lab = rgbToLab(c);
    vec3 matched = (lab - uSrcMean) / max(uSrcStd, vec3(1e-3)) * uRefStd + uRefMean;
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

  if (uSharpen > 0.0) c += uSharpen * 1.2 * (src - low0) * (1.0 - 0.8 * skin * uSmooth);

  if (uVignette != 0.0) {
    float d = length(vUv - 0.5) / sqrt(0.5);
    c *= 1.0 + uVignette * 0.6 * smoothstep(0.35, 1.0, d);
  }
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

/** Draws the processed/original textures into the canvas with zoom, pan and a split line. */
export const DISPLAY_FRAG = /* glsl */ `#version 300 es
precision highp float;
out vec4 outColor;
uniform sampler2D uProcessed, uOriginal, uMask;
uniform vec2 uCanvas;   // canvas size in device pixels
uniform vec2 uImage;    // proxy size in pixels
uniform float uScale;   // device pixels per image pixel
uniform vec2 uOffset;   // device-pixel position of the image's top-left corner
uniform int uMode;      // 0 edited, 1 original, 2 split, 3 side by side
uniform float uSplit;   // split position in device pixels (mode 2)
uniform bool uShowMask;
uniform vec3 uBg;
uniform float uDpr;

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
  if (uShowMask) {
    vec4 m = texture(uMask, ip / uImage);
    c = mix(c, vec3(1.0, 0.25, 0.35), m.r * 0.45);
    c = mix(c, vec3(0.2, 0.9, 0.4), m.g * 0.6);
    c = mix(c, vec3(0.3, 0.5, 1.0), m.b * 0.6);
    c = mix(c, vec3(1.0, 0.9, 0.2), m.a * 0.6);
  }
  outColor = vec4(c, 1.0);
}`;
