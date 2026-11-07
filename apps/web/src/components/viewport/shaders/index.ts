/**
 * GLSL for the pro studio previews. Each feature lives in its own module as a GLSL
 * snippet plus the notes on how it relates to the full-resolution engine stage;
 * studio.ts assembles them into the STUDIO and OVERLAY fragment shaders the renderer runs.
 */
export { GLARE_GLSL, GLARE_PREVIEW_MAX_VEIL } from './glareReduction';
export { IRON_GLSL } from './clothingIron';
export { LUT_GLSL } from './lut3d';
export { TEXTURE_RETENTION_GLSL } from './textureRetention';
export { OVERLAY_FRAG, OVERLAY_MODES, STUDIO_FRAG, type StudioOverlay } from './studio';
