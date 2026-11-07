import { describe, expect, it } from 'vitest';
import {
  DEFAULT_ADJUSTMENTS,
  decodeLut,
  encodeLut,
  identityTable,
  isDefault,
  lookFromCube,
  lutDataLength,
  lutToCube,
  parseCube,
  sampleLut,
  sanitizeAdjustments,
  paletteSyncAdjustments,
  smartSyncAdjustments,
} from './index';

describe('look LUTs', () => {
  it('round-trips through base64 and .cube', () => {
    const t = identityTable(5);
    t[7] = 0.25;
    const data = encodeLut(t);
    expect(data.length).toBe(lutDataLength(5));
    const back = decodeLut({ data, size: 5 });
    back.forEach((v, i) => expect(Math.abs(v - t[i])).toBeLessThan(1 / 65535 + 1e-7));
    const cube = parseCube(lutToCube(t, 5, 'Warm "look"'));
    expect(cube.title).toBe("Warm 'look'");
    cube.table.forEach((v, i) => expect(Math.abs(v - t[i])).toBeLessThan(2e-6));
  });

  it('samples an identity LUT exactly and honours DOMAIN_MAX', () => {
    const t = identityTable(9);
    const s = sampleLut(t, 9, [0.13, 0.5, 0.97]);
    expect(s[0]).toBeCloseTo(0.13, 5);
    expect(s[2]).toBeCloseTo(0.97, 5);
    const lines = ['LUT_3D_SIZE 2', 'DOMAIN_MAX 2 2 2'];
    for (let b = 0; b < 2; b++) for (let g = 0; g < 2; g++) for (let r = 0; r < 2; r++) lines.push(`${2 * r} ${2 * g} ${2 * b}`);
    expect(Array.from(parseCube(lines.join('\n')).table)).toEqual(Array.from(identityTable(2)));
  });

  it('rejects broken files', () => {
    expect(() => parseCube('LUT_1D_SIZE 4\n0 0 0')).toThrow(/1D/);
    expect(() => parseCube('LUT_3D_SIZE 2\n0 0 0')).toThrow(/Expected 8/);
    expect(() => parseCube('LUT_3D_SIZE 64\n')).toThrow(/outside/);
  });

  it('sanitises looks, clothing and the new sliders inside adjustments', () => {
    const look = lookFromCube(lutToCube(identityTable(3), 3, 'Film'), 70);
    const a = sanitizeAdjustments({
      lut: look,
      glareReduction: 300,
      teethStainRemoval: -4,
      clothing: { enabled: true, wrinkleStrength: 200, hueShift: -500 },
    });
    expect(a.lut).toEqual({ name: 'Film', size: 3, data: look.data, amount: 70 });
    expect(a.glareReduction).toBe(100);
    expect(a.teethStainRemoval).toBe(0);
    expect(a.clothing).toEqual({ enabled: true, wrinkleStrength: 100, texturePreservation: 80, hueShift: -180, saturation: 0 });
    expect(isDefault(a)).toBe(false);
    expect(sanitizeAdjustments({ lut: { ...look, size: 4 } }).lut).toBeNull(); // data length does not match
    expect(sanitizeAdjustments({ lut: { ...look, data: '!!!' } }).lut).toBeNull();
    expect(isDefault(sanitizeAdjustments({}))).toBe(true);
  });
});

describe('batch smart sync', () => {
  it('matches the Python port (services/ai/tests/test_batch_engine.py)', () => {
    const hero = { ...DEFAULT_ADJUSTMENTS, exposure: 10, temperature: 5, tint: -3, skinSmoothing: 40 };
    const st = (L: number, a: number, b: number) => ({ mean: [L, a, b] as [number, number, number], std: [20, 5, 5] as [number, number, number] });
    const o = smartSyncAdjustments(hero, st(55, 4, 12), st(48, 6, 6));
    expect([o.exposure, o.temperature, o.tint]).toEqual([32, 55, -53]);
    const o2 = smartSyncAdjustments(hero, st(55, 4, 12), st(54, 4.5, 11));
    expect([o2.exposure, o2.temperature, o2.tint]).toEqual([13, 17, -15]);
  });
});

describe('batch palette sync', () => {
  it('keeps the retouch, drops tone and colour sliders, and matches the graded reference', () => {
    const graded = { mean: [60, 5, 14] as [number, number, number], std: [22, 6, 7] as [number, number, number] };
    const ref = { ...DEFAULT_ADJUSTMENTS, exposure: 12, saturation: 20, skinSmoothing: 50, teethWhiten: 30 };
    const out = paletteSyncAdjustments(ref, graded, 80);
    expect([out.exposure, out.saturation, out.skinSmoothing, out.teethWhiten]).toEqual([0, 0, 50, 30]);
    expect(out.colorMatch).toEqual({ reference: graded, amount: 80 });
  });
});
