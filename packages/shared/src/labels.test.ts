import { describe, expect, it } from 'vitest';
import { LABEL_KINDS, isKeepKind, labelAt, labelCounts, labelsFromReport, makeLabelFile, parseLabelFile, type QaLabel } from './labels.js';
import type { PerfectionReport } from './perfection.js';

const L = (o: Partial<QaLabel>): QaLabel => ({ id: 'a', x: 0.5, y: 0.5, r: 0.01, kind: 'acne', source: 'manual', confirmed: true, ...o });

describe('QA label files', () => {
  it('round-trips through JSON and rejects bad files', () => {
    const f = makeLabelFile({ name: 'a.jpg', width: 400, height: 200 }, [L({}), L({ id: 'b', kind: 'mole', source: 'engine', confirmed: false })], 'Sam');
    const back = parseLabelFile(JSON.stringify(f));
    expect(back).toEqual(f);
    expect(() => parseLabelFile('{"format":"x","version":1}')).toThrow(/Not a FineRetouch/);
    expect(() => parseLabelFile(JSON.stringify({ ...f, labels: [{ ...L({}), kind: 'zit' }] }))).toThrow(/unknown kind/);
    expect(() => parseLabelFile(JSON.stringify({ ...f, labels: [L({ x: 2 })] }))).toThrow(/out of range/);
  });

  it('suggests unconfirmed labels from the engine, skipping covered marks and shine', () => {
    const report = { marks: [
      { id: 1, x: 0.5, y: 0.5, r: 0.01, kind: 'acne' },
      { id: 2, x: 0.2, y: 0.2, r: 0.01, kind: 'freckle' },
      { id: 3, x: 0.8, y: 0.8, r: 0.01, kind: 'shine' },
    ] } as unknown as PerfectionReport;
    const add = labelsFromReport(report, [L({})]);
    expect(add.map((l) => [l.kind, l.confirmed, l.source])).toEqual([['freckle', false, 'engine']]);
  });

  it('hits the smallest label and counts kinds', () => {
    const ls = [L({ id: 'big', r: 0.02 }), L({ id: 'small', r: 0.004, kind: 'freckle', confirmed: false })];
    expect(labelAt(ls, 0.5, 0.5, 1.5)?.id).toBe('small');
    expect(labelAt(ls, 0.9, 0.9, 1.5)).toBeNull();
    expect(labelCounts(ls)).toEqual({ remove: 1, keep: 1, unconfirmed: 1 });
    expect(LABEL_KINDS.filter(isKeepKind)).toEqual(['mole', 'freckle', 'birthmark', 'follicle', 'hair']);
  });
});
