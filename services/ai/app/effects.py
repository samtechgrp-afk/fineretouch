"""Per-effect retouch controls: enable, 1x/2x/3x multiplier, opacity and global strength.

Mirror of packages/shared/src/effects.ts (see that file for the curve). Both are pinned to the
same vectors by tests. Renderers call `resolve_adjustments` first and render the effective
values; the user's slider values in the edit are never rewritten.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import Adjustments, EffectControl


@dataclass(frozen=True)
class EffectCurve:
    knee: float
    cap: float


# Keep in sync with EFFECT_CURVES in packages/shared/src/effects.ts.
EFFECT_CURVES: dict[str, EffectCurve] = {
    "skinSmoothing": EffectCurve(0.8, 100),
    "blemishRemoval": EffectCurve(0.85, 100),
    "acneRemoval": EffectCurve(0.85, 100),
    "darkSpotRemoval": EffectCurve(0.75, 100),
    "skinToneUnify": EffectCurve(0.75, 100),
    "darkCircles": EffectCurve(0.75, 100),
    "eyeBrighten": EffectCurve(0.6, 85),
    "teethWhiten": EffectCurve(0.6, 85),
    "teethStainRemoval": EffectCurve(0.8, 100),
    "glareReduction": EffectCurve(0.85, 100),
    "sharpen": EffectCurve(0.6, 90),
    "strayHair": EffectCurve(0.8, 100),
    "clothingIron": EffectCurve(0.8, 100),
}

# Effect id -> (field,) or (structured field, sub-field).
EFFECT_SOURCE: dict[str, tuple[str, ...]] = {
    "skinSmoothing": ("skinSmoothing",),
    "blemishRemoval": ("blemishRemoval",),
    "acneRemoval": ("acneRemoval",),
    "darkSpotRemoval": ("darkSpotRemoval",),
    "skinToneUnify": ("skinToneUnify",),
    "darkCircles": ("darkCircles",),
    "eyeBrighten": ("eyeBrighten",),
    "teethWhiten": ("teethWhiten",),
    "teethStainRemoval": ("teethStainRemoval",),
    "glareReduction": ("glareReduction",),
    "sharpen": ("sharpen",),
    "strayHair": ("strayHairRemoval", "strength"),
    "clothingIron": ("clothing", "wrinkleStrength"),
}


def boost_strength(value: float, multiplier: float, knee: float, cap: float) -> float:
    """Slider x multiplier through a soft-knee curve capped at `cap`; 1x returns the slider."""
    s = min(100.0, max(0.0, float(value)))
    if multiplier <= 1 or s == 0:
        return s
    x = s * multiplier
    k = knee * cap
    room = cap - k
    if x <= k or room <= 0:
        soft = min(x, cap)
    else:
        soft = k + room * (1.0 - math.exp(-(x - k) / room))
    return max(s, soft)


def effective_strength(value: float, control: EffectControl, curve: EffectCurve, global_strength: float) -> float:
    if not control.enabled:
        return 0.0
    return boost_strength(value, control.multiplier, curve.knee, curve.cap) * control.opacity * global_strength / 10000.0


def detector_multiplier(adj: Adjustments, effect: str) -> int:
    """Detection reach of a spot tool: max(skin boost, that tool's multiplier when it is on)."""
    c = getattr(adj.effects, effect)
    return max(int(adj.skinMultiplier), c.multiplier if c.enabled else 1)


def resolve_adjustments(adj: Adjustments) -> tuple[Adjustments, int]:
    """Return (adjustments with effective effect values, spot detector multiplier).

    Per-tool detector multipliers: detector_multiplier(adj, 'acneRemoval' | 'darkSpotRemoval');
    call it on the *original* adjustments (the resolved copy has neutral controls).
    """
    g = min(100.0, max(0.0, float(adj.globalStrength)))
    out = adj.model_copy(deep=True)
    for eid, src in EFFECT_SOURCE.items():
        control: EffectControl = getattr(adj.effects, eid)
        curve = EFFECT_CURVES[eid]
        if len(src) == 1:
            setattr(out, src[0], effective_strength(getattr(adj, src[0]), control, curve, g))
        else:
            holder = getattr(out, src[0])
            setattr(holder, src[1], effective_strength(getattr(holder, src[1]), control, curve, g))
    spot_multiplier = detector_multiplier(adj, "blemishRemoval")
    out.effects = type(adj.effects)()
    out.globalStrength = 100.0
    return out, spot_multiplier
