"""Advanced retouch stages for the full-resolution renderer (FineRetouch v0.2).

Since v0.9 the code lives in app/engines/, one module per tool family; this module re-exports
all of it so existing imports keep working:

  engines/common.py    colour conversions, blurs, inpainting helpers
  engines/boost.py     apply_boosted_skin_retouch
  engines/spots.py     detect_spots, heal_spots and their thresholds
  engines/skin.py      natural_skin_retouch, acne / dark-spot tools, skin_tone_guard
  engines/hair.py      stray hair      engines/bokeh.py  depth bokeh
  engines/style.py     style sync      engines/studio.py glasses glare, teeth
  engines/clothing.py  virtual iron and recolour


  natural_skin_retouch        texture-preserving frequency separation (v0.5): spots healed, then
                              blotches smoothed and median-evened on the tone layer only, and
                              the pore/detail layer re-injected by the Texture retention factor.
  skin_tone_guard             CIELAB a*/b* lock so retouched skin never turns ashen or off-hue.
  heal_spots                  local acne / pimple / dark-spot healing (v0.5): scale-normalised
                              Laplacian-of-Gaussian detection on a darkness + redness signal
                              inside the skin mask, beauty-mark and feature protection, then
                              per-spot inpainting of the tone layer and healthy-skin patch
                              cloning of the texture layer. 1x / 2x / 3x set how far it reaches.
                              Raised comedones that read lighter than the skin (deeper or
                              glossy skin) are found too; shading hollows, one-sided edges and
                              dyed strands on the skin are rejected.
  apply_boosted_skin_retouch  2x/3x multi-stage frequency separation: recursive guided
                              filtering of the low-frequency colour layer plus adaptive
                              blemish detection and patch-based repair of the detail layer.
  remove_stray_hairs          flyaway hairs around the face outline and forehead:
                              morphological + Gabor line detection, then inpainting.
  apply_depth_bokeh           layered disk / hexagon depth-of-field from a depth map,
                              with highlight bloom and a 100-300% radius multiplier.
  reduce_glasses_glare        eyeglass lens reflections (v0.6): reflection model inverted per pixel
                              on a landmark lens region, blown-out cores inpainted, catchlights kept.
  correct_teeth               CIELAB whitening and stain removal on an enamel mask (v0.6).
  remove_clothing_wrinkles    "virtual iron" (v0.6): crease / fold bands of log-luminance attenuated
                              inside the garment mask, weave and drape kept, colour untouched.
  shift_clothing_color        garment recolour in CIELAB, skin excluded (v0.6).
  style_profile /             reference-photo metrics and the mapping to editable
  extract_and_sync_style      adjustments for a whole batch (port of packages/shared/src/style.ts).

All functions take and return float32 sRGB in [0, 1] unless stated. The WebGL preview
(apps/web/src/engine/shaders.ts) approximates these stages in real time; exports from
this service run the full versions.
"""

from __future__ import annotations

from .engines.common import *  # noqa: F401,F403
from .engines.boost import *  # noqa: F401,F403
from .engines.spots import *  # noqa: F401,F403
from .engines.skin import *  # noqa: F401,F403
from .engines.hair import *  # noqa: F401,F403
from .engines.bokeh import *  # noqa: F401,F403
from .engines.style import *  # noqa: F401,F403
from .engines.studio import *  # noqa: F401,F403
from .engines.clothing import *  # noqa: F401,F403
