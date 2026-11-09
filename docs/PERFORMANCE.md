# Rendering and performance (v0.12, Phase 7)

Target machine: Ryzen 3 7330U (4 cores / 8 threads, integrated Radeon GPU), 16 GB RAM, Windows.
Nothing here needs CUDA or a vendor GPU driver: the GPU path is WebGL2 and the CPU path is
NumPy / OpenCV.

## The two render engines

| | GPU (WebGL2) | CPU engine (`services/ai`) |
| --- | --- | --- |
| Used for | Live preview, fast export | Reference export, 16-bit camera RAW |
| Spot healing, stray hair, bokeh | Preview versions | Full strength |
| Limits | Texture size and GPU memory, now handled by tiles | System RAM, now guarded by the memory gate |

### CPU / GPU selection

In the Export dialog, **Default engine** is a saved setting: **Auto**, **GPU** or **CPU engine**.
The **Processing** buttons next to it still override it for one export.

- **Auto** uses the GPU when the browser has a hardware WebGL2 device.
- **Auto** uses the CPU engine when the only WebGL2 device is a software rasteriser:
  SwiftShader, llvmpipe or Microsoft Basic Render Driver. Software GPU rendering is slower than
  the CPU engine itself.
- The dialog shows the detected GPU, its maximum texture size, and why Auto chose what it did.
- If the CPU engine is not running, every choice falls back to the GPU and says so.

Code: `apps/web/src/lib/processing.ts`, which has `detectGpu`, `resolveEngine` and
`isSoftwareRenderer`.

## High-resolution rendering and tiled processing

### GPU export in tiles

Before this phase, a GPU export larger than the GPU's maximum texture size was scaled down. That
limit is 8192 px on many integrated GPUs and in SwiftShader. A full-frame render also took about
56 bytes of GPU memory per pixel: roughly 13 render targets for the source, the blur bands and
the passes.

Now `planExport` (in `apps/web/src/lib/export.ts`) decides how to render:

- **One pass** when the photo fits both the texture limit and the **GPU memory** budget. The
  budget is set in the Export dialog: Low 512 MB, Balanced 1.5 GB (the default) or High 3 GB.
- **Tiles** otherwise. Each tile renders its part of the photo plus an **apron**, a margin of
  surrounding pixels that gives every effect the context it would have in a single pass.
- The apron (`exportApron`) is the sum of what each stage reaches:
  - 3σ of the widest blur band (3% of the long edge)
  - the blur of every frequency-separation layer
  - the bokeh gather radius
- Inside a tile, every radius is computed from the **whole photo's** long edge, not the tile's.
- Whole-photo masks are sampled at the tile's position through a frame uniform (`uFrame`,
  `fuv()` in the shaders). These are the face mask, strokes, hair and lens regions, layer
  masks, depth and segmentation.
- Only each tile's inner rect is read back and written to the output canvas.
- **Scaled down** only when even tiles cannot fit, for example a photo so large that the apron
  fills the tile. The dialog then says so and suggests the CPU engine.

Measured in the real app on a 4000 x 5000 photo (Chromium with SwiftShader). Each export was
run once in one pass and once tiled with the Low budget, and the two PNGs were compared:

| Edit | Tiles | Difference from one pass |
| --- | --- | --- |
| Skin smoothing, vignette, sharpen, exposure | 6 | Max 1/255 on 0.16% of pixels, mean 0.0005/255 |
| The same plus a frequency-separation layer and background bokeh | 12 | Mean 0.09/255. 99.9% of pixels are within 2/255. The rest are single pixels along the subject's edge in the bokeh transition, where the bokeh downsample grid falls differently. No tile seams |

### CPU engine strips

Stages 7 to 9 of the engine are all per-pixel: colour match, exposure and white balance, tone,
saturation, the HSL mixer and grading, the LUT, sharpening and the vignette. They now run in
256-row strips (`pipeline._finish_strip`). Their temporaries therefore cost one strip of memory
instead of several full-frame float copies.

The coarser blur bands are also built only when a stage reads them.

Measured on a 20 MP photo, render stage only:

| Edit | Before | After |
| --- | --- | --- |
| Plain (default settings) | 14.8 s, 3.03 GB peak | 7.5 s, 1.83 GB peak |
| Skin tools on | 41.6 s, 3.57 GB peak | 38.9 s, 2.75 GB peak |

- Strip results equal a single strip exactly (`tests/test_performance.py`).
- Compared with the previous code on the sample portrait, results are identical or differ by
  1/255 on at most 8 of 840,000 pixels. The working copy now stays float32 throughout.

## Preview caching

The renderer runs four stages: retouch, studio (glare, clothing, LUT), local layers, and bokeh.
Each stage now has a cache key: its settings, everything before it, and a version counter for
its input textures. The counters are source and masks for retouch, lens and garment masks for
studio, layer masks for local, and depth for bokeh.

A frame re-runs only the stage that changed and the stages after it:

- **Pan, zoom, split and compare** re-run nothing. They only redraw the display pass.
- **A bokeh slider** re-runs bokeh only.
- **Adding a layer** re-runs layers and bokeh.
- **A retouch slider** re-runs all four stages.

The viewport readout shows what ran, for example `· 1/4 stages` or `· cached`. These were
confirmed in the app: zoom showed cached, a new frequency-separation layer ran 2/4 stages, and a
bokeh change or the depth map arriving ran 1/4.

The millisecond readout measures the time to submit GPU commands, not GPU execution time, so it
is not a frame-time benchmark.

## Memory management

**Editor previews.** Decoded 2560 px previews take about 20 MB each. They are kept for the 12
most recently used photos (`PROXY_CACHE_SIZE`), together with the pixel caches built from them.
Older ones are freed and decoded again from the original when the photo is opened.

- Engine results (depth, segmentation, face parsing) are small and kept.
- Checked in the app: after importing 15 photos and switching between them, the first photo came
  back with its edits.
- Unit test: `apps/web/src/lib/proxyCache.test.ts`.

**Engine memory gate.** `services/ai/app/memory.py` estimates each full-resolution render as
150 bytes per pixel plus 200 MB, about 3.8 GB for 24 MP. A render is admitted only while the
renders in flight fit the budget.

- The budget is `FINERETOUCH_RENDER_MEMORY_MB`, or 60% of physical memory, which is about 9.6 GB
  on a 16 GB machine.
- One render is always admitted, so a huge photo still renders, one at a time.
- Physical memory is read through psutil when it is installed, otherwise from `/proc/meminfo`,
  `GlobalMemoryStatusEx` on Windows or `sysconf` on macOS.
- The gate covers `/v1/render`, render-object and the desktop batch queue.
- `/health` reports the CPU threads, the budget and the renders in flight.

## Export optimisation

- **One WebGL context for a whole GPU export.** Before, each photo created a new context and
  compiled every shader again. The time saved was not measured.
- **The next photo decodes while the current one renders** (GPU path).
- **Tiles yield to the browser** between each other, so the UI stays responsive.

## Status

| Feature | Status |
| --- | --- |
| High-resolution GPU export past the texture limit | Implemented (tiles) |
| Tiled processing, GPU | Implemented and tested against a single pass |
| Tiled processing, CPU engine | Partial. The per-pixel stages run in strips. The skin stages already run on the skin's bounding box. Blur bands are still full-frame |
| Preview caching | Implemented (per-stage cache) |
| CPU/GPU selection | Implemented (Auto, GPU, CPU with software-GPU detection) |
| Memory management | Implemented: preview cache limit, engine memory gate, GPU memory budget for tiles |
| Export optimisation | Implemented: one GPU context per export, decode overlap |
| Measured timings on the Ryzen 3 7330U | Not done. The numbers above are from the cloud test machine, and GPU timings there use SwiftShader, not a real GPU. A timing run on the target laptop belongs in Phase 8 (hardware and performance testing) |
| Tiling the CPU engine's blur bands for photos above about 60 MP | Planned |
