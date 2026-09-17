# The GPU 3D view: implementation report

This report covers the solid 3D renderer that replaces the Canvas 2D wireframe in the 3D model view (branch `3d-solid-renderer`, from `main` at `bf06b13`). It follows §76 of the implementation specification. The numbers come from the benchmark and test suites in this branch, run on the development machine: Windows 11, an NVIDIA RTX 5070, and Chrome 153.

A second round on the same branch added detector hit maps, light-source glyphs, saved view settings, and a ray width setting. [Follow-up](#follow-up-hit-maps-source-glyphs-saved-settings-ray-width) describes them. The last commit removed the old wireframe view, so the "legacy" column below is a historical measurement. The component view, beam splitters, and dichroic filters, which add two display roles to this renderer, are in [component-view.md](component-view.md).

Three.js materials give physically plausible display shading, not optical simulation. Nothing in this renderer computes an optical quantity, and nothing it draws is a measurement. Every displayed number still comes from the f64 reference tracer.

## Baseline (Phase 0)

Recorded on `bf06b13` before any change:

| Check | Result |
|---|---|
| `cargo test --workspace --exclude oc-py` | 558 passed, 7 failed, 3 ignored |
| `cargo test -p oc-core --features scene3` | 228 passed |
| `cargo test -p oc-trace --features trace3` | 85 passed, 1 ignored |
| Browser tests (`npm run test:e2e`) | 41 passed |
| Main JS bundle | 127,091 B (38,894 B gzip) |
| wasm | 2,199,249 B |

All 7 failures are in `oc-gpu`'s `gpu3_vs_cpu3`. On this machine DirectX 12 rejects the 3D compute pipeline ("3D GPU pipeline rejected: Validation Error"), the shader bug already known on this RTX 5070. CI runs on Linux without a GPU, where those tests do not reach the pipeline. This work does not touch `oc-gpu`.

The frame times of the old wireframe view are the "legacy" column in the performance table, and [before-wireframe.jpg](3d-renderer/before-wireframe.jpg) shows its picture.

## Architecture

### Modules

Browser (`web/src/render3d/`):

| Module | Imports three.js | Role |
|---|---|---|
| `view3d.ts` | no | The 3D view: camera gestures, picking, hover, selection, cards, keyboard verbs, the fast/exact ray refresh, frame pacing, debug hooks |
| `camera.ts` | no | Orbit camera state and projection math, shared by the GPU cameras and the 2D overlay |
| `quality.ts` | no | Interactive and settled quality states, pixel-ratio policy |
| `overlays.ts` | no | Axis labels, selection dot, TIR rings, notices on the 2D overlay canvas |
| `cards3d.ts` | no | The Scene3 inspector-card adapter over `cards.ts` |
| `types.ts` | no | The render contract (descriptors, face groups, stats) |
| `renderer.ts` | yes | `OpticalRenderer3D`: backend selection, cameras, passes, stats |
| `scene-cache.ts` | yes | One render proxy per Rust object, keyed by revision |
| `mesh-adapter.ts` | yes | Rust meshes to `BufferGeometry` |
| `materials.ts` | yes | The shading vocabulary |
| `environments.ts` | yes | Studio presets, PMREM cache, backgrounds |
| `rays.ts` | yes | The ray buffer (lines, or strips when wider than 1 px) and the light-background ink pass |
| `hitmaps.ts` | yes | Detector hit maps from the exact trace's cells |
| `cues.ts` | yes | Hover and selection cues |
| `scaffold.ts` | yes | Axes, grid, ground |

`web/src/heat.ts` holds the flux colour ramp that the detector dock and the hit maps share.

`main.ts` creates the view. The old wireframe (`viewport3d.ts`, reachable with `?renderer=legacy` while the GPU view was new) has been removed. `?render3d=webgl` forces the WebGL2 backend, and `?render3d=off` simulates a browser with no usable renderer. The GPU canvas (`#view3d`) sits under `#view`, which becomes a transparent overlay and still receives the pointer, so the page's DOM ids and event wiring are unchanged.

### Rust and wasm changes

- `oc_core::display_mesh` (feature `scene3`) tessellates `Scene3` objects for display. See Geometry.
- `oc_wasm::render_view` adds two `Canvas` methods:
  - `renderObjectsJSON()` returns one descriptor per body, surface, aperture, detector and source: kind, three revisions, visibility, position, quaternion, shading role, front and back roles, display `n_d`, a picking radius, and (for a source) its light's display colour.
  - `renderMesh(id)` returns a `RenderMesh` with `topology()`, `positions()`, `normals()`, `indices()`, `groups()`, `edges()`, `bounds()` and `note()` as typed arrays.
- The follow-up adds two more:
  - `lastDetectorMapsF32()` returns the detectors' flux cells from the most recent `traceProjected3D()`.
  - `sourceObjectsJSON()` lists the sources for the 3D view's object list.

None of these methods changes the scene, the tracer, or the `.ocanvas` format.

### Revisions and the render cache

Each descriptor carries separate geometry, transform and material revisions. A fingerprint decides each one: FNV-1a over the `Debug` text of exactly the fields that affect that aspect. A revision therefore moves only when its content changes, whichever setter changed it. The numbers come from one counter per tracker, so an object that reuses a deleted object's id never matches a stale cache entry.

The cache (`scene-cache.ts`) compares descriptors with what it holds:

| Change | Work done |
|---|---|
| New object | Fetch the mesh, build geometry, create the proxy |
| Removed object | Remove the proxy, dispose its geometries |
| Geometry revision | Fetch a new mesh, replace and dispose the old geometry |
| Transform revision | Set position and quaternion |
| Material revision | Reassign shared materials |
| No change | Nothing |

The view syncs only when the scene revision moves (or on an explicit refresh), never because the camera moved. A drag writes the new position through `setObjectPosition` first. The proxy follows on the next frame from what Rust reports; nothing is moved in three.js first.

### Backend and frames

- **Backend:** `THREE.WebGPURenderer` (three.js 0.186.0, pinned), with three's own WebGL2 fallback when WebGPU is missing or fails.
- **Tone mapping:** Neutral tone mapping and sRGB output. The renderer draws into a half-float buffer and tone-maps once at output, so additive rays accumulate the way they do in the 2D view.
- **Event-driven frames:** a frame is drawn only when something asks for one (a camera change, a scene change, a new ray picture, hover, selection, a setting, a resize, a quality change), at most once per animation frame.
- **No idle loop:** three.js 0.186 starts its own `requestAnimationFrame` loop in `init()`. That loop is stopped, and its per-frame bookkeeping runs inside `render()`. An idle view draws nothing, and a test checks this.
- **Shader preparation:** a lighting change and the first picture compile their shaders with `compileAsync`, and the last frame stays on screen until they finish.
- **Loading:** the renderer is a separate chunk. It is fetched during idle time about 2 s after startup, and the GPU device is created only when the 3D view first opens.
- **Status line:** it names the presentation backend (`render WebGPU`, `render WebGL2 fallback`, `render unavailable`) separately from the ray-preview backend.

### Coordinates and units

- **Units:** 1 three.js unit is 1 mm.
- **World axes:** the Rust scene's, unpermuted: +X is the optical axis, +Y is up, +Z is out of the meridional plane. The camera's up vector is +Y.
- **UI labels:** they keep the Zemax convention: world +X reads "Z", +Y reads "Y", +Z reads "X". The relabelling happens in `overlays.ts` only.
- **Near and far planes:** they follow the framed assembly. The far plane extends at least 5 m beyond the target, because rays run past the hardware.
- **Ray positions:** a ray's 3D position is rebuilt from the projected trace (in-plane position plus signed distance). This assumes the default XY view plane, as the wireframe view did.

## Geometry

### Tessellation

Meshes are built in Rust from the scene's own definitions: the profile curves, and the kernel's `even_asphere_sag` and `even_asphere_dsag`. Each mesh is in its object's local frame.

- **Revolved bodies (lenses):**
  - Each profile curve's upper half (`y >= 0`) is sampled adaptively and revolved about local +X.
  - Every sample lies on the analytic surface.
  - A step is split until the chord error is at most 0.02 mm and neighbouring normals differ by at most 0.105 rad (6°).
  - The number of segments around the axis follows the same two limits. It is clamped to 24–128 and rounded up to a multiple of 4, so the meridional plane is sampled exactly.
  - The default lens (24 mm across) gets 60 segments around the axis. The default scene's frame draws 3,853 triangles in total.
- **Normals:** analytic. Conics use the implicit gradient, which stays finite at any aperture. Aspheres use the kernel's `dsag/dr`. Normals are oriented outward by a containment probe.
- **Extrusions (prisms):** walls from the sampled curves, and caps from the chained outline, ear-clipped.
- **Analytic solids:** sphere, box and cylinder, built from their dimensions. The UI cannot create them yet.
- **Watertight seams:** the rim and the faces meet on identical vertices, computed by the same formula.

### Display-only thickness

Mirrors and plates, apertures and detectors have zero thickness in the model. They are drawn as slabs centred on their plane, with a half-thickness of 3% of their radius, clamped to 0.15–1 mm (at least 0.25 mm for detectors). This thickness exists only in the picture: the tracer still intersects a zero-thickness surface, and nothing is exported.

An aperture draws its opaque region: from the clear opening's boundary to the mask's boundary, plus a central obstruction if the opening is a ring. The boundaries are exact for discs. Rectangles and polygons add their corner angles to the sampling, so corners fall on vertices. A mask with no outer extent is unbounded in the model and is drawn out to 2.5 times the opening radius.

### Feature edges

Each mesh lists its creases as line pairs (`edges()`), taken from how it was built:

- a lens's rim rings (never the pole on the axis)
- a prism's corners and cap outlines
- slab outlines and corners
- box edges and cylinder rims

Duplicates are removed by welded position. The browser uses them for the faint outlines on clear bodies and for the selection outline. The first version derived outlines in JavaScript with `EdgesGeometry`, which cost 4.7 ms per curvature edit; with the Rust edges the whole rebuild takes 0.3 ms.

### What is not drawn as a solid

A 2D-only body (a v1 migration with no authored third dimension) and a profile the kernel refuses to revolve or extrude come back as their outline, drawn as dashed orange lines, with a note. The renderer never invents depth. A legacy line-probe detector is drawn as its line.

### Tests

The first round's `display_mesh` tests (14) check:

- vertices and normals against the analytic faces, including an even asphere
- the chord error along a steep meridian
- watertightness (every welded edge shared by two triangles), winding, and signed volume against a numerical integral (within 0.5%)
- bounds
- exact aperture radii and rectangle corners
- that a move does not change the mesh
- the crease counts
- lenses covering meniscus, plano, steep, biconcave, conic and near-zero cases

The `render_view` tests (3) check that each revision moves only with its own content, that a reused id cannot match an old cache, and that every object kind has a descriptor and a mesh.

## Materials

| Scene role | Display material |
|---|---|
| Glass body (`n_d` known) | `MeshPhysicalNodeMaterial`: transmission 1, roughness 0.04, IOR = `n_d` clamped to [1, 2.333], no dispersion, faint grazing-angle edge light, no depth write |
| Lens rim (a line in the profile) | Pale frosted band, 70% opaque |
| Clear or refracting plate | Transmissive plate at IOR 1.5 |
| Mirror | Metal, roughness 0.2 |
| Aperture | Blackened: dark, rough, slightly metallic so the edge catches light |
| Absorber | Dark, fully rough |
| Detector faces | Dark teal, mostly specular, light clearcoat; both faces, because the tracer records a crossing from either side |
| Detector edge | Dark semigloss housing |
| Detector face with a hit map (follow-up) | The detector finish, with lit cells in the dock's heat colours |
| Source emitter (follow-up) | Glows in its light's colour |
| Source housing (follow-up) | Dark satin metal |
| Slab edge | Neutral grey |
| No solid (outline) | Dashed orange lines |

Materials are shared: one per role, and one per display index for glass.

Glass limitations, also stated in `materials.ts`:

- **Refraction:** three.js transmission refracts the opaque scene behind the glass using a display index. It is not a ray trace through the system, it ignores other glass behind it, and it must not be read for focus, magnification, or any measurement.
- **Index clamping:** an `n_d` outside [1, 2.333] is clamped for shading only. The optical material is untouched, and the clamped value is never shown.
- **No display dispersion or coating colours:** the only colour separation on screen is the tracer's, in the rays.
- **Display aids:** the frosted rim, the edge light, and the crease lines show where a clear body ends. None of them models a surface finish.

## Lighting

Four procedural studios, each a box room with soft emissive panels, prefiltered once into a PMREM and cached:

| Preset | Look | Default background |
|---|---|---|
| Dark studio (default) | Dim room, soft panels, restrained highlights | `#07090e` |
| Light studio | Bright soft room | `#d9dbde` |
| Optical bench | Dark room, overhead strip, key light, visible ground | `#141619` |
| Inspection | Narrow bright strips on dark walls, rim light; shows curvature and tilt | `#6f7378` |

- **Background:** chosen independently of the lighting: match lighting, black, dark charcoal, medium grey, or light grey. The grid, crease, cue, axis and ray styles follow the background actually shown.
- **Exposure:** −2 to +2 stops.
- **Ground:** the ground plane fades out, so it never ends in a hard line.
- **PMREM:** each preset is built at most once per session (the benchmark records 4 builds across 12 switches, one per preset). No image files are loaded.
- **Scope:** these lights are display lighting and have no relation to the optical source.

Shadows are not implemented. Perspective, occlusion, smooth shading, reflections, the ground and the grid gave enough depth cues in the screenshots, so a shadow map was not worth its cost in this milestone.

## Rays

- **Data:**
  - The segments are the tracer's, unchanged: the f32 preview during a drag (CPU mirror or WebGPU, chosen by `fastpreview.ts`), the exact f64 trace otherwise.
  - Nothing is retraced for the renderer.
  - A segment dimmer than 1e-5 of the brightest is not uploaded.
- **Buffer:** one interleaved `Float32Array` with 8 floats per vertex: position, linear wavelength colour, relative flux, and ghost flag. It is re-uploaded only when the traced picture changes, and grows by 1.5× when it runs out.
- **Draw calls:** one `LineSegments` draw for all rays. On light backgrounds there is one more pass (the rays into a half-float target) and one fullscreen quad. The number of scene objects does not grow with the segment count.
- **Line technique:** 1-pixel lines with 4× MSAA by default. At the capped pixel ratios they read like the 2D view's lines. The follow-up added the spec's Phase B strips as a user setting for wider rays (see Follow-up).
- **Dark backgrounds:** additive, as in the 2D view. Overlapping wavelengths sum toward white, dispersion shows as colour, and ghosts stay dim. Ghost gain (the 2D control ×8) is a shader uniform, so changing it uploads nothing.
- **Light backgrounds:** additive light cannot show on a light ground. The rays are summed additively in their own pass, and the sum is laid over the scene as ink:
  - opacity follows the summed brightness, compressed as in the 2D view
  - colour follows the summed hue
  - an unsaturated sum (white light) inks dark grey
  This keeps white neutral and dispersion coloured regardless of draw order. An earlier version that inked each segment separately showed white bundles in the colour of the last wavelength drawn, which was wrong.
- **Visibility:** "through components" (no depth test; the default) or "occluded" (depth-tested against opaque parts). Glass never hides rays in either mode, because it does not write depth. The ink pass draws opaque parts depth-only in occluded mode.
- **TIR markers:** they stay on the overlay, projected from the segment ends.
- **Ray gain and ghost gain:** display parameters, not traced flux.

## UI changes

- A bottom bar, shown only in the 3D view, with lighting, background, exposure, ray visibility, and ground.
- The status line gains `· render <backend>` in the 3D view.
- The HUD stops short of the view toolbar, so a long status line wraps instead of running under the toolbar.
- Hover changes the cursor.
- Selection adds a faint glow and a cyan outline and keeps the object's material. The cyan centre dot remains the drag handle.
- View settings are display state. In the first round they were not saved. The follow-up remembers them in the browser. They are never written to `.ocanvas`.

## Follow-up: hit maps, source glyphs, saved settings, ray width

### Detector hit maps

Each area detector's face shows its incident-flux cells, the same cells and colours as the detector dock's heatmap ([hit-map.jpg](3d-renderer/hit-map.jpg)).

- **Source of the cells:** `traceProjected3D()` already runs the f64 trace for the rays, and the same trace fills every detector's cells. The call keeps them, and `lastDetectorMapsF32()` hands them over. The map therefore costs no second trace. A Rust test checks that the cells equal `detectorResultJSON().incident`.
- **Drawing:** one `nu × nv` texture per detector, one texel per cell, nearest filtering, so cells stay cells. The face is the detector's local XY plane, so the shader reads a fragment's cell straight from its local position. No UV channel was added to the mesh.
- **Scale and colour:** each detector is scaled to its own peak, as the dock is. Empty cells show the plain face. `web/src/heat.ts` holds the ramp, shared with the dock.
- **Staleness:** a map belongs to one exact trace. While a drag shows the interactive preview, or once the scene has changed since that trace, the maps are hidden until the next exact trace.
- **Material:** the per-detector material replaces only the active-face slot, through a slot override in the render cache that survives restyles. A detector's first map compiles its shader before the frame that shows it.
- **Toggle:** "hit maps" in the bottom bar, on by default.

The picture is a view of the readout, not a number: a sharp focus lights one or two cells, as the dock shows it.

### Light-source glyphs

The source used to be invisible, so rays began in empty space. Each source now has a glyph ([source-glyph.jpg](3d-renderer/source-glyph.jpg)):

- **Emitter:** a thin plate the size and shape of the emitting aperture (rectangle or disc). Its front face lies exactly in the emission plane, so no drawn ray starts inside it. A point source is a 1 mm ball.
- **Housing:** a box or cylinder behind the aperture, a little wider than it (a stem for a point source). It has no size in the scene; the tracer does not see it.
- **Colour:** the emitter glows in its light's colour, the weighted sum of its lines' ray colours. A white beam is white, and the fold demo's orange line is orange.
- **Legacy sources:** a legacy line source has no area and is drawn as its line, with a note.
- **Editing:** sources are selectable, draggable and turnable in the 3D view, through the setters that already took any object id. Their card shows position and tilt. `sourceObjectsJSON()` lists them for this view only; the 2D cross-section draws no glyph, so its object list is unchanged.
- **Revisions:** geometry follows the spatial and angular distribution, material follows the spectrum.
- **Tests:** a Rust test checks that the glyph is closed, lies behind the emission plane for both directions, and has its emitter in the plane at the aperture's size. A browser test checks that the exact rays start on the moved glyph's face.

### Saved view settings

Lighting, background, exposure, ray visibility, ray width, hit maps, ground and projection are kept in `localStorage` under `optiland-canvas:view3d`:

- A reload restores them, including the perspective button's state.
- Each field is checked on load. An unknown or out-of-range value falls back to its default, and unreadable storage (a private window, damaged JSON) gives the defaults.
- They are browser preferences, not part of the scene or its file.

### Ray width

A "width" slider sets the ray line width from 1 to 6 px (CSS pixels) ([wide-rays.jpg](3d-renderer/wide-rays.jpg)):

- **At 1 px:** the GPU lines as before.
- **Wider:** one instanced screen-facing quad per segment. Each quad is placed in clip space around its segment, as three.js's `Line2` does, and cut at the near plane in perspective. It has a one-pixel soft edge.
- **Draw calls:** one, the same as for lines. The material draws both faces in a single pass.
- **Brightness:** additive and ink rendering, ghost gain and occlusion work as for lines. Brightness per pixel scales by 1/√width, so dense bundles stay readable.
- **Units:** the width is in CSS pixels, so it holds while a drag lowers the pixel ratio.
- **Performance:** on the 4-lens ghost scene (118,936 segments), GPU time per frame was 1.6 ms at 1 px, 2.2 ms at 2 px and 3.3 ms at 4 px.

## Performance

`OC_BENCH=1 npx playwright test --project=bench` (`web/tests/render3d-bench.spec.ts`) writes `test-results/render3d-bench.{json,md}`. The run below used Chrome 153 on the WebGPU backend, a 1600 × 900 viewport, and pixel ratio 1.

These tables were recorded before the follow-up. It adds one object per scene (the source glyph: two more draw calls and a few hundred triangles) and one texture per detector; the wide-ray costs are under Follow-up.

For the GPU view:

- **CPU frame:** everything on the main thread (a stale trace, wasm sync, ray upload, render submission, overlay), measured over 60 orbiting frames.
- **GPU frame:** from timestamp queries, which Chrome quantizes, so values below 0.1 ms are coarse.
- **Legacy frame:** the removed wireframe view with a static camera, which included rasterizing the rays in JavaScript. The benchmark no longer measures it.
- **Exact trace:** one f64 retrace of the scene. It is the tracer, not the renderer.
- **Ray upload:** the upload that follows that retrace.

| Scene | Segments | Triangles | Draws | CPU frame median / p95 (ms) | GPU frame median / p95 (ms) | Legacy frame median (ms) | Exact trace (ms) | Ray upload (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default singlet | 567 | 3853 | 12 | 0.60 / 0.90 | 0.04 / 0.04 | 2.80 | 0.90 | 0.10 |
| 2 lenses | 945 | 5773 | 16 | 0.50 / 0.70 | 0.04 / 0.05 | 5.10 | 2.20 | 0.10 |
| 3 lenses | 1323 | 7693 | 20 | 0.50 / 0.70 | 0.04 / 0.05 | 5.50 | 4.40 | 0.10 |
| 5 lenses | 2079 | 11533 | 28 | 0.50 / 0.80 | 0.04 / 0.05 | 4.00 | 10.10 | 0.20 |
| 7 lenses | 2835 | 15373 | 36 | 0.60 / 0.90 | 0.05 / 0.06 | 6.00 | 18.40 | 0.10 |
| 10 solids, no rays | 0 | 21133 | 48 | 0.50 / 0.70 | 0.03 / 0.04 | 0.80 | 0.00 | 0.00 |
| 2 lenses + ghosts | 13131 | 5773 | 16 | 1.00 / 1.30 | 0.28 / 0.35 | 39.80 | 54.90 | 0.50 |
| 3 lenses + ghosts | 47592 | 7693 | 20 | 1.10 / 1.40 | 0.77 / 0.90 | 119.30 | 417.70 | 2.50 |
| 4 lenses + ghosts | 118936 | 9613 | 24 | 2.40 / 3.10 | 1.50 / 1.95 | 275.40 | 1458.80 | 5.20 |
| 7 lenses + ghosts (very heavy) | 425593 | 15373 | 36 | 6.10 / 10.40 | 3.15 / 3.67 | 1099.60 | 9158.70 | 18.30 |

The ghost scenes bracket the spec's "~30k" and "~100k" cases: 47,592 and 118,936 segments are drawn.

**Against the targets:**

- **Typical scenes (1–5 components, target ≤ 3 ms median and ≤ 6 ms p95):** CPU frames at most 0.6 ms median and 0.9 ms p95, GPU frames at most 0.05 ms median.
- **~30k segments (target ≤ 5 ms):** met, and the 118,936-segment case is 1.5 ms of GPU and 2.4 ms of CPU.
- **425,593 segments:** 3.2 ms of GPU and 6.1 ms of CPU (p95 10.4). The view stays responsive at that size; the exact trace behind it takes 9.2 s.

**Drag:**

- **Frame work:** a pointer drag of a lens in a 3-lens scene takes 0.8 ms per frame (median; p95 1.1), with 1.4 ms of fast-path trace per step, at pixel ratio 1.
- **Mesh rebuilds:** none (the same result as the render-cache test).
- **Settling:** a drag ends with one exact retrace.

**Curvature edits:** 30 edits caused 30 mesh rebuilds, at 0.3 ms each (p95 0.4). The wasm sync for such a frame is 0.3 ms (p95 0.5).

**Lighting and projection:**

- **First use of a preset:** 58–423 ms before the new picture appears. This is PMREM generation plus asynchronous shader compilation, and the previous frame stays on screen meanwhile. No main-thread task longer than 50 ms occurred during any of the switches, so the page stays responsive through them.
- **Cached switches:** a 1.0 ms frame.
- **Perspective and orthographic toggles:** a 0.6 ms frame.
- **WebGL2:** the first switch on that backend took about 540 ms in development, most of it shader compilation.

**Memory (stress run):**

| Stage | Geometries | Textures | JS heap (MB) | Cached objects |
|---|---:|---:|---:|---:|
| start | 19 | 8 | 44.3 | 2 |
| after 100 add/delete cycles | 19 | 8 | 44.6 | 2 |
| after 100 curvature changes | 19 | 8 | 57.6 | 2 |
| after 50 lighting switches | 21 | 16 | 66.2 | 2 |
| after 20 2D/3D switches | 21 | 16 | 56.1 | 2 |

GPU resources return to the same counts after adds, deletes and curvature edits. The two extra geometries and eight extra textures appear once, when the three presets other than the startup one are first built (the optical bench also shows the ground plane), and the counts stay flat across the 50 switches. The JS heap moves with garbage collection and shows no trend.

**Bundle:**

| Asset | Before | After | After the follow-up |
|---|---:|---:|---:|
| Main JS | 127,091 B (38,894 gzip) | 149,824 B (45,964 gzip) | 153,020 B (46,924 gzip) |
| Renderer chunk (lazy; three.js and render3d) | none | 911,568 B (249,571 gzip) | 918,654 B (251,273 gzip) |
| wasm | 2,199,249 B | 2,310,891 B | 2,322,541 B |

The three.js WebGPU build does not tree-shake well; almost all of the lazy chunk is three.js. The addons that import the classic `three` build (`RoomEnvironment`, `BufferGeometryUtils`) were deliberately avoided, because they would add the WebGL renderer to the chunk.

**Where stalls come from:** the exact f64 trace runs on the main thread, for example 1.5 s for the 4-lens ghost scene. That is unchanged by this work and is the dominant stall in heavy scenes. The renderer's own contributions for that scene are small next to it: mesh builds at 0.3 ms, wasm sync at 0.3 ms, a 5.2 ms ray upload for 119k segments, and a 2.4 ms frame.

## Validation

- **Rust:** `display_mesh` (15 tests with the source glyph), `render_view` (4) and two `Canvas` tests for the detector maps and the source list. With them:
  - `cargo test --workspace --exclude oc-py`: 579 passed, 7 failed, 3 ignored. The failures are the same `gpu3_vs_cpu3` tests as in the baseline.
  - `oc-core --features scene3`: 243 passed.
  - `oc-trace --features trace3`: 85 passed, 1 ignored.
  - Clippy with `-D warnings` is clean on all four CI steps, `oc-py` included.
  - These suites cover the spec's regression requirement for the f64 trace, detector results, OPL, the energy ledger, `.ocanvas`, and the Rust half of the Optiland round trip, and they pass unchanged. This work changes none of that code.
- **Python bridge:** the CI job (`pip install '.[test]'`, then `pytest` against a live Optiland) was run locally in a Python 3.12 environment: 38 passed.
- **Browser, CI project:**
  - `tests/render3d.spec.ts` (18 tests): opening and backend naming; orbit, pan, zoom and reset; presets, perspective and the `0` key; picking and dragging without a mesh rebuild; the card and the 2D/3D selection hand-off; copy, paste, delete and nudge; lighting, background, exposure and ray visibility leaving the saved scene, the exact trace and the detector readout bit-identical; mesh rebuilds only on geometry edits; the ray buffer (every segment, bounded draws, no re-upload for ghost opacity, depth mode changing depth only); quality states; no idle frames; no resource growth; the forced WebGL2 backend; and the no-renderer path with the 2D view still working. A test that the legacy view was still reachable went with that view.
  - The follow-up adds four: the source glyph (picked, dragged, rays starting on its face, turned from its card); hit maps (drawn, hidden during a drag, toggled without touching the optics); wide rays (same segments, same draw count, more pixels); and saved settings (restored after a reload, damaged storage ignored field by field).
  - The existing 3D view-controls test now reads the camera and the GPU picture through the `?debug` hooks.
  - The full suite (59 tests) passes. Headless CI has no WebGPU device, so this suite runs there on the WebGL2 fallback; the same suite passes locally on Playwright's bundled Chromium, which also lands on WebGL2.
- **Browser, opt-in:**
  - `OC_WEBGPU=1 … --project=webgpu` also runs `render3d.spec.ts` on a real device. With Chrome 153 on the RTX 5070, its 20 tests (these 18 plus the 2 WebGPU preview tests) pass.
  - `OC_VISUAL=1 … --project=visual` compares 14 screenshots in 12 tests (default scene dark and light, prism, fold mirror, aperture with a second detector, overlapping glass, three lenses with ghosts, selected and hovered lens, occluded rays on the bench in perspective, inspection on grey, wide rays dark and light, a hit map end-on) at 1% pixel tolerance, with the status line masked. The baselines are for this machine (`*-visual-win32.png`), were re-recorded for the follow-up (every scene now shows its source), and are stable across runs.
- **Backends seen:** WebGPU (Chrome 153 on the RTX 5070), WebGL2 (forced, and three.js's fallback in Playwright's bundled Chromium), SwiftShader software GL, and "unavailable" (simulated).

## Limitations

- **Real-time transparency:**
  - Glass shows only the opaque scene through it. A lens behind another lens is drawn by sort order, but it is not refracted in the front lens's picture.
  - Clear bodies do not write depth, so where several overlap their order comes from object sorting and can be wrong at some angles.
  - Order-independent transparency was not attempted.
- **Refraction is not physical:** what the glass bends is a screen-space image at a clamped display index. The rays are the optical picture.
- **Selection glow on glass:** the glow is drawn over the glass. On a clear body it tints the whole silhouette slightly, not only the rim.
- **Light backgrounds:** rays are ink, and the ink's darkness is a display mapping. On a light ground a bright focus reads as dark, not bright.
- **First use of a lighting preset:** it takes up to about 0.4 s on WebGPU and about 0.5 s on WebGL2 before the new picture appears. The page stays responsive meanwhile.
- **Ray lines:** 1 pixel wide by default. On a high-density display at the settled pixel ratio (1.5) they are thinner than CSS pixels; the width setting fixes that.
- **View plane:** rays are placed in 3D from the projected trace, which assumes the default XY view plane. The app has no control that changes the view plane yet.
- **Hit maps:**
  - Each detector is scaled to its own peak, so two detectors' colours are not comparable.
  - Disc and annular detectors are binned over their bounding square, as the tracer bins them, so their outer cells are partly off the face.
  - A map is hidden during a drag and returns with the next exact trace.
- **Source glyphs:**
  - The housing is invented for display.
  - A source aimed backwards along its local X gets its housing on the other side. Any other aim is drawn as if along +X.
  - The 2D cross-section still draws no source.
- **2D-only bodies:** outlines, not solids.
- **Shadows:** none.
- **No fallback view:** with the wireframe view gone, a browser with neither WebGPU nor WebGL2 gets a notice in the 3D view. The 2D cross-section still works there.
- **Timing precision:** GPU times below 0.1 ms are quantized by the browser.
- **CI coverage:** CI exercises only the WebGL2 path (SwiftShader). The WebGPU path is covered by the opt-in project on a machine with a GPU.

## Screenshots

| | |
|---|---|
| Default scene, dark studio | [default-dark-studio.jpg](3d-renderer/default-dark-studio.jpg) |
| Default scene, light studio | [default-light-studio.jpg](3d-renderer/default-light-studio.jpg) |
| Two lenses with ghosts | [two-lenses-ghosts.jpg](3d-renderer/two-lenses-ghosts.jpg) |
| Fold mirror demo | [fold-mirror.jpg](3d-renderer/fold-mirror.jpg) |
| Prism ahead of the lens | [prism.jpg](3d-renderer/prism.jpg) |
| Lens selected, prism hovered | [selected-and-hovered.jpg](3d-renderer/selected-and-hovered.jpg) |
| Rays through components (orthographic) | [rays-through-components.jpg](3d-renderer/rays-through-components.jpg) |
| Rays occluded | [rays-occluded.jpg](3d-renderer/rays-occluded.jpg) |
| Perspective | [perspective.jpg](3d-renderer/perspective.jpg) |
| Inspection lighting on grey | [inspection-grey.jpg](3d-renderer/inspection-grey.jpg) |
| Hit map, end-on, point source | [hit-map.jpg](3d-renderer/hit-map.jpg) |
| Rays at 3 px, three lenses with ghosts | [wide-rays.jpg](3d-renderer/wide-rays.jpg) |
| Disc source glyph, selected | [source-glyph.jpg](3d-renderer/source-glyph.jpg) |
| Before: the Canvas 2D wireframe | [before-wireframe.jpg](3d-renderer/before-wireframe.jpg) |

The images were re-captured after the follow-up, so every scene shows its source. The WebGL2 fallback looks the same as WebGPU in side-by-side captures, so it has no separate image.

## Not done

The follow-up built strip rays, saved preferences, source glyphs and hit maps. Still open:

- Shadows.
- Optional HDRI environments.
- Order-independent transparency for glass behind glass.
- A source glyph in the 2D cross-section.
