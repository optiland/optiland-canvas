// The 3D model view on the GPU renderer: interaction, state, and frame pacing.
//
// This module holds only UI state (camera, selection, hover, clipboard, the
// cached ray picture) and never imports three.js. The renderer arrives through a
// dynamic import the first time the 3D view is shown, so the 2D app starts
// without it.
//
// The scene lives in Rust. A drag computes the intended world position, writes
// it through the Rust setter, and the render proxies follow on the next frame
// from what Rust reports; nothing is moved in three.js first.
//
// Rays follow the app's two-tier rule: the interactive f32 picture while a body
// is dragged (fastpreview.ts picks the CPU mirror or WebGPU), and the exact f64
// trace once the drag ends or after any other edit. No number is computed here.

import type { Canvas } from "../../pkg/oc_wasm.js";
import type { FastPreview } from "../fastpreview";
import type { RayLayer } from "../rays";
import { CameraRig, type Bounds3, type CameraPreset, type CameraState, type V2, type V3 } from "./camera";
import { createScene3Cards, type SceneObject } from "./cards3d";
import { drawAxisLabels, drawNotice, drawSelectionDot, drawTirRings } from "./overlays";
import { QualityController } from "./quality";
import type { OpticalRenderer3D } from "./renderer";
import type { RenderBackend, RenderStats } from "./types";

// The projected trace layout (traceProjected3D): 11 floats per segment,
// [au,av,bu,bv, r,g,b, flux, flags, dist_a, dist_b]. With the default XY view
// plane the in-plane (au,av) are world (x,y) and the signed distance is world z.
const STRIDE = 11;

export type FrameReport = {
  /** Ray segments drawn. */
  count: number;
  /** Whole frame, ms (trace + upload + render submission + overlay). */
  ms: number;
  /** Of which the ray trace, ms. */
  traceMs: number;
};

export type View3dThree = {
  kind: "three";
  /** Ask for a frame. Frames are coalesced to one per animation frame. */
  invalidate: () => void;
  refresh: () => void;
  selectedBody: () => number;
  activate: (selectId: number) => void;
  standDown: () => void;
  focusObject: (id: number) => void;
  resetView: () => void;
  setPreset: (name: string) => void;
  setPerspective: (on: boolean) => void;
  isPerspective: () => boolean;
  /** "render WebGPU" etc., for the status line. */
  backendLabel: () => string;
  /** Debug hooks (the `?debug` build of the page reads these). */
  debug: View3dDebug;
};

export type View3dDebug = {
  ready: () => Promise<RenderBackend>;
  camera: () => CameraState;
  stats: () => RenderStats;
  /** Render now and count lit pixels of the GPU canvas (tests). */
  litPixels: () => Promise<number>;
  project: (p: V3) => V2;
  hovered: () => number;
};

export type View3dOptions = {
  scene: Canvas;
  /** The overlay canvas (#view): 2D annotations, and it receives the pointer. */
  view: HTMLCanvasElement;
  /** The GPU canvas (#view3d), underneath the overlay. */
  canvas: HTMLCanvasElement;
  owns: () => boolean;
  /** Repaint through main (which calls `invalidate` while 3D is active). */
  render: () => void;
  onSelect?: (id: number) => void;
  detectorRefresh?: () => void;
  fast?: FastPreview;
  /** Phase 1 ray drawing: the shared HDR ray layer, composited on the overlay. */
  rays: RayLayer;
  rayOptions: () => { ghostOpacity: number; exposure: number; showTir: boolean };
  grid: () => { on: boolean; spacing: number };
  /** Write the status line for a finished frame. */
  status: (r: FrameReport) => void;
  forceWebGL?: boolean;
  /** Simulate a platform with no usable 3D renderer (tests). */
  disable?: boolean;
};

export function initView3dThree(opts: View3dOptions): View3dThree {
  const { scene, view, canvas } = opts;
  const ctx = view.getContext("2d")!;
  const rig = new CameraRig();
  let renderer: OpticalRenderer3D | null = null;
  let backend: RenderBackend = "loading";
  let failure = "";
  let loading: Promise<RenderBackend> | null = null;
  let active = false;
  let framed = false;
  let pending = 0;

  const quality = new QualityController(() => invalidate());

  // --- scene data (re-read from Rust; never mirrored) --------------------
  let segs: Float32Array<ArrayBufferLike> = new Float32Array(0);
  let tracedRevision = -1;
  let shownExact = false;
  let objList: SceneObject[] = [];
  let selected = -1;
  let hovered = -1;
  let clipboard = -1;
  let pasteN = 0;
  let lastTraceMs = 0;
  // The scene revision the render proxies were last synced at.
  let syncedRevision = -1;

  /** Bring the GPU proxies up to date with Rust, if the scene moved. */
  function syncRender(force = false) {
    if (!renderer) return;
    const rev = scene.sceneRevision();
    if (!force && rev === syncedRevision) return;
    const t0 = performance.now();
    const descriptors = JSON.parse(scene.renderObjectsJSON());
    renderer.syncScene(descriptors, (id) => scene.renderMesh(id), t0);
    syncedRevision = rev;
  }

  function select(id: number) {
    selected = id;
    opts.onSelect?.(id);
  }

  // A fast frame can arrive after `retrace` returns (WebGPU), so it is drawn only
  // when it is newer than the picture on screen, and never over an exact frame of
  // the same scene.
  function retrace(fast = false) {
    const t0 = performance.now();
    if (!fast) {
      opts.fast?.settled();
      segs = scene.traceProjected3D();
      tracedRevision = scene.sceneRevision();
      shownExact = true;
    } else if (!opts.fast) {
      segs = scene.traceProjected3DFast();
      tracedRevision = scene.sceneRevision();
      shownExact = false;
    } else {
      opts.fast.requestProjected(scene, (f) => {
        if (f.revision < tracedRevision || (f.revision === tracedRevision && shownExact)) return;
        segs = f.segs;
        tracedRevision = f.revision;
        shownExact = false;
        if (f.late) invalidate();
      });
    }
    lastTraceMs = performance.now() - t0;
  }

  const objectsPanel = document.getElementById("objects3d");
  function refreshObjects() {
    objList = JSON.parse(scene.sceneObjectsJSON());
    if (!objectsPanel) return;
    objectsPanel.replaceChildren();
    for (const o of objList) {
      const row = document.createElement("div");
      row.className = "row";
      const [x, y, z] = o.pos.map((v) => v.toFixed(1));
      const shape = o.kind === "profile" ? `profile · ${o.construction}` : o.kind;
      const planar = o.is3d ? "" : ` <span class="planar" title="a 2D-only body: not traced in full 3D">⚠ 2D-only</span>`;
      row.innerHTML =
        `<span class="oid">#${o.id}</span>` +
        `<span class="okind">${shape}${planar}</span>` +
        `<span class="opos">(${x}, ${y}, ${z})</span>`;
      objectsPanel.appendChild(row);
    }
  }

  const project = (p: V3) => rig.project(p);
  const cards = createScene3Cards({
    scene,
    objects: () => objList,
    project,
    render: () => invalidate(),
    afterEdit: () => { afterEdit(); opts.detectorRefresh?.(); },
  });

  function afterEdit() {
    retrace(false);
    refreshObjects();
    cards.renderAll();
    invalidate();
  }

  // --- framing -----------------------------------------------------------
  // Frame the hardware, plus the ray points near it, so a steeply dispersed or
  // stray ray running to the scene bounds cannot blow up the box.
  function hardwareBounds(): Bounds3 | null {
    const fromRenderer = renderer?.sceneBounds();
    if (fromRenderer) return fromRenderer;
    if (objList.length === 0) return null;
    const lo = { x: Infinity, y: Infinity, z: Infinity };
    const hi = { x: -Infinity, y: -Infinity, z: -Infinity };
    for (const o of objList) {
      const r = Math.max(o.radius, 1);
      lo.x = Math.min(lo.x, o.pos[0] - r); hi.x = Math.max(hi.x, o.pos[0] + r);
      lo.y = Math.min(lo.y, o.pos[1] - r); hi.y = Math.max(hi.y, o.pos[1] + r);
      lo.z = Math.min(lo.z, o.pos[2] - r); hi.z = Math.max(hi.z, o.pos[2] + r);
    }
    return { min: lo, max: hi };
  }

  // The side panels cover the canvas edges; framing keeps the assembly between them.
  function panelInset(): number {
    let inset = 0;
    for (const id of ["m4dock", "controls", "dock"]) {
      const r = document.getElementById(id)?.getBoundingClientRect();
      if (!r || r.width === 0) continue;
      inset = Math.max(inset, r.left < window.innerWidth / 2 ? r.right : window.innerWidth - r.left);
    }
    return Math.min(inset, window.innerWidth * 0.3);
  }

  function frameScene() {
    rig.sideInset = panelInset();
    const hw = hardwareBounds();
    if (!hw) { rig.frame(null); return; }
    const lo = { ...hw.min }, hi = { ...hw.max };
    const c0 = { x: (lo.x + hi.x) / 2, y: (lo.y + hi.y) / 2, z: (lo.z + hi.z) / 2 };
    const span = Math.max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
    const limit = Math.max(80, span * 2);
    const grow = (x: number, y: number, z: number) => {
      if (Math.abs(x - c0.x) > limit || Math.abs(y - c0.y) > limit || Math.abs(z - c0.z) > limit) return;
      lo.x = Math.min(lo.x, x); hi.x = Math.max(hi.x, x);
      lo.y = Math.min(lo.y, y); hi.y = Math.max(hi.y, y);
      lo.z = Math.min(lo.z, z); hi.z = Math.max(hi.z, z);
    };
    for (let i = 0; i + STRIDE <= segs.length; i += STRIDE) {
      grow(segs[i], segs[i + 1], segs[i + 9]);
      grow(segs[i + 2], segs[i + 3], segs[i + 10]);
    }
    rig.frame({ min: lo, max: hi });
  }

  // --- renderer lifecycle --------------------------------------------------
  function ensureRenderer(): Promise<RenderBackend> {
    if (loading) return loading;
    loading = (async () => {
      if (opts.disable) {
        failure = "disabled for this page";
        backend = "unavailable";
        return backend;
      }
      try {
        const mod = await import("./renderer");
        renderer = await mod.OpticalRenderer3D.create(canvas, { forceWebGL: opts.forceWebGL });
        backend = renderer.backend;
        refreshObjects();
        syncRender(true);
      } catch (e) {
        console.warn("3D renderer unavailable.", e);
        failure = e instanceof Error ? e.message : String(e);
        backend = "unavailable";
      }
      invalidate();
      return backend;
    })();
    return loading;
  }

  function invalidate() {
    if (!active || pending) return;
    pending = requestAnimationFrame(() => {
      pending = 0;
      drawFrame();
    });
  }

  function drawFrame() {
    if (!active) return;
    const t0 = performance.now();
    const W = window.innerWidth, H = window.innerHeight;
    rig.setSize(W, H);
    const dpr = window.devicePixelRatio || 1;

    // A stale picture is retraced exactly, unless a GPU frame for a newer
    // revision is already on its way (then the old picture holds until it lands).
    let traceMs = 0;
    if (scene.sceneRevision() !== tracedRevision && !opts.fast?.pending()) {
      retrace(false);
      refreshObjects();
      traceMs = lastTraceMs;
    }

    const grid = opts.grid();
    let count = 0;
    let tir: V2[] = [];
    if (renderer) {
      syncRender();
      renderer.setGrid(grid.on, grid.spacing);
      renderer.resize(W, H, quality.pixelRatio(dpr));
      renderer.render(rig);
    }

    // The overlay: transparent, so the GPU canvas shows through.
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (renderer) {
      const ro = opts.rayOptions();
      const B = rig.basis();
      const out = new Float32Array((segs.length / STRIDE) * 9);
      let k = 0;
      for (let i = 0; i + STRIDE <= segs.length; i += STRIDE) {
        const pa = rig.project({ x: segs[i], y: segs[i + 1], z: segs[i + 9] }, B);
        const pb = rig.project({ x: segs[i + 2], y: segs[i + 3], z: segs[i + 10] }, B);
        out.set([pa.x, pa.y, pb.x, pb.y, segs[i + 4], segs[i + 5], segs[i + 6], segs[i + 7], segs[i + 8]], k);
        k += 9;
      }
      const drawn = opts.rays.draw(out, (p) => p, {
        ghostOpacity: ro.ghostOpacity,
        exposure: ro.exposure,
        dpr,
        width: W,
        height: H,
      });
      count = drawn.count;
      tir = ro.showTir ? drawn.tirDots : [];
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalCompositeOperation = "lighter";
      ctx.drawImage(opts.rays.canvas, 0, 0);
      ctx.restore();
      drawTirRings(ctx, tir);
      drawAxisLabels(ctx, rig, renderer.axisEnds());
      const so = objList.find((o) => o.id === selected);
      if (so) drawSelectionDot(ctx, rig, { x: so.pos[0], y: so.pos[1], z: so.pos[2] });
      cards.positionAll();
      cards.drawLeaders(ctx, selected);
    } else if (backend === "unavailable") {
      drawNotice(ctx, W, H, `3D renderer unavailable: ${failure}`, "error");
    } else {
      drawNotice(ctx, W, H, "loading the 3D renderer…", "info");
    }
    ctx.restore();

    opts.status({ count, ms: performance.now() - t0, traceMs });
  }

  // --- picking -------------------------------------------------------------
  // Phase 1: the projected centre and a coarse radius decide a hit.
  function pick(cx: number, cy: number): SceneObject | null {
    const B = rig.basis();
    let best: SceneObject | null = null, bestD = Infinity;
    for (const o of objList) {
      const P = { x: o.pos[0], y: o.pos[1], z: o.pos[2] };
      const c = rig.project(P, B);
      const d = Math.hypot(cx - c.x, cy - c.y);
      if (d <= o.radius * rig.pxPerMm(P, B) + 10 && d < bestD) { bestD = d; best = o; }
    }
    return best;
  }

  // --- interaction ---------------------------------------------------------
  type Mode = "orbit" | "pan" | "drag";
  let mode: Mode | null = null;
  let last: V2 = { x: 0, y: 0 };
  let dragId = -1;
  let dragStart: V3 = { x: 0, y: 0, z: 0 };
  let grab: V2 = { x: 0, y: 0 };
  const gridSnap = (v: number) => {
    const g = opts.grid();
    return g.on ? Math.round(v / g.spacing) * g.spacing : v;
  };

  view.addEventListener("pointerdown", (e) => {
    if (!owns() || e.button === 2) return;
    last = { x: e.clientX, y: e.clientY };
    const hit = e.button === 0 ? pick(e.clientX, e.clientY) : null;
    if (hit) {
      mode = "drag";
      dragId = hit.id;
      select(hit.id);
      dragStart = { x: hit.pos[0], y: hit.pos[1], z: hit.pos[2] };
      grab = { x: e.clientX, y: e.clientY };
    } else if (e.button === 1 || e.shiftKey) {
      mode = "pan";
    } else {
      mode = "orbit";
    }
    view.setPointerCapture(e.pointerId);
    view.style.cursor = "grabbing";
    quality.begin();
    invalidate();
    e.preventDefault();
  });

  view.addEventListener("pointermove", (e) => {
    if (!owns()) return;
    if (!mode) {
      updateHover(e.clientX, e.clientY);
      return;
    }
    const dx = e.clientX - last.x, dy = e.clientY - last.y;
    last = { x: e.clientX, y: e.clientY };
    if (mode === "orbit") {
      rig.orbit(dx, dy);
    } else if (mode === "pan") {
      rig.pan(dx, dy);
    } else {
      const d = rig.screenDeltaToWorld(e.clientX - grab.x, e.clientY - grab.y, dragStart);
      scene.setObjectPosition(dragId, gridSnap(dragStart.x + d.x), gridSnap(dragStart.y + d.y), gridSnap(dragStart.z + d.z));
      retrace(true);
      refreshObjects();
      cards.renderAll();
    }
    invalidate();
  });

  function endPointer(e: PointerEvent) {
    if (!mode) return;
    const settled = mode === "drag";
    mode = null;
    try { view.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    view.style.cursor = "";
    quality.end();
    if (settled) {
      retrace(false);
      refreshObjects();
      opts.detectorRefresh?.();
    }
    invalidate();
  }
  view.addEventListener("pointerup", endPointer);
  view.addEventListener("pointercancel", endPointer);

  function updateHover(cx: number, cy: number) {
    const id = pick(cx, cy)?.id ?? -1;
    if (id === hovered) return;
    hovered = id;
    view.style.cursor = id >= 0 ? "pointer" : "";
    invalidate();
  }

  view.addEventListener("wheel", (e) => {
    if (!owns()) return;
    e.preventDefault();
    rig.zoom(e.deltaY < 0 ? 1.12 : 1 / 1.12);
    quality.pulse();
    invalidate();
  }, { passive: false });

  view.addEventListener("dblclick", (e) => {
    if (!owns()) return;
    const o = pick(e.clientX, e.clientY);
    if (o) { select(o.id); cards.open(o.id); invalidate(); }
  });

  function pasteBody() {
    if (clipboard < 0) return;
    pasteN++;
    const g = opts.grid();
    const d = g.on ? g.spacing : 5;
    const id = scene.cloneBody3D(clipboard, d * pasteN, -d * pasteN, 0);
    if (id >= 0) { select(id); afterEdit(); cards.open(id); opts.detectorRefresh?.(); }
  }
  function deleteSelected() {
    if (selected < 0) return;
    if (scene.deleteObject3D(selected)) {
      cards.close(selected);
      selected = -1;
      afterEdit();
      opts.detectorRefresh?.();
    }
  }
  function nudge(key: string) {
    const o = objList.find((x) => x.id === selected);
    if (!o) return;
    const g = opts.grid();
    const s = g.on ? g.spacing : 1;
    // Arrows nudge along the image plane's horizontal and vertical.
    const { right, up } = rig.imagePlaneAxes();
    const h = key === "ArrowLeft" ? -s : key === "ArrowRight" ? s : 0;
    const v = key === "ArrowUp" ? s : key === "ArrowDown" ? -s : 0;
    scene.setObjectPosition(
      selected,
      o.pos[0] + right.x * h + up.x * v,
      o.pos[1] + right.y * h + up.y * v,
      o.pos[2] + right.z * h + up.z * v,
    );
    afterEdit();
    opts.detectorRefresh?.();
  }

  window.addEventListener("keydown", (e) => {
    if (!owns() || e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === "Escape") { cards.closeAll(); selected = -1; invalidate(); return; }
    if (e.key === "0") { frameScene(); invalidate(); return; }
    if (mod && (e.key === "c" || e.key === "C")) { if (selected >= 0) clipboard = selected; e.preventDefault(); return; }
    if (mod && (e.key === "v" || e.key === "V")) { pasteBody(); e.preventDefault(); return; }
    if (mod && (e.key === "x" || e.key === "X")) { if (selected >= 0) { clipboard = selected; deleteSelected(); } e.preventDefault(); return; }
    if (mod && (e.key === "d" || e.key === "D")) { if (selected >= 0) { clipboard = selected; pasteBody(); } e.preventDefault(); return; }
    if (e.key === "Delete" || e.key === "Backspace") { deleteSelected(); e.preventDefault(); return; }
    if (e.key.startsWith("Arrow")) { nudge(e.key); e.preventDefault(); return; }
  });

  window.addEventListener("resize", () => invalidate());

  function owns() {
    return active && opts.owns();
  }

  // --- view lifecycle ------------------------------------------------------
  function activate(selectId: number) {
    active = true;
    canvas.style.display = "block";
    retrace(false);
    refreshObjects();
    if (selectId >= 0) select(selectId);
    void ensureRenderer().then(() => {
      if (!framed && renderer) {
        rig.setSize(window.innerWidth, window.innerHeight);
        frameScene();
        framed = true;
      }
      invalidate();
    });
    invalidate();
  }

  function standDown() {
    active = false;
    mode = null;
    hovered = -1;
    quality.settleNow();
    cards.closeAll();
    canvas.style.display = "none";
    view.style.cursor = "";
    if (pending) { cancelAnimationFrame(pending); pending = 0; }
  }

  function focusObject(id: number) {
    retrace(false);
    refreshObjects();
    select(id);
    cards.open(id);
    invalidate();
  }

  const backendLabel = () => {
    switch (backend) {
      case "webgpu": return "render WebGPU";
      case "webgl2": return "render WebGL2 fallback";
      case "unavailable": return "render unavailable";
      default: return "render loading";
    }
  };

  const debug: View3dDebug = {
    ready: () => ensureRenderer(),
    camera: () => rig.state(),
    stats: () => ({
      backend,
      quality: quality.state,
      dpr: 0,
      frames: 0,
      renderMs: 0,
      gpuMs: null,
      drawCalls: 0,
      triangles: 0,
      lines: 0,
      raySegments: segs.length / STRIDE,
      meshBuilds: 0,
      meshBuildsLastSync: 0,
      meshBuildMs: 0,
      syncMs: 0,
      rayUploadMs: 0,
      environmentBuilds: 0,
      geometries: 0,
      textures: 0,
      cachedObjects: objList.length,
      ...(renderer?.stats() ?? {}),
    }),
    litPixels: async () => {
      await ensureRenderer();
      if (!renderer) return 0;
      drawFrame();
      const c = document.createElement("canvas");
      c.width = canvas.width;
      c.height = canvas.height;
      const c2 = c.getContext("2d")!;
      c2.drawImage(canvas, 0, 0);
      c2.drawImage(view, 0, 0, c.width, c.height);
      const { data } = c2.getImageData(0, 0, c.width, c.height);
      let n = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i] + data[i + 1] + data[i + 2] > 60) n++;
      return n;
    },
    project,
    hovered: () => hovered,
  };

  return {
    kind: "three",
    invalidate,
    refresh: () => { retrace(false); refreshObjects(); syncRender(true); cards.renderAll(); invalidate(); },
    selectedBody: () => selected,
    activate,
    standDown,
    focusObject,
    resetView: () => { rig.resetOrientation(); frameScene(); invalidate(); },
    setPreset: (name) => { rig.setPreset(name as CameraPreset); frameScene(); invalidate(); },
    setPerspective: (on) => { rig.setPerspective(on); invalidate(); },
    isPerspective: () => rig.perspective,
    backendLabel,
    debug,
  };
}
