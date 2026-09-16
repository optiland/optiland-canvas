// The 3D model view — an orbitable camera over the canonical Scene3.
//
// The 2D cross-section (slice3d) and this module are two views of the *same*
// Rust scene: they read the same authoritative f64 trace and the same geometry,
// and differ only in projection and interaction. Here an orbit/pan/zoom camera
// projects the scene onto the shared canvas; the rays are the real 3D bundle
// (reconstructed from the projected trace plus each endpoint's out-of-plane
// distance), and lens bodies are drawn as revolved wireframes. Editing carries
// over: click to select, drag to move in the camera plane, double-click for the
// inspector card, and the same copy/paste/delete/nudge verbs.
//
// No WebGL: the camera is a plain orthographic projection onto the 2D context,
// and rays go through the same HDR RayLayer as the 2D view (their endpoints are
// projected to screen px, so RayLayer draws them with an identity map). Additive
// rendering is order-independent, so no depth buffer is needed.

import type { Canvas } from "../pkg/oc_wasm.js";
import { createCardManager, type CardSource } from "./cards";
import type { FastPreview } from "./fastpreview";

type V2 = { x: number; y: number };
type V3 = { x: number; y: number; z: number };

const sub = (a: V3, b: V3): V3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const add = (a: V3, b: V3): V3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const scl = (a: V3, s: number): V3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
const dot = (a: V3, b: V3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: V3, b: V3): V3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
const norm = (a: V3): V3 => {
  const l = Math.hypot(a.x, a.y, a.z) || 1;
  return { x: a.x / l, y: a.y / l, z: a.z / l };
};

export type Viewport3d = {
  /** Ray segments (9 floats each: ax ay bx by r g b flux flags) already in
   *  screen px — the main renderer feeds them to the shared RayLayer with an
   *  identity screen map. */
  raySegments: () => Float32Array;
  /** Ground grid + world axes, drawn before the rays. */
  drawScaffold: (ctx: CanvasRenderingContext2D, W: number, H: number) => void;
  /** Body wireframes, the selection handle, and the inspector cards, after rays. */
  overlay: (ctx: CanvasRenderingContext2D) => void;
  /** Re-trace the f64 path and refresh geometry after an external edit. */
  refresh: () => void;
  /** The selected body id, or -1. */
  selectedBody: () => number;
  /** Become the active view: frame the scene, adopt a selection, take the canvas. */
  activate: (selectId: number) => void;
  /** Stand down: cancel drags and close cards when the 2D view takes over. */
  standDown: () => void;
  /** Select an object and open its inspector card (after adding an element). */
  focusObject: (id: number) => void;
  /** Re-frame the scene at the default 3/4 orientation. */
  resetView: () => void;
  /** Snap to a canonical view: `iso` / `front` / `top` / `side`. */
  setPreset: (name: string) => void;
  /** Orthographic (false) or perspective (true) projection. */
  setPerspective: (on: boolean) => void;
  isPerspective: () => boolean;
};

// The projected trace layout (traceProjected3D): 11 floats per segment,
// [au,av,bu,bv, r,g,b, flux, flags, dist_a, dist_b]. With the default XY view
// plane the in-plane (au,av) are world (x,y) and the signed distance is world z,
// so each endpoint reconstructs to the full 3D point (au, av, dist). §10.1.
const STRIDE = 11;

export function initViewport3d(opts: {
  scene: Canvas;
  view: HTMLCanvasElement;
  render: () => void;
  owns: () => boolean;
  onSelect?: (id: number) => void;
  detectorRefresh?: () => void;
  /** The interactive f32 picture, on WebGPU or the CPU mirror. Without it the
   *  view calls the CPU mirror directly. */
  fast?: FastPreview;
}): Viewport3d {
  const { scene, view, render } = opts;
  const owns = () => opts.owns();

  // --- camera (orthographic orbit) --------------------------------------
  // A 3/4 view by default: the optical axis (+X) runs roughly left-right, +Y is
  // up, and yaw reveals the out-of-plane +Z so the revolved barrel is visible.
  let yaw = 0.6, pitch = 0.42;
  let scale = 5; // px per mm
  let target: V3 = { x: 0, y: 0, z: 0 };
  let panX = 0, panY = 0; // screen-px pan on top of the look-at
  let framed = false;
  // Orthographic by default (predictable for optical layout); a toggle switches
  // to a mild perspective (eye `camDist` mm back), which converges at the target
  // so the framing barely shifts on toggle.
  let perspective = false;
  let camDist = 500;
  const DEFAULT_YAW = 0.6, DEFAULT_PITCH = 0.42;

  const W = () => window.innerWidth;
  const H = () => window.innerHeight;

  // The camera basis: `dir` points from the target to the eye; `right`/`up` span
  // the image plane. worldUp is +Y; pitch is clamped shy of the pole so `right`
  // never degenerates.
  function basis(): { right: V3; up: V3; dir: V3 } {
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const dir: V3 = { x: cp * sy, y: sp, z: cp * cy };
    const f = scl(dir, -1); // eye → target
    const right = norm(cross(f, { x: 0, y: 1, z: 0 }));
    const up = cross(right, f); // f, right unit and orthogonal ⇒ up is unit
    return { right, up, dir };
  }

  // World → screen (CSS px). Orthographic, or a perspective divide with the eye
  // `camDist` behind the target along `dir` and focal `scale·camDist` so the
  // on-target scale equals the orthographic one (framing is stable on toggle).
  function project(p: V3, B = basis()): V2 {
    const rel = sub(p, target);
    const xc = dot(rel, B.right), yc = dot(rel, B.up);
    const cx = W() / 2 + panX, cy = H() / 2 + panY;
    if (!perspective) return { x: cx + scale * xc, y: cy - scale * yc };
    const zc = Math.max(1, camDist - dot(rel, B.dir)); // depth from eye toward scene
    const f = scale * camDist;
    return { x: cx + (f * xc) / zc, y: cy - (f * yc) / zc };
  }

  // Approximate on-screen size of a `world`-mm length at point `p` (accounts for
  // perspective foreshortening), for hit-testing and drag mapping.
  function pxPerMm(p: V3, B = basis()): number {
    if (!perspective) return scale;
    const zc = Math.max(1, camDist - dot(sub(p, target), B.dir));
    return (scale * camDist) / zc;
  }

  // --- scene data (re-read from Rust; never mirrored) --------------------
  type Obj = { id: number; kind: string; construction: string; is3d: boolean; body: boolean; pos: number[]; radius: number; rotZ: number; rot?: number[]; opening?: number };
  type XLine = { kind: string; closed: boolean; pts: number[] };
  let segs: Float32Array = new Float32Array(0);
  let tracedRevision = -1;
  let objList: Obj[] = [];
  let xsec: XLine[] = [];

  let selected = -1;
  function select(id: number) { selected = id; opts.onSelect?.(id); }
  let clipboard = -1;
  let pasteN = 0;

  const gridBox = document.getElementById("gridBox") as HTMLInputElement | null;
  const gridSpacing = document.getElementById("gridSpacing") as HTMLInputElement | null;
  const gridOn = () => gridBox?.checked ?? false;
  const gridStep = () => Math.max(0.1, Number(gridSpacing?.value) || 5);
  const snap = (v: number) => Math.round(v / gridStep()) * gridStep();

  const objectsPanel = document.getElementById("objects3d");

  // A fast frame can arrive after `retrace` returns (WebGPU), so it is drawn only
  // when it is newer than the picture on screen, and never over an exact frame of
  // the same scene.
  let shownExact = false;
  function retrace(fast = false) {
    if (!fast) {
      opts.fast?.settled();
      segs = scene.traceProjected3D();
      tracedRevision = scene.sceneRevision();
      shownExact = true;
      return;
    }
    if (!opts.fast) {
      segs = scene.traceProjected3DFast();
      tracedRevision = scene.sceneRevision();
      shownExact = false;
      return;
    }
    opts.fast.requestProjected(scene, (f) => {
      if (f.revision < tracedRevision || (f.revision === tracedRevision && shownExact)) return;
      segs = f.segs;
      tracedRevision = f.revision;
      shownExact = false;
      if (f.late) render();
    });
  }
  function refreshGeometry() { xsec = JSON.parse(scene.sceneCrossSectionJSON()); }
  // Re-read the objects and repaint the shared inspector panel (same rows as the
  // 2D view), so #objects3d stays live after an edit made in the 3D view.
  function refreshInspector() {
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

  // The full 3D endpoints of a segment: in-plane (au,av) plus the out-of-plane
  // signed distance recovers world (x, y, z) for the default view plane.
  function seg3d(i: number): { a: V3; b: V3 } {
    return {
      a: { x: segs[i], y: segs[i + 1], z: segs[i + 9] },
      b: { x: segs[i + 2], y: segs[i + 3], z: segs[i + 10] },
    };
  }

  // --- initial framing ---------------------------------------------------
  // Centre the camera on the traced scene and pick a scale that fits it, so
  // opening the 3D view always shows the whole bundle regardless of the design.
  function frame() {
    let lo: V3 = { x: Infinity, y: Infinity, z: Infinity };
    let hi: V3 = { x: -Infinity, y: -Infinity, z: -Infinity };
    const grow = (p: V3) => {
      lo = { x: Math.min(lo.x, p.x), y: Math.min(lo.y, p.y), z: Math.min(lo.z, p.z) };
      hi = { x: Math.max(hi.x, p.x), y: Math.max(hi.y, p.y), z: Math.max(hi.z, p.z) };
    };
    // Frame on the hardware first (object positions ± radius), so a steeply
    // dispersed / stray ray running off to the scene bounds can't blow up the box.
    for (const o of objList) {
      const r = Math.max(o.radius, 1);
      grow({ x: o.pos[0] - r, y: o.pos[1] - r, z: o.pos[2] - r });
      grow({ x: o.pos[0] + r, y: o.pos[1] + r, z: o.pos[2] + r });
    }
    // Then include ray points, but only those near the hardware — reject the far
    // exit rays. Fall back to raw ray bounds if there are no objects.
    const hasObjects = isFinite(lo.x);
    const c0: V3 = hasObjects ? scl(add(lo, hi), 0.5) : { x: 0, y: 0, z: 0 };
    const objSpan = hasObjects ? Math.max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z) : Infinity;
    const limit = hasObjects ? Math.max(80, objSpan * 2) : Infinity;
    const near = (p: V3) => Math.abs(p.x - c0.x) <= limit && Math.abs(p.y - c0.y) <= limit && Math.abs(p.z - c0.z) <= limit;
    for (let i = 0; i + STRIDE <= segs.length; i += STRIDE) {
      const s = seg3d(i);
      if (near(s.a)) grow(s.a);
      if (near(s.b)) grow(s.b);
    }
    if (!isFinite(lo.x)) { target = { x: 0, y: 0, z: 0 }; scale = 5; panX = panY = 0; return; }
    target = scl(add(lo, hi), 0.5);
    // Fit the projected extent into ~80% of the viewport.
    const B = basis();
    let mx = 1, my = 1;
    const corners: V3[] = [];
    for (const x of [lo.x, hi.x]) for (const y of [lo.y, hi.y]) for (const z of [lo.z, hi.z]) corners.push({ x, y, z });
    for (const c of corners) {
      const rel = sub(c, target);
      mx = Math.max(mx, Math.abs(dot(rel, B.right)));
      my = Math.max(my, Math.abs(dot(rel, B.up)));
    }
    scale = Math.max(0.5, Math.min((W() * 0.42) / mx, (H() * 0.42) / my));
    panX = panY = 0;
  }

  // --- rays --------------------------------------------------------------
  function raySegments(): Float32Array {
    // A stale picture is retraced exactly, unless a GPU frame for a newer
    // revision is already on its way (then the old picture holds until it lands).
    if (scene.sceneRevision() !== tracedRevision && !opts.fast?.pending()) retrace();
    const B = basis();
    const out: number[] = [];
    for (let i = 0; i + STRIDE <= segs.length; i += STRIDE) {
      const { a, b } = seg3d(i);
      const pa = project(a, B), pb = project(b, B);
      out.push(pa.x, pa.y, pb.x, pb.y, segs[i + 4], segs[i + 5], segs[i + 6], segs[i + 7], segs[i + 8]);
    }
    return new Float32Array(out);
  }

  // --- geometry wireframe ------------------------------------------------
  // Each body cross-section is revolved about its own optical axis (local +X
  // through the body centre) to a wireframe: meridian copies of the meridional
  // outline plus a few latitude rings at the rims. Non-revolved bodies and
  // planar objects (detector, mirror) are drawn as their flat section.
  const STYLE: Record<string, { stroke: string; width: number }> = {
    body: { stroke: "rgba(150,185,225,0.75)", width: 1 },
    mirror: { stroke: "rgba(190,205,235,0.9)", width: 1.5 },
    surface: { stroke: "rgba(150,160,175,0.6)", width: 1 },
    detector: { stroke: "rgba(127,214,255,0.8)", width: 1.5 },
    aperture: { stroke: "rgba(210,165,120,0.8)", width: 1.5 },
  };

  function polyline(ctx: CanvasRenderingContext2D, pts3: V3[], B: ReturnType<typeof basis>) {
    ctx.beginPath();
    for (let i = 0; i < pts3.length; i++) {
      const s = project(pts3[i], B);
      if (i === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
    }
    ctx.stroke();
  }

  // Even-asphere / conic sag at radial distance `r` (mm) — the same closed form
  // the kernel uses. Guards the negative discriminant of a too-steep surface.
  function sag(c: number, k: number, r: number): number {
    const d = 1 - (1 + k) * c * c * r * r;
    return (c * r * r) / (1 + Math.sqrt(Math.max(0, d)));
  }

  type SingletGeom = { r1: number | null; r2: number | null; thickness: number; semiAperture: number; conic1: number; conic2: number };

  // The singlet's meridional outline in its LOCAL frame (optical axis = local +X,
  // centred on the body origin): front face, top rim, back face, bottom rim. Built
  // straight from the geometry, so the drawn size is exactly the clear aperture —
  // independent of where the lens sits (fixes the off-axis "lens grows" bug).
  function singletLoop(g: SingletGeom): { x: number; y: number }[] {
    const ha = g.semiAperture, t = g.thickness;
    const c1 = g.r1 && isFinite(g.r1) ? 1 / g.r1 : 0;
    const c2 = g.r2 && isFinite(g.r2) ? 1 / g.r2 : 0;
    const N = 18;
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= N; i++) { const y = -ha + (2 * ha * i) / N; pts.push({ x: -t / 2 + sag(c1, g.conic1, Math.abs(y)), y }); }
    for (let i = 0; i <= N; i++) { const y = ha - (2 * ha * i) / N; pts.push({ x: t / 2 + sag(c2, g.conic2, Math.abs(y)), y }); }
    return pts;
  }

  // The object's full local→world rotation. App-authored objects carry a pure-Z
  // rotation (rotZ); imported scenes (the optikit materializer) carry arbitrary
  // quaternions in `rot` [x,y,z,w] — vertical-axis lenses would otherwise draw
  // lying down. Falls back to rotZ when `rot` is absent.
  function rotOf(o: Obj): (v: V3) => V3 {
    if (o.rot && o.rot.length === 4) {
      const [qx, qy, qz, qw] = o.rot;
      return (v: V3): V3 => {
        // v + 2 u × (u × v + w v), u = quat vector part.
        const cx = qy * v.z - qz * v.y + qw * v.x;
        const cy = qz * v.x - qx * v.z + qw * v.y;
        const cz = qx * v.y - qy * v.x + qw * v.z;
        return {
          x: v.x + 2 * (qy * cz - qz * cy),
          y: v.y + 2 * (qz * cx - qx * cz),
          z: v.z + 2 * (qx * cy - qy * cx),
        };
      };
    }
    const a = (o.rotZ * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
    return (v: V3): V3 => ({ x: v.x * ca - v.y * sa, y: v.x * sa + v.y * ca, z: v.z });
  }

  // A local point (x, radius r), revolved by θ about the body's optical axis
  // (local +X), then placed into the world by the body's rotation + translation.
  function placer(o: Obj) {
    const R = rotOf(o);
    return (x: number, r: number, theta: number): V3 => {
      const w = R({ x, y: r * Math.cos(theta), z: r * Math.sin(theta) });
      return { x: o.pos[0] + w.x, y: o.pos[1] + w.y, z: o.pos[2] + w.z };
    };
  }

  function drawSingletWireframe(ctx: CanvasRenderingContext2D, o: Obj, g: SingletGeom, B: ReturnType<typeof basis>) {
    const loop = singletLoop(g);
    const P = placer(o);
    const MER = 8;
    for (let m = 0; m < MER; m++) {
      const th = (m / MER) * Math.PI * 2;
      polyline(ctx, loop.map((p) => P(p.x, p.y, th)), B);
    }
    // Rim rings at the front and back edges (radius = the clear semi-aperture).
    const ha = g.semiAperture;
    const c1 = g.r1 && isFinite(g.r1) ? 1 / g.r1 : 0, c2 = g.r2 && isFinite(g.r2) ? 1 / g.r2 : 0;
    for (const rimX of [-g.thickness / 2 + sag(c1, g.conic1, ha), g.thickness / 2 + sag(c2, g.conic2, ha)]) {
      const ring: V3[] = [];
      for (let k = 0; k <= 28; k++) ring.push(P(rimX, ha, (k / 28) * Math.PI * 2));
      polyline(ctx, ring, B);
    }
  }

  // A planar surface (mirror/beam block/element) as a disc: a rim circle in its
  // plane plus a diameter, sized by its radius and oriented by its rotation
  // (the disc lies in the object's local z = 0 plane, kernel convention).
  function drawSurfaceDisc(ctx: CanvasRenderingContext2D, o: Obj, B: ReturnType<typeof basis>) {
    const rot = rotOf(o);
    const u1: V3 = rot({ x: 1, y: 0, z: 0 });
    const u2: V3 = rot({ x: 0, y: 1, z: 0 });
    const C: V3 = { x: o.pos[0], y: o.pos[1], z: o.pos[2] };
    const R = o.radius;
    const ring: V3[] = [];
    for (let k = 0; k <= 32; k++) {
      const t = (k / 32) * Math.PI * 2;
      ring.push(add(C, add(scl(u1, R * Math.cos(t)), scl(u2, R * Math.sin(t)))));
    }
    polyline(ctx, ring, B);
    polyline(ctx, [add(C, scl(u1, -R)), add(C, scl(u1, R))], B); // the meridional diameter
  }

  // A local (x, y) point extruded to z-height `zl`, placed into the world by
  // the body's rotation + translation.
  function extrudePlacer(o: Obj) {
    const R = rotOf(o);
    return (x: number, y: number, zl: number): V3 => {
      const w = R({ x, y, z: zl });
      return { x: o.pos[0] + w.x, y: o.pos[1] + w.y, z: o.pos[2] + w.z };
    };
  }

  // A prism (extruded profile): front + back polygons joined by rails.
  function drawExtrude(ctx: CanvasRenderingContext2D, o: Obj, ext: { zMin: number; zMax: number; points: number[][] }, B: ReturnType<typeof basis>) {
    if (!ext.points || ext.points.length < 2) return;
    const wp = extrudePlacer(o);
    const front = ext.points.map(([x, y]) => wp(x, y, ext.zMin));
    const back = ext.points.map(([x, y]) => wp(x, y, ext.zMax));
    polyline(ctx, [...front, front[0]], B);
    polyline(ctx, [...back, back[0]], B);
    for (let i = 0; i < front.length; i++) polyline(ctx, [front[i], back[i]], B);
  }

  // An aperture: two concentric circles (the absorbing mask rim and the clear
  // opening) in the aperture plane.
  function drawAperture(ctx: CanvasRenderingContext2D, o: Obj, B: ReturnType<typeof basis>) {
    const rot = rotOf(o);
    const u1: V3 = rot({ x: 1, y: 0, z: 0 });
    const u2: V3 = rot({ x: 0, y: 1, z: 0 });
    const C: V3 = { x: o.pos[0], y: o.pos[1], z: o.pos[2] };
    const circle = (R: number) => {
      const ring: V3[] = [];
      for (let k = 0; k <= 32; k++) { const t = (k / 32) * Math.PI * 2; ring.push(add(C, add(scl(u1, R * Math.cos(t)), scl(u2, R * Math.sin(t))))); }
      polyline(ctx, ring, B);
    };
    circle(o.radius); // absorbing mask rim
    const ap = JSON.parse(scene.apertureOpeningJSON(o.id)); // {inner, outer} min/max radius
    circle(ap ? ap.outer : o.opening ?? o.radius * 0.5); // clear opening (max radius)
    if (ap && ap.inner > 0) circle(ap.inner); // central obstruction (min radius)
  }

  function drawGeometry(ctx: CanvasRenderingContext2D) {
    const B = basis();
    for (const o of objList) {
      const st = STYLE[o.kind] ?? (o.body ? STYLE.body : STYLE.surface);
      ctx.strokeStyle = st.stroke;
      ctx.lineWidth = st.width;
      if (o.body) {
        const g = JSON.parse(scene.singletGeomJSON(o.id)) as SingletGeom | null;
        if (g) drawSingletWireframe(ctx, o, g, B);
        else {
          const ext = JSON.parse(scene.bodyExtrudeJSON(o.id));
          if (ext) drawExtrude(ctx, o, ext, B); // a prism
          else drawBodyFromXsec(ctx, o, B); // other bodies: flat section fallback
        }
      } else if (o.kind === "aperture") {
        drawAperture(ctx, o, B);
      } else {
        drawSurfaceDisc(ctx, o, B);
      }
    }
  }

  // Fallback for a non-singlet body: draw its z=0 cross-section flat (rare — the
  // authored scenes are singlets). Matches the polyline nearest this body.
  function drawBodyFromXsec(ctx: CanvasRenderingContext2D, o: Obj, B: ReturnType<typeof basis>) {
    let best: XLine | null = null, bestD = Infinity;
    for (const el of xsec) {
      if (el.kind !== "body" || el.pts.length < 4) continue;
      let cx = 0, cy = 0;
      for (let i = 0; i < el.pts.length; i += 2) { cx += el.pts[i]; cy += el.pts[i + 1]; }
      const n = el.pts.length / 2;
      const d = (o.pos[0] - cx / n) ** 2 + (o.pos[1] - cy / n) ** 2;
      if (d < bestD) { bestD = d; best = el; }
    }
    if (!best) return;
    const flat: V3[] = [];
    for (let i = 0; i < best.pts.length; i += 2) flat.push({ x: best.pts[i], y: best.pts[i + 1], z: 0 });
    if (best.closed) flat.push(flat[0]);
    polyline(ctx, flat, B);
  }

  // --- selection handle --------------------------------------------------
  function drawSelection(ctx: CanvasRenderingContext2D) {
    if (selected < 0) return;
    const o = objList.find((x) => x.id === selected);
    if (!o) return;
    const C: V3 = { x: o.pos[0], y: o.pos[1], z: o.pos[2] };
    const c = project(C);
    ctx.strokeStyle = "rgba(127,214,255,0.8)";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(c.x, c.y, o.radius * pxPerMm(C) + 4, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "#7fd6ff";
    ctx.beginPath(); ctx.arc(c.x, c.y, 3.5, 0, Math.PI * 2); ctx.fill();
  }

  // --- scaffold: world axes + ground grid --------------------------------
  function drawScaffold(ctx: CanvasRenderingContext2D, _w: number, _h: number) {
    const B = basis();
    if (gridOn()) {
      const sp = gridStep();
      const span = 12 * sp; // a modest ground patch around the target
      const line = (a: V3, b: V3, style: string) => {
        ctx.strokeStyle = style; ctx.lineWidth = 1;
        const pa = project(a, B), pb = project(b, B);
        ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
      };
      const cx0 = Math.round(target.x / sp) * sp, cz0 = Math.round(target.z / sp) * sp;
      for (let x = cx0 - span; x <= cx0 + span + 1e-9; x += sp) {
        line({ x, y: 0, z: cz0 - span }, { x, y: 0, z: cz0 + span },
          Math.abs(x) < 1e-9 ? "rgba(120,150,190,0.22)" : "rgba(120,150,190,0.06)");
      }
      for (let z = cz0 - span; z <= cz0 + span + 1e-9; z += sp) {
        line({ x: cx0 - span, y: 0, z }, { x: cx0 + span, y: 0, z },
          Math.abs(z) < 1e-9 ? "rgba(120,150,190,0.22)" : "rgba(120,150,190,0.06)");
      }
    }
    // The three world axes from the origin (Zemax labels: Z along the optical
    // axis +X, Y up, X-out-of-plane reads as +Z). Kept short so they orient
    // without cluttering.
    const O: V3 = { x: 0, y: 0, z: 0 };
    const L = 18;
    const axis = (v: V3, col: string, label: string) => {
      const o = project(O, B), e = project(v, B);
      ctx.strokeStyle = col; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(e.x, e.y); ctx.stroke();
      ctx.fillStyle = col; ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(label, e.x + 3, e.y - 3);
    };
    axis({ x: L, y: 0, z: 0 }, "rgba(150,175,210,0.7)", "Z"); // optical axis
    axis({ x: 0, y: L, z: 0 }, "rgba(150,210,175,0.7)", "Y"); // vertical
    axis({ x: 0, y: 0, z: L }, "rgba(210,175,150,0.7)", "X"); // out-of-plane
  }

  // --- inspector card (same engine as the 2D view) ----------------------
  const card3dTemplate = document.getElementById("card3dTemplate") as HTMLTemplateElement;

  function afterEdit() {
    retrace(false);
    refreshGeometry();
    refreshInspector();
    cardMgr.renderAll();
    render();
  }

  function commitCard(id: number, root: HTMLElement) {
    const num = (n: string) => Number((root.querySelector(`input[data-name="${n}"]`) as HTMLInputElement).value);
    const o = objList.find((o) => o.id === id);
    scene.setObjectPosition(id, num("x"), num("y"), o ? o.pos[2] : 0);
    // Tilt: a body Z-rotation, or a surface's normal-angle setter.
    if (o) { if (o.body) scene.setObjectRotationZDeg(id, num("rot")); else scene.setSurfaceTiltZDeg(id, num("rot")); }
    if (JSON.parse(scene.singletGeomJSON(id))) {
      scene.setSingletGeom(id, num("r1"), num("r2"), num("thickness"), num("semiAperture"), num("conic1"), num("conic2"));
    }
    if (JSON.parse(scene.bodyGlassJSON(id))) {
      scene.setBodyGlassAbbe(id, num("nd"), num("vd"));
    }
    if (o && !o.body) {
      const sel = (n: string) => (root.querySelector(`select[data-name="${n}"]`) as HTMLSelectElement)?.value;
      const f = sel("front"), b = sel("back");
      if (f && b) scene.setSurfaceResponse(id, f, b);
    }
    afterEdit();
  }

  // Show/populate the per-face response block for a surface object (hidden for
  // bodies), reading the current responses from Rust.
  function renderSurfaceCard(id: number, root: HTMLElement, o: Obj) {
    const block = root.querySelector(".c3-surface") as HTMLElement | null;
    if (!block) return;
    const r = o.body ? null : JSON.parse(scene.surfaceResponseJSON(id));
    if (!r) { block.style.display = "none"; return; } // bodies + apertures have no per-face response
    block.style.display = "block";
    const set = (n: string, v: string) => {
      const s = root.querySelector(`select[data-name="${n}"]`) as HTMLSelectElement;
      if (s && document.activeElement !== s) s.value = v;
    };
    set("front", r.front); set("back", r.back);
  }

  const cardSource: CardSource = {
    template: card3dTemplate,
    // The 3D anchor is already in screen px (projected through the camera), so
    // the manager's world→screen step is the identity.
    toScreen: (p) => p,
    render2d: render,
    anchorWorld: (id) => {
      const o = objList.find((o) => o.id === id);
      return o ? project({ x: o.pos[0], y: o.pos[1], z: o.pos[2] }) : null;
    },
    render(id, root) {
      const o = objList.find((o) => o.id === id);
      if (!o) return false;
      const q = (sel: string) => root.querySelector(sel) as HTMLElement;
      q(".c-title").textContent = `#${id} · ${o.kind === "profile" ? o.construction : o.kind}`;
      const setV = (name: string, v: number | null) => {
        const inp = root.querySelector(`input[data-name="${name}"]`) as HTMLInputElement;
        if (!inp || document.activeElement === inp) return;
        inp.value = v == null ? "" : String(Number(v.toFixed(4)));
      };
      setV("x", o.pos[0]);
      setV("y", o.pos[1]);
      setV("rot", o.rotZ); // tilt: body Z-rotation or surface normal angle
      const g = JSON.parse(scene.singletGeomJSON(id));
      const glass = JSON.parse(scene.bodyGlassJSON(id));
      q(".c3-singlet").style.display = g ? "block" : "none";
      if (g) {
        setV("r1", g.r1); setV("r2", g.r2); setV("thickness", g.thickness);
        setV("semiAperture", g.semiAperture); setV("conic1", g.conic1); setV("conic2", g.conic2);
      }
      q(".c3-glass").style.display = glass ? "block" : "none"; // any glass body (singlet, prism)
      if (glass) { setV("nd", glass.nd); setV("vd", glass.vd); }
      renderSurfaceCard(id, root, o);
      return true;
    },
    wire(id, root) {
      root.querySelectorAll<HTMLInputElement>("input[data-name]").forEach((input) => {
        let before = "";
        input.addEventListener("focus", () => { before = input.value; });
        input.addEventListener("keydown", (e) => {
          if (e.key === "Escape") { input.value = before; input.blur(); commitCard(id, root); }
          e.stopPropagation();
        });
        input.addEventListener("input", () => commitCard(id, root));
        input.addEventListener("wheel", (e) => {
          e.preventDefault();
          const step = Number(input.step) || 0.5;
          const dir = e.deltaY < 0 ? 1 : -1;
          input.value = String(Number((Number(input.value) + dir * step * (e.shiftKey ? 10 : 1)).toFixed(6)));
          commitCard(id, root);
        }, { passive: false });
      });
      root.querySelectorAll<HTMLSelectElement>("select[data-name]").forEach((sel) => {
        sel.addEventListener("keydown", (e) => e.stopPropagation());
        sel.addEventListener("change", () => commitCard(id, root));
      });
    },
  };

  const cardMgr = createCardManager(cardSource);

  // --- interaction: orbit / pan / zoom / select / drag ------------------
  type Mode = "orbit" | "pan" | "drag";
  let mode: Mode | null = null;
  let last: V2 = { x: 0, y: 0 };
  let dragId = -1;
  let dragStart: V3 = { x: 0, y: 0, z: 0 };
  let grab: V2 = { x: 0, y: 0 };

  function pickBody(cx: number, cy: number): Obj | null {
    const B = basis();
    let best: Obj | null = null, bestD = Infinity;
    for (const o of objList) {
      const P: V3 = { x: o.pos[0], y: o.pos[1], z: o.pos[2] };
      const c = project(P, B);
      const d = Math.hypot(cx - c.x, cy - c.y);
      if (d <= o.radius * pxPerMm(P, B) + 10 && d < bestD) { bestD = d; best = o; }
    }
    return best;
  }

  view.addEventListener("pointerdown", (e) => {
    if (!owns() || e.button === 2) return;
    last = { x: e.clientX, y: e.clientY };
    const hit = e.button === 0 ? pickBody(e.clientX, e.clientY) : null;
    if (hit) {
      mode = "drag";
      dragId = hit.id;
      select(hit.id);
      dragStart = { x: hit.pos[0], y: hit.pos[1], z: hit.pos[2] };
      grab = { x: e.clientX, y: e.clientY };
      render();
    } else if (e.button === 1 || e.shiftKey) {
      mode = "pan";
    } else {
      mode = "orbit";
    }
    view.setPointerCapture(e.pointerId);
    e.preventDefault();
  });

  view.addEventListener("pointermove", (e) => {
    if (!mode) return;
    const dx = e.clientX - last.x, dy = e.clientY - last.y;
    last = { x: e.clientX, y: e.clientY };
    if (mode === "orbit") {
      yaw += dx * 0.01;
      pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.01));
      cardMgr.positionAll();
      render();
    } else if (mode === "pan") {
      panX += dx; panY += dy;
      cardMgr.positionAll();
      render();
    } else if (mode === "drag") {
      const B = basis();
      // Screen delta → world delta in the camera plane (an intuitive grab-move),
      // scaled by the depth-aware px/mm at the object so it tracks the cursor.
      const ppm = pxPerMm(dragStart, B);
      const du = (e.clientX - grab.x) / ppm;
      const dv = -(e.clientY - grab.y) / ppm;
      let p = add(dragStart, add(scl(B.right, du), scl(B.up, dv)));
      if (gridOn()) p = { x: snap(p.x), y: snap(p.y), z: snap(p.z) };
      scene.setObjectPosition(dragId, p.x, p.y, p.z);
      retrace(true);
      refreshGeometry();
      refreshInspector();
      cardMgr.renderAll();
      render();
    }
  });

  function endPointer(e: PointerEvent) {
    if (!mode) return;
    const settled = mode === "drag";
    mode = null;
    try { view.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    if (settled) { retrace(false); refreshInspector(); render(); opts.detectorRefresh?.(); }
  }
  view.addEventListener("pointerup", endPointer);
  view.addEventListener("pointercancel", endPointer);

  view.addEventListener("wheel", (e) => {
    if (!owns()) return;
    e.preventDefault();
    scale = Math.max(0.3, Math.min(200, scale * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
    cardMgr.positionAll();
    render();
  }, { passive: false });

  view.addEventListener("dblclick", (e) => {
    if (!owns()) return;
    const o = pickBody(e.clientX, e.clientY);
    if (o) { select(o.id); cardMgr.open(o.id); render(); }
  });

  // --- copy / paste / delete / nudge on the selection -------------------
  function pasteBody() {
    if (clipboard < 0) return;
    pasteN++;
    const d = gridOn() ? gridStep() : 5;
    const id = scene.cloneBody3D(clipboard, d * pasteN, -d * pasteN, 0);
    if (id >= 0) { select(id); afterEdit(); cardMgr.open(id); opts.detectorRefresh?.(); }
  }
  function deleteSelected() {
    if (selected < 0) return;
    if (scene.deleteObject3D(selected)) { cardMgr.close(selected); selected = -1; afterEdit(); opts.detectorRefresh?.(); }
  }
  function nudge(key: string) {
    const o = objList.find((x) => x.id === selected);
    if (!o) return;
    const s = gridOn() ? gridStep() : 1;
    // In the 3D view the arrows nudge in the image plane's horizontal/vertical.
    const B = basis();
    const h = key === "ArrowLeft" ? -s : key === "ArrowRight" ? s : 0;
    const v = key === "ArrowUp" ? s : key === "ArrowDown" ? -s : 0;
    const p = add({ x: o.pos[0], y: o.pos[1], z: o.pos[2] }, add(scl(B.right, h), scl(B.up, v)));
    scene.setObjectPosition(selected, p.x, p.y, p.z);
    afterEdit();
    opts.detectorRefresh?.();
  }

  window.addEventListener("keydown", (e) => {
    if (!owns() || e.target instanceof HTMLInputElement) return;
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === "Escape") { cardMgr.closeAll(); selected = -1; render(); return; }
    if (e.key === "0") { frame(); cardMgr.positionAll(); render(); return; }
    if (mod && (e.key === "c" || e.key === "C")) { if (selected >= 0) clipboard = selected; e.preventDefault(); return; }
    if (mod && (e.key === "v" || e.key === "V")) { pasteBody(); e.preventDefault(); return; }
    if (mod && (e.key === "x" || e.key === "X")) { if (selected >= 0) { clipboard = selected; deleteSelected(); } e.preventDefault(); return; }
    if (mod && (e.key === "d" || e.key === "D")) { if (selected >= 0) { clipboard = selected; pasteBody(); } e.preventDefault(); return; }
    if (e.key === "Delete" || e.key === "Backspace") { deleteSelected(); e.preventDefault(); return; }
    if (e.key.startsWith("Arrow")) { nudge(e.key); e.preventDefault(); return; }
  });

  function activate(selectId: number) {
    retrace(false);
    refreshGeometry();
    refreshInspector();
    if (!framed) { frame(); framed = true; }
    if (selectId >= 0) select(selectId);
  }
  function standDown() {
    mode = null;
    cardMgr.closeAll();
  }
  // Re-read, select an object, and open its card (used after adding an element).
  function focusObject(id: number) {
    retrace(false); refreshGeometry(); refreshInspector();
    select(id); cardMgr.open(id); render();
  }

  // --- view controls -----------------------------------------------------
  // Reset re-frames the scene at the default 3/4 orientation. Presets snap to the
  // canonical optical views (all Zemax-labelled: the optical axis is +X → "Z").
  function resetView() { yaw = DEFAULT_YAW; pitch = DEFAULT_PITCH; frame(); cardMgr.positionAll(); render(); }
  const PRESETS: Record<string, [number, number]> = {
    iso: [DEFAULT_YAW, DEFAULT_PITCH],
    front: [0, 0.0001],       // meridional Y–Z plane, face-on
    top: [0, 1.45],           // looking down the vertical
    side: [Math.PI / 2, 0.0001], // end-on down the optical axis
  };
  function setPreset(name: string) {
    const p = PRESETS[name];
    if (!p) return;
    [yaw, pitch] = p;
    frame(); cardMgr.positionAll(); render();
  }
  function setPerspective(on: boolean) { perspective = on; cardMgr.positionAll(); render(); }
  const isPerspective = () => perspective;

  return {
    raySegments,
    drawScaffold,
    refresh: () => { retrace(false); refreshGeometry(); refreshInspector(); cardMgr.renderAll(); },
    selectedBody: () => selected,
    activate,
    standDown,
    focusObject,
    resetView,
    setPreset,
    setPerspective,
    isPerspective,
    overlay(ctx) {
      cardMgr.positionAll();
      drawGeometry(ctx);
      drawSelection(ctx);
      cardMgr.drawLeaders(ctx, selected);
    },
  };
}
