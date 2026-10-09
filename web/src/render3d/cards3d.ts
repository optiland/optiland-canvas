// Inspector cards for the GPU 3D view: the Scene3 card adapter over `../cards`.
//
// The same fields and write-back rules as the 2D cross-section's cards
// (slice3d.ts): every field writes straight to the Rust scene, and a refused
// edit is reverted by the next render, which re-reads the real value.

import type { Canvas } from "../../pkg/oc_wasm.js";
import { createCardManager, type CardManager, type V2 } from "../cards";
import { commitAperture, commitLink, fieldChanged, renderAperture, renderLink } from "../card-extras";

export type SceneObject = {
  id: number;
  kind: string;
  construction: string;
  is3d: boolean;
  body: boolean;
  pos: number[];
  radius: number;
  rotZ: number;
  rot?: number[];
  opening?: number;
};

export function createScene3Cards(opts: {
  scene: Canvas;
  objects: () => SceneObject[];
  /** A world point to CSS pixels, through the current camera. */
  project: (p: { x: number; y: number; z: number }) => V2;
  /** Redraw (card drags move leader lines). */
  render: () => void;
  /** After a field wrote to the scene: retrace exactly and refresh. */
  afterEdit: () => void;
}): CardManager {
  const { scene } = opts;
  const template = document.getElementById("card3dTemplate") as HTMLTemplateElement;
  const find = (id: number) => opts.objects().find((o) => o.id === id);

  function commit(id: number, root: HTMLElement) {
    const num = (n: string) => Number((root.querySelector(`input[data-name="${n}"]`) as HTMLInputElement).value);
    const o = find(id);
    // Only what was changed: an untouched x, y or tilt written back would
    // flatten a 3D pose, or undo where a link placed the element.
    if (o && (fieldChanged(root, "x", o.pos[0]) || fieldChanged(root, "y", o.pos[1]))) {
      scene.setObjectPosition(id, num("x"), num("y"), o.pos[2]);
    }
    if (o && fieldChanged(root, "rot", o.rotZ)) {
      // A source turns like a body (its aim is its local +X).
      if (o.body || o.kind === "source") scene.setObjectRotationZDeg(id, num("rot"));
      else scene.setSurfaceTiltZDeg(id, num("rot"));
    }
    if (JSON.parse(scene.singletGeomJSON(id))) {
      scene.setSingletGeom(id, num("r1"), num("r2"), num("thickness"), num("semiAperture"), num("conic1"), num("conic2"));
    }
    if (JSON.parse(scene.bodyGlassJSON(id))) scene.setBodyGlassAbbe(id, num("nd"), num("vd"));
    commitAperture(scene, id, root);
    commitLink(scene, id, root);
    if (o && !o.body && o.kind !== "source") {
      const sel = (n: string) => (root.querySelector(`select[data-name="${n}"]`) as HTMLSelectElement | null)?.value;
      const f = sel("front"), b = sel("back");
      if (f && b && JSON.parse(scene.surfaceResponseJSON(id))) scene.setSurfaceResponse(id, f, b);
    }
    opts.afterEdit();
  }

  return createCardManager({
    template,
    // The anchor is already projected to screen pixels.
    toScreen: (p) => p,
    render2d: opts.render,
    anchorWorld: (id) => {
      const o = find(id);
      return o ? opts.project({ x: o.pos[0], y: o.pos[1], z: o.pos[2] }) : null;
    },
    render(id, root) {
      const o = find(id);
      if (!o) return false; // deleted: the manager closes the card
      const q = (sel: string) => root.querySelector(sel) as HTMLElement;
      q(".c-title").textContent = `#${id} · ${o.kind === "profile" ? o.construction : o.kind}`;
      const setV = (name: string, v: number | null) => {
        const inp = root.querySelector(`input[data-name="${name}"]`) as HTMLInputElement | null;
        if (!inp || document.activeElement === inp) return;
        inp.value = v == null ? "" : String(Number(v.toFixed(4)));
      };
      setV("x", o.pos[0]);
      setV("y", o.pos[1]);
      setV("rot", o.rotZ);
      const g = JSON.parse(scene.singletGeomJSON(id));
      q(".c3-singlet").style.display = g ? "block" : "none";
      if (g) {
        setV("r1", g.r1); setV("r2", g.r2); setV("thickness", g.thickness);
        setV("semiAperture", g.semiAperture); setV("conic1", g.conic1); setV("conic2", g.conic2);
      }
      const glass = JSON.parse(scene.bodyGlassJSON(id));
      q(".c3-glass").style.display = glass ? "block" : "none";
      if (glass) { setV("nd", glass.nd); setV("vd", glass.vd); }
      renderAperture(scene, id, root);
      renderLink(scene, id, root, opts.objects());
      const block = q(".c3-surface");
      const r = o.body || o.kind === "source" ? null : JSON.parse(scene.surfaceResponseJSON(id));
      block.style.display = r ? "block" : "none";
      if (r) {
        for (const [name, value] of [["front", r.front], ["back", r.back]] as const) {
          const s = root.querySelector(`select[data-name="${name}"]`) as HTMLSelectElement | null;
          if (s && document.activeElement !== s) s.value = value;
        }
      }
      return true;
    },
    wire(id, root) {
      root.querySelectorAll<HTMLInputElement>("input[data-name]").forEach((input) => {
        let before = "";
        input.addEventListener("focus", () => { before = input.value; });
        input.addEventListener("keydown", (e) => {
          if (e.key === "Escape") { input.value = before; input.blur(); commit(id, root); }
          e.stopPropagation(); // keep field keys out of the view shortcuts
        });
        input.addEventListener("input", () => commit(id, root));
        input.addEventListener("wheel", (e) => {
          e.preventDefault();
          const step = Number(input.step) || 0.5;
          const dir = e.deltaY < 0 ? 1 : -1;
          input.value = String(Number((Number(input.value) + dir * step * (e.shiftKey ? 10 : 1)).toFixed(6)));
          commit(id, root);
        }, { passive: false });
      });
      root.querySelectorAll<HTMLSelectElement>("select[data-name]").forEach((sel) => {
        sel.addEventListener("keydown", (e) => e.stopPropagation());
        sel.addEventListener("change", () => commit(id, root));
      });
    },
  });
}
