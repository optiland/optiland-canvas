// Hover and selection cues for the 3D view.
//
// A cue never replaces an object's material. It is an extra, light-weight draw
// of the same geometry:
//   hover     a faint glow that grows toward the silhouette
//   selection a stronger glow plus a thin outline of the object's creases
// Both are drawn after the glass so they stay visible on clear bodies. The
// outline is the object's crease lines from Rust (rims, corners, outlines) and
// ignores depth, so a selected object's outline is visible even where another
// object covers it. The colour follows the background.

import * as THREE from "three/webgpu";
import { abs, dot, float, normalView, positionViewDirection, uniform } from "three/tsl";
import type { SceneRenderCache } from "./scene-cache";

const DARK_BG_CUE = new THREE.Color(0x7fd6ff);
const LIGHT_BG_CUE = new THREE.Color(0x0f6fa8);

function glowMaterial(color: ReturnType<typeof makeColor>, base: number, rim: number): THREE.MeshBasicNodeMaterial {
  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
  const facing = abs(dot(normalView, positionViewDirection));
  const edge = float(1).sub(facing).pow(2);
  m.colorNode = color;
  m.opacityNode = edge.mul(rim).add(base).min(0.9);
  m.name = "cue-glow";
  return m;
}

function makeColor(c: THREE.Color) {
  return uniform(c.clone());
}

type Cue = { id: number; glow: THREE.Mesh; outline: THREE.LineSegments | null; geometry: THREE.BufferGeometry };

export class CueLayer {
  private readonly color = makeColor(DARK_BG_CUE);
  private readonly hoverMaterial = glowMaterial(this.color, 0, 0.18);
  private readonly selectMaterial = glowMaterial(this.color, 0.02, 0.3);
  private readonly outlineMaterial = new THREE.LineBasicNodeMaterial({
    transparent: true,
    opacity: 0.95,
    depthTest: false,
    depthWrite: false,
  });
  private hover: Cue | null = null;
  private select: Cue | null = null;
  hoveredId = -1;
  selectedId = -1;

  constructor(private readonly cache: SceneRenderCache) {
    this.outlineMaterial.colorNode = this.color;
  }

  setLightBackground(light: boolean) {
    this.color.value.copy(light ? LIGHT_BG_CUE : DARK_BG_CUE);
  }

  setHovered(id: number) {
    // The selected object already carries a stronger cue.
    const target = id === this.selectedId ? -1 : id;
    if (target === this.hoveredId && this.fresh(this.hover, target)) return;
    this.hoveredId = target;
    this.hover = this.replace(this.hover, target, false);
  }

  setSelected(id: number) {
    if (id === this.selectedId && this.fresh(this.select, id)) return;
    this.selectedId = id;
    this.select = this.replace(this.select, id, true);
    if (this.hoveredId === id) this.setHovered(id);
  }

  /** Re-attach cues after the cache rebuilt an object's geometry. */
  refresh(id: number) {
    if (id === this.selectedId) this.select = this.replace(this.select, id, true);
    if (id === this.hoveredId) this.hover = this.replace(this.hover, id, false);
  }

  /** Is `cue` still attached to the current geometry of object `id`? */
  private fresh(cue: Cue | null, id: number): boolean {
    if (id < 0) return cue === null;
    const e = this.cache.get(id);
    return !!cue && !!e && cue.id === id && cue.geometry === e.root.geometry && cue.glow.parent === e.root;
  }

  private replace(old: Cue | null, id: number, selected: boolean): Cue | null {
    if (old) {
      old.glow.removeFromParent();
      // The outline's geometry belongs to the render cache.
      old.outline?.removeFromParent();
    }
    const e = id >= 0 ? this.cache.get(id) : undefined;
    if (!e || !(e.root instanceof THREE.Mesh)) return null;
    const geometry = e.root.geometry;
    const glow = new THREE.Mesh(geometry, selected ? this.selectMaterial : this.hoverMaterial);
    glow.renderOrder = 30;
    glow.raycast = () => {};
    e.root.add(glow);
    let outline: THREE.LineSegments | null = null;
    if (selected && e.edges) {
      outline = new THREE.LineSegments(e.edges, this.outlineMaterial);
      outline.renderOrder = 31;
      outline.raycast = () => {};
      e.root.add(outline);
    }
    return { id, glow, outline, geometry };
  }

  dispose() {
    this.replace(this.hover, -1, false);
    this.replace(this.select, -1, true);
    this.hoverMaterial.dispose();
    this.selectMaterial.dispose();
    this.outlineMaterial.dispose();
  }
}
