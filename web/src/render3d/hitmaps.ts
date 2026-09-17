// Detector hit maps: each area detector's incident-flux cells, painted on its
// active face.
//
// The cells come from the exact f64 trace the view already ran for the rays
// (`lastDetectorMapsF32`), the same cells the detector dock's heatmap shows, in
// the same colours (heat.ts). Each detector is scaled to its own peak, as the
// dock is. This is a picture of the readout; it computes nothing and is never
// read back as a number. While a drag shows the interactive preview, the maps
// are hidden, because they belong to the last exact trace.

import * as THREE from "three/webgpu";
import { heatRgb } from "../heat";
import { MaterialLibrary } from "./materials";
import type { SceneRenderCache } from "./scene-cache";
import { FaceGroup, type DetectorMap } from "./types";

type Painted = {
  texture: THREE.DataTexture;
  material: THREE.Material;
  half: { value: THREE.Vector2 };
  visible: { value: number };
};

export class HitMapLayer {
  private readonly painted = new Map<number, Painted>();
  private enabled = true;
  private shown = true;
  /** Hit-map materials created since start (a new one needs its shader). */
  created = 0;

  constructor(private readonly cache: SceneRenderCache) {}

  /** Paint these maps; detectors missing from `maps` go back to plain faces.
   *  Returns true when a new material was created. */
  update(maps: DetectorMap[]): boolean {
    let fresh = false;
    const live = new Set<number>();
    for (const map of maps) {
      live.add(map.id);
      let p = this.painted.get(map.id);
      if (p && (p.texture.image.width !== map.nu || p.texture.image.height !== map.nv)) {
        this.drop(map.id);
        p = undefined;
      }
      if (!p) {
        p = this.create(map);
        this.painted.set(map.id, p);
        fresh = true;
      }
      fill(p.texture, map);
      p.half.value.set(map.halfW, map.halfH);
      p.visible.value = this.shown ? 1 : 0;
      if (this.enabled) this.cache.setSlotOverride(map.id, FaceGroup.DetectorActive, p.material);
    }
    for (const id of [...this.painted.keys()]) if (!live.has(id)) this.drop(id);
    return fresh;
  }

  /** The user's toggle: with maps off, detector faces are plain. */
  setEnabled(on: boolean) {
    if (on === this.enabled) return;
    this.enabled = on;
    for (const [id, p] of this.painted) {
      this.cache.setSlotOverride(id, FaceGroup.DetectorActive, on ? p.material : null);
    }
  }

  /** Hide the maps without dropping them (a drag preview is on screen). */
  setShown(shown: boolean) {
    if (shown === this.shown) return;
    this.shown = shown;
    for (const p of this.painted.values()) p.visible.value = shown ? 1 : 0;
  }

  get size(): number {
    return this.painted.size;
  }

  private create(map: DetectorMap): Painted {
    const data = new Uint8Array(map.nu * map.nv * 4);
    const texture = new THREE.DataTexture(data, map.nu, map.nv, THREE.RGBAFormat, THREE.UnsignedByteType);
    // One texel per cell, drawn as cells.
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    texture.generateMipmaps = false;
    texture.colorSpace = THREE.SRGBColorSpace;
    const { material, half, visible } = MaterialLibrary.hitMap(texture, map.halfW, map.halfH);
    this.created++;
    return { texture, material, half, visible };
  }

  private drop(id: number) {
    const p = this.painted.get(id);
    if (!p) return;
    this.cache.setSlotOverride(id, FaceGroup.DetectorActive, null);
    p.material.dispose();
    p.texture.dispose();
    this.painted.delete(id);
  }

  dispose() {
    for (const id of [...this.painted.keys()]) this.drop(id);
  }
}

/** Write a map into its texture: empty cells transparent, the rest on the heat
 *  ramp relative to the detector's peak. */
function fill(texture: THREE.DataTexture, map: DetectorMap) {
  const data = texture.image.data as Uint8Array;
  let peak = 0;
  for (const v of map.cells) if (v > peak) peak = v;
  for (let i = 0; i < map.cells.length; i++) {
    const v = map.cells[i];
    const k = i * 4;
    if (!(v > 0) || peak <= 0) {
      data[k] = data[k + 1] = data[k + 2] = data[k + 3] = 0;
      continue;
    }
    const [r, g, b] = heatRgb(v / peak);
    data[k] = r;
    data[k + 1] = g;
    data[k + 2] = b;
    data[k + 3] = 255;
  }
  texture.needsUpdate = true;
}
