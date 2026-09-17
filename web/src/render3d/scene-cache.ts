// The render cache: one disposable three.js proxy per Rust object.
//
// A cached entry means "the picture of Rust object N at revisions (g, t, m)",
// never the object itself. On each sync the descriptors from Rust are compared
// with the cache:
//   new object        -> fetch its mesh, build geometry, create the proxy
//   removed object    -> remove the proxy, dispose its geometry
//   geometry revision -> fetch a new mesh, replace and dispose the old geometry
//   transform revision-> set position and quaternion only
//   material revision -> reassign shared materials only
//   nothing changed   -> nothing
// Materials are shared (materials.ts) and never disposed here.

import * as THREE from "three/webgpu";
import type { RenderMesh } from "../../pkg/oc_wasm.js";
import type { Bounds3 } from "./camera";
import { buildGeometry, readMesh } from "./mesh-adapter";
import type { MaterialLibrary } from "./materials";
import type { RenderObjectDescriptor } from "./types";

export type CachedRenderable = {
  id: number;
  root: THREE.Mesh | THREE.LineSegments;
  descriptor: RenderObjectDescriptor;
  geometryRevision: number;
  transformRevision: number;
  materialRevision: number;
  /** Why the object is drawn as an outline instead of a solid, if it is. */
  note: string | null;
};

export type SyncReport = {
  /** Geometries built in this sync. */
  meshBuilds: number;
  /** Time spent fetching and building those geometries, ms. */
  meshBuildMs: number;
  added: number;
  removed: number;
  transformed: number;
  restyled: number;
};

export class SceneRenderCache {
  readonly root = new THREE.Group();
  private readonly entries = new Map<number, CachedRenderable>();
  /** Geometries built since the cache was created. */
  meshBuilds = 0;
  /** Called after an entry's geometry or material changes (cue caches). */
  onRebuilt: (id: number) => void = () => {};

  constructor(private readonly materials: MaterialLibrary) {
    this.root.name = "components";
  }

  get size(): number {
    return this.entries.size;
  }

  get(id: number): CachedRenderable | undefined {
    return this.entries.get(id);
  }

  values(): IterableIterator<CachedRenderable> {
    return this.entries.values();
  }

  sync(descriptors: RenderObjectDescriptor[], fetchMesh: (id: number) => RenderMesh | undefined): SyncReport {
    const report: SyncReport = { meshBuilds: 0, meshBuildMs: 0, added: 0, removed: 0, transformed: 0, restyled: 0 };
    const live = new Set<number>();
    for (const d of descriptors) {
      live.add(d.id);
      let e = this.entries.get(d.id);
      if (!e || e.geometryRevision !== d.geometryRevision) {
        const t0 = performance.now();
        const next = this.build(d, fetchMesh, e);
        report.meshBuildMs += performance.now() - t0;
        if (!next) continue;
        report.meshBuilds++;
        if (!e) report.added++;
        e = next;
        this.entries.set(d.id, e);
        // A new proxy needs its placement and materials regardless.
        this.place(e, d);
        this.style(e, d);
        this.onRebuilt(d.id);
        continue;
      }
      if (e.transformRevision !== d.transformRevision) {
        this.place(e, d);
        report.transformed++;
      }
      if (e.materialRevision !== d.materialRevision) {
        this.style(e, d);
        report.restyled++;
        this.onRebuilt(d.id);
      }
      e.root.visible = d.visible;
      e.descriptor = d;
    }
    for (const [id, e] of this.entries) {
      if (live.has(id)) continue;
      this.root.remove(e.root);
      e.root.geometry.dispose();
      this.entries.delete(id);
      report.removed++;
    }
    this.meshBuilds += report.meshBuilds;
    return report;
  }

  /** Build (or rebuild) the proxy for `d`, reusing the old proxy object. */
  private build(
    d: RenderObjectDescriptor,
    fetchMesh: (id: number) => RenderMesh | undefined,
    old: CachedRenderable | undefined,
  ): CachedRenderable | null {
    const wasm = fetchMesh(d.id);
    if (!wasm) return null;
    const data = readMesh(wasm);
    const geometry = buildGeometry(data);
    const lines = data.topology === "lines";
    let root: THREE.Mesh | THREE.LineSegments;
    if (old && (old.root instanceof THREE.LineSegments) === lines) {
      old.root.geometry.dispose();
      old.root.geometry = geometry;
      root = old.root;
    } else {
      if (old) {
        this.root.remove(old.root);
        old.root.geometry.dispose();
      }
      root = lines ? new THREE.LineSegments(geometry) : new THREE.Mesh(geometry);
      root.userData.objectId = d.id;
      root.name = `${d.kind}#${d.id}`;
      this.root.add(root);
    }
    return {
      id: d.id,
      root,
      descriptor: d,
      geometryRevision: d.geometryRevision,
      transformRevision: -1,
      materialRevision: -1,
      note: data.note ?? d.error,
    };
  }

  private place(e: CachedRenderable, d: RenderObjectDescriptor) {
    e.root.position.set(d.pos[0], d.pos[1], d.pos[2]);
    e.root.quaternion.set(d.rot[0], d.rot[1], d.rot[2], d.rot[3]);
    e.root.updateMatrixWorld();
    e.transformRevision = d.transformRevision;
  }

  private style(e: CachedRenderable, d: RenderObjectDescriptor) {
    e.root.material = e.root instanceof THREE.LineSegments ? this.materials.outline() : this.materials.forObject(d);
    e.materialRevision = d.materialRevision;
    e.descriptor = d;
  }

  /** World bounds of every visible proxy. */
  bounds(): Bounds3 | null {
    const box = new THREE.Box3();
    const tmp = new THREE.Box3();
    for (const e of this.entries.values()) {
      if (!e.root.visible || !e.root.geometry.boundingBox) continue;
      tmp.copy(e.root.geometry.boundingBox).applyMatrix4(e.root.matrixWorld);
      box.union(tmp);
    }
    if (box.isEmpty()) return null;
    return {
      min: { x: box.min.x, y: box.min.y, z: box.min.z },
      max: { x: box.max.x, y: box.max.y, z: box.max.z },
    };
  }

  /** The proxies a pointer ray may hit (solids only). */
  pickables(): THREE.Object3D[] {
    const out: THREE.Object3D[] = [];
    for (const e of this.entries.values()) if (e.root.visible && e.root instanceof THREE.Mesh) out.push(e.root);
    return out;
  }

  dispose() {
    for (const e of this.entries.values()) {
      this.root.remove(e.root);
      e.root.geometry.dispose();
    }
    this.entries.clear();
  }
}
