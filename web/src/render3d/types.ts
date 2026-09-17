// Shared types for the 3D view (no three.js here).
//
// The render descriptors mirror what `oc-wasm` returns from `renderObjectsJSON`
// (crates/oc-wasm/src/render_view.rs). A descriptor is "the visual
// representation of Rust object N at revision M", never the optical definition
// of object N: the canonical scene stays in Rust.

export type { RenderQualityState } from "./quality";
export type { CameraState, CameraPreset } from "./camera";

/** The visual material vocabulary. Deliberately small: these are display
 *  shading roles, not optical materials. */
export type MaterialRole =
  | "optical_glass"
  | "mirror"
  | "aperture"
  | "absorber"
  | "detector_active"
  | "detector_body"
  | "source_emitter"
  | "source_housing"
  | "generic_dielectric"
  | "error";

/** What kind of display object a descriptor stands for. */
export type RenderObjectKind =
  | "lens" // a revolved profile body
  | "prism" // an extruded profile body
  | "solid" // sphere, box, cylinder
  | "planar_only" // a 2D-only body: no third dimension exists, drawn as an outline
  | "surface" // a zero-thickness surface object (mirror, plate)
  | "aperture"
  | "detector"
  | "source"; // a light source's glyph (not an object the tracer hits)

export type RenderObjectDescriptor = {
  id: number;
  kind: RenderObjectKind;
  /** Bumped when the mesh must change. */
  geometryRevision: number;
  /** Bumped when only the placement changes. */
  transformRevision: number;
  /** Bumped when only the visual material changes. */
  materialRevision: number;
  visible: boolean;
  /** World position, mm. */
  pos: [number, number, number];
  /** World rotation quaternion [x, y, z, w]. */
  rot: [number, number, number, number];
  /** The body-level role (glass, mirror, absorber, ...). */
  role: MaterialRole;
  /** Display roles for the front (+Z) and back (-Z) faces of a surface object. */
  front: MaterialRole | null;
  back: MaterialRole | null;
  /** The body's d-line index, when it has one. Display IOR only. */
  nd: number | null;
  /** A coarse radius for proximity picking, mm. */
  radius: number;
  /** Why no solid mesh exists (a 2D-only body), or null. */
  error: string | null;
  /** A source's light colour (linear RGB, brightest channel 1), or null. */
  emit: [number, number, number] | null;
};

/** Mesh group codes, as `oc-wasm` numbers them. A group is a run of indices
 *  that shares a display role within one object. */
export const enum FaceGroup {
  /** A body's optical surfaces (lens faces, prism walls and caps). */
  Optical = 0,
  /** A lens's ground rim, or the sides of an analytic solid. */
  Rim = 1,
  /** A surface object's front (+Z) face. */
  Front = 2,
  /** A surface object's back (-Z) face. */
  Back = 3,
  /** The edge band of a display-only slab. */
  Edge = 4,
  /** A detector's active face. */
  DetectorActive = 5,
  /** A detector's housing (back and edges). */
  DetectorHousing = 6,
  /** A light source's emitting aperture. */
  SourceEmitter = 7,
  /** A light source's display housing. */
  SourceHousing = 8,
}

export type MeshTopology = "triangles" | "lines";

/** Mesh data in the object's local frame, mm. */
export type RenderMeshData = {
  topology: MeshTopology;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** Triples [start, count, FaceGroup]. */
  groups: Uint32Array;
  /** Crease lines as vertex-index pairs (rims, corners, outlines). */
  edges: Uint32Array;
  /** [minx, miny, minz, maxx, maxy, maxz], local frame. */
  bounds: Float32Array;
};

export type RayVisibilityMode = "through" | "occluded";

/** One area detector's incident-flux map, from the exact trace
 *  (`lastDetectorMapsF32`). Cells are row-major from local (-halfW, -halfH). */
export type DetectorMap = {
  id: number;
  nu: number;
  nv: number;
  halfW: number;
  halfH: number;
  cells: Float32Array;
};

/** Split the flat `lastDetectorMapsF32` layout into one map per detector. */
export function parseDetectorMaps(flat: Float32Array): DetectorMap[] {
  const out: DetectorMap[] = [];
  const count = flat.length > 0 ? flat[0] : 0;
  let k = 1;
  for (let i = 0; i < count && k + 5 <= flat.length; i++) {
    const nu = flat[k + 1], nv = flat[k + 2];
    const n = nu * nv;
    if (k + 5 + n > flat.length) break;
    out.push({ id: flat[k], nu, nv, halfW: flat[k + 3], halfH: flat[k + 4], cells: flat.subarray(k + 5, k + 5 + n) });
    k += 5 + n;
  }
  return out;
}

export type EnvironmentId = "dark_studio" | "light_studio" | "optical_bench" | "inspection";

export type BackgroundId = "match" | "black" | "charcoal" | "mid_gray" | "light_gray";

/** Which graphics backend presents the 3D view. Distinct from the ray-preview
 *  backend (fastpreview.ts), which decides where the f32 trace runs. */
export type RenderBackend = "loading" | "webgpu" | "webgl2" | "unavailable";

/** Developer diagnostics for one rendered frame. */
export type RenderStats = {
  backend: RenderBackend;
  quality: "interactive" | "settled";
  dpr: number;
  frames: number;
  /** CPU time of the last frame's render call (submission, not GPU work). */
  renderMs: number;
  /** GPU time of the last resolved frame, when timestamp queries exist. */
  gpuMs: number | null;
  drawCalls: number;
  triangles: number;
  lines: number;
  raySegments: number;
  /** Geometry builds since start, and in the last scene sync. */
  meshBuilds: number;
  meshBuildsLastSync: number;
  meshBuildMs: number;
  /** Time to fetch descriptors and meshes across the wasm boundary. */
  syncMs: number;
  rayUploadMs: number;
  /** Ray buffer uploads since start. */
  rayUploads: number;
  /** Ray line width, CSS px. */
  rayWidth: number;
  /** Detectors showing a hit map. */
  hitMaps: number;
  environmentBuilds: number;
  geometries: number;
  textures: number;
  cachedObjects: number;
  /** The last frame, broken down (ms). */
  frame: FrameCost;
  /** JS heap in use, where the browser reports it (bytes). */
  jsHeap: number | null;
};

export type FrameCost = {
  total: number;
  trace: number;
  sync: number;
  upload: number;
  render: number;
  overlay: number;
};
