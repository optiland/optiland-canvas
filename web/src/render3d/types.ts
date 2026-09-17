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
  | "detector";

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
  /** [minx, miny, minz, maxx, maxy, maxz], local frame. */
  bounds: Float32Array;
};

export type RayVisibilityMode = "through" | "occluded";

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
  environmentBuilds: number;
  geometries: number;
  textures: number;
  cachedObjects: number;
};
