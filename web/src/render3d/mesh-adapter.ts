// Display meshes from Rust, as three.js geometry.
//
// The mesh is `oc_core::display_mesh`: a tessellation of the analytic geometry
// the tracer uses, in the object's local frame and in millimetres. Nothing here
// computes a shape; it only copies arrays and tags index ranges with their face
// group so each range can be shaded by its own display role.

import * as THREE from "three/webgpu";
import type { RenderMesh } from "../../pkg/oc_wasm.js";
import type { RenderMeshData } from "./types";

/** Copy a wasm mesh into JS arrays and free the wasm object. */
export function readMesh(mesh: RenderMesh): RenderMeshData & { note: string | null } {
  try {
    return {
      topology: mesh.topology() === "lines" ? "lines" : "triangles",
      positions: mesh.positions(),
      normals: mesh.normals(),
      indices: mesh.indices(),
      groups: mesh.groups(),
      bounds: mesh.bounds(),
      note: mesh.note() ?? null,
    };
  } finally {
    mesh.free();
  }
}

/** Number of face-group slots; a material array is indexed by group code. */
export const FACE_GROUP_SLOTS = 7;

export function buildGeometry(data: RenderMeshData): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
  if (data.topology === "triangles") g.setAttribute("normal", new THREE.BufferAttribute(data.normals, 3));
  g.setIndex(new THREE.BufferAttribute(data.indices, 1));
  for (let i = 0; i + 2 < data.groups.length; i += 3) {
    g.addGroup(data.groups[i], data.groups[i + 1], data.groups[i + 2]);
  }
  // Bounds come from Rust, so three.js does not walk the vertices for them.
  const b = data.bounds;
  g.boundingBox = new THREE.Box3(new THREE.Vector3(b[0], b[1], b[2]), new THREE.Vector3(b[3], b[4], b[5]));
  g.boundingSphere = g.boundingBox.getBoundingSphere(new THREE.Sphere());
  return g;
}
