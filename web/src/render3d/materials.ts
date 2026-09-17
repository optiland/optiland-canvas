// Display materials for the 3D view: a small, fixed vocabulary of shading roles.
//
// These are shading choices, not optical materials. A role comes from the Rust
// scene (glass, mirror, absorber, detector); nothing here writes back to it.
//
// Phase 2 (mesh bridge): flat, opaque debug materials so the geometry can be
// checked before any appearance work.

import * as THREE from "three/webgpu";
import { FaceGroup, type MaterialRole, type RenderObjectDescriptor } from "./types";
import { FACE_GROUP_SLOTS } from "./mesh-adapter";

const DEBUG_COLORS: Record<MaterialRole, number> = {
  optical_glass: 0x8fb8e8,
  mirror: 0xd0d8e4,
  aperture: 0x3a3a40,
  absorber: 0x2a2a2e,
  detector_active: 0x2f7f8f,
  detector_body: 0x30343a,
  generic_dielectric: 0xa0b4c8,
  error: 0xff7a5c,
};

export class MaterialLibrary {
  private cache = new Map<string, THREE.Material>();

  private material(role: MaterialRole): THREE.Material {
    let m = this.cache.get(role);
    if (!m) {
      m = new THREE.MeshStandardNodeMaterial({ color: DEBUG_COLORS[role], roughness: 0.6, metalness: 0 });
      m.name = role;
      this.cache.set(role, m);
    }
    return m;
  }

  /** The display role for one face group of an object. */
  static roleFor(d: RenderObjectDescriptor, group: FaceGroup): MaterialRole {
    switch (group) {
      case FaceGroup.Front:
        return d.front ?? d.role;
      case FaceGroup.Back:
        return d.back ?? d.role;
      case FaceGroup.DetectorActive:
        return "detector_active";
      case FaceGroup.DetectorHousing:
        return "detector_body";
      case FaceGroup.Edge:
        return d.kind === "aperture" ? "aperture" : d.front ?? d.role;
      default:
        return d.role;
    }
  }

  /** Materials for an object, indexed by face group. Shared instances. */
  forObject(d: RenderObjectDescriptor): THREE.Material[] {
    const out: THREE.Material[] = [];
    for (let g = 0; g < FACE_GROUP_SLOTS; g++) out.push(this.material(MaterialLibrary.roleFor(d, g)));
    return out;
  }

  /** Line material for an outline stand-in (a body with no solid). */
  outline(): THREE.Material {
    let m = this.cache.get("outline");
    if (!m) {
      m = new THREE.LineBasicNodeMaterial({ color: DEBUG_COLORS.error });
      this.cache.set("outline", m);
    }
    return m;
  }

  dispose() {
    for (const m of this.cache.values()) m.dispose();
    this.cache.clear();
  }
}
