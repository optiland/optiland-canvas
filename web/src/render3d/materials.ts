// Display materials for the 3D view: a small, fixed vocabulary of shading roles.
//
// Three.js materials provide physically plausible display shading, not optical
// simulation. In particular:
//   - Glass uses real-time screen-space transmission. What you see through a
//     lens is the opaque scene behind it, bent by a display index; it is not a
//     ray trace through the system, it ignores other glass behind it, and it
//     must never be read for focus, magnification, or any measurement.
//   - The display index is the body's n_d, clamped to what the shader accepts.
//     The clamp changes only the picture; the optical material is untouched and
//     the clamped value is never shown as a number.
//   - No display dispersion (`MeshPhysicalMaterial.dispersion` stays 0): the
//     only colour separation on screen is the tracer's, in the rays.
//   - No coating colours: clear glass stays neutral, because the scene's scalar
//     coating model has nothing a colour could honestly stand for.
//   - Mirror reflections are environment reflections, not the scene's mirror
//     reflectivity.
//
// A role comes from the Rust scene (render_view.rs); nothing here writes back.

import * as THREE from "three/webgpu";
import { abs, dot, float, normalView, positionViewDirection, vec3 } from "three/tsl";
import { FaceGroup, type MaterialRole, type RenderObjectDescriptor } from "./types";
import { FACE_GROUP_SLOTS } from "./mesh-adapter";

/** Internal shading roles on top of the scene-facing vocabulary. */
type ShadingRole = MaterialRole | "glass_edge" | "slab_edge";

/** three.js accepts an index of refraction in [1, 2.333]. */
const IOR_MIN = 1.0;
const IOR_MAX = 2.333;

export function displayIor(nd: number | null): number {
  return Math.min(IOR_MAX, Math.max(IOR_MIN, nd ?? 1.5));
}

/** A faint light at grazing view angles, so a clear surface keeps a visible
 *  outline on a dark background. A display aid; it has no optical meaning. */
function fresnelEdge(strength: number) {
  const facing = abs(dot(normalView, positionViewDirection));
  return vec3(0.75, 0.82, 0.9).mul(float(1).sub(facing).pow(3).mul(strength));
}

export class MaterialLibrary {
  private cache = new Map<string, THREE.Material>();

  private get(key: string, make: () => THREE.Material): THREE.Material {
    let m = this.cache.get(key);
    if (!m) {
      m = make();
      m.name = key;
      this.cache.set(key, m);
    }
    return m;
  }

  private glass(nd: number | null): THREE.Material {
    const ior = displayIor(nd);
    // Keyed to three decimals: enough to show a glass change, few enough
    // materials to share.
    return this.get(`glass:${ior.toFixed(3)}`, () => {
      const m = new THREE.MeshPhysicalNodeMaterial({
        color: 0xffffff,
        metalness: 0,
        roughness: 0.04,
        transmission: 1,
        ior,
        // Screen-space refraction offset only, in mm; a modest constant.
        thickness: 2,
        specularIntensity: 1,
        dispersion: 0,
      });
      m.emissiveNode = fresnelEdge(0.3);
      // Glass never hides the rays behind it; opaque parts still do.
      m.depthWrite = false;
      return m;
    });
  }

  private material(role: ShadingRole, nd: number | null): THREE.Material {
    switch (role) {
      case "optical_glass":
        return this.glass(nd);
      case "glass_edge":
        // The ground rim, drawn as a pale frosted band so the lens outline reads
        // against any background. A display aid, not a modelled surface finish.
        return this.get("glass_edge", () => {
          const m = new THREE.MeshStandardNodeMaterial({
            color: new THREE.Color(0.5, 0.54, 0.58),
            metalness: 0,
            roughness: 0.65,
            transparent: true,
            opacity: 0.7,
          });
          m.depthWrite = false;
          return m;
        });
      case "generic_dielectric":
        // A clear or refracting plate with no index in the scene: a neutral
        // thin plate at a typical index.
        return this.get("plate", () => {
          const m = new THREE.MeshPhysicalNodeMaterial({
            color: 0xffffff,
            metalness: 0,
            roughness: 0.08,
            transmission: 1,
            ior: 1.5,
            thickness: 0.5,
          });
          m.emissiveNode = fresnelEdge(0.16);
          m.depthWrite = false;
          return m;
        });
      case "mirror":
        return this.get("mirror", () => new THREE.MeshStandardNodeMaterial({
          color: new THREE.Color(0.93, 0.94, 0.96),
          metalness: 1,
          // Slightly soft, so a mirror facing a dark part of the room still
          // reads as a surface rather than a hole.
          roughness: 0.2,
        }));
      case "aperture":
        // Blackened hardware: dark, rough, a little metallic so the edge still
        // catches light and the silhouette reads.
        return this.get("aperture", () => new THREE.MeshStandardNodeMaterial({
          color: new THREE.Color(0.035, 0.037, 0.042),
          metalness: 0.3,
          roughness: 0.78,
        }));
      case "absorber":
        return this.get("absorber", () => new THREE.MeshStandardNodeMaterial({
          color: new THREE.Color(0.025, 0.025, 0.028),
          metalness: 0,
          roughness: 0.9,
        }));
      case "detector_active":
        return this.get("detector_active", () => new THREE.MeshPhysicalNodeMaterial({
          // Dark and mostly specular, like a sensor die: it reads as an active
          // area without glowing when it faces a bright part of the studio.
          color: new THREE.Color(0.04, 0.13, 0.16),
          metalness: 0.6,
          roughness: 0.38,
          // A light cover-glass sheen; stronger, and a face-on detector turns
          // into a mirror of the studio lights.
          clearcoat: 0.25,
          clearcoatRoughness: 0.2,
        }));
      case "detector_body":
        return this.get("detector_body", () => new THREE.MeshStandardNodeMaterial({
          color: new THREE.Color(0.07, 0.075, 0.085),
          metalness: 0.4,
          roughness: 0.5,
        }));
      case "slab_edge":
        // The edge of a display-only slab (a mirror or plate substrate).
        return this.get("slab_edge", () => new THREE.MeshStandardNodeMaterial({
          color: new THREE.Color(0.22, 0.23, 0.25),
          metalness: 0.1,
          roughness: 0.55,
        }));
      case "error":
        return this.get("error", () => new THREE.MeshBasicNodeMaterial({ color: 0xff7a5c }));
    }
  }

  /** The shading role for one face group of an object. */
  static roleFor(d: RenderObjectDescriptor, group: FaceGroup): ShadingRole {
    switch (group) {
      case FaceGroup.Rim:
        return d.role === "optical_glass" ? "glass_edge" : d.role;
      case FaceGroup.Front:
        return d.front ?? d.role;
      case FaceGroup.Back:
        return d.back ?? d.role;
      case FaceGroup.Edge:
        return d.kind === "aperture" ? "aperture" : "slab_edge";
      case FaceGroup.DetectorActive:
        return "detector_active";
      case FaceGroup.DetectorHousing:
        return "detector_body";
      default:
        return d.role;
    }
  }

  /** Materials for an object, indexed by face group. Shared instances. */
  forObject(d: RenderObjectDescriptor): THREE.Material[] {
    const out: THREE.Material[] = [];
    for (let g = 0; g < FACE_GROUP_SLOTS; g++) out.push(this.material(MaterialLibrary.roleFor(d, g), d.nd));
    return out;
  }

  /** Faint crease lines on clear bodies (lens rims, prism edges), so a glass
   *  body keeps a readable outline. A display aid; colour follows the
   *  background (`setCreaseTone`). */
  crease(): THREE.LineBasicNodeMaterial {
    return this.get("crease", () => {
      const m = new THREE.LineBasicNodeMaterial({ color: 0xc8d8ea, transparent: true, opacity: 0.35 });
      m.depthWrite = false;
      return m;
    }) as THREE.LineBasicNodeMaterial;
  }

  /** Crease lines light on a dark background, dark on a light one. */
  setCreaseTone(lightBackground: boolean) {
    const m = this.crease();
    m.color.setHex(lightBackground ? 0x3a4450 : 0xc8d8ea);
    m.opacity = lightBackground ? 0.45 : 0.35;
  }

  /** Line material for an outline stand-in (a body with no solid). */
  outline(): THREE.Material {
    return this.get("outline", () => new THREE.LineDashedNodeMaterial({ color: 0xff9b8a, dashSize: 2, gapSize: 1.2 }));
  }

  /** Materials created so far. */
  get size(): number {
    return this.cache.size;
  }

  dispose() {
    for (const m of this.cache.values()) m.dispose();
    this.cache.clear();
  }
}
