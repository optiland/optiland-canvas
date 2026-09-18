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
//   - Coating colours only where the scene has a coating: a beam splitter or a
//     dichroic plate shows what its reflectance and transmittance tables pass
//     across the visible band (render_view.rs weighs them). Clear glass stays
//     neutral, because a scalar Fresnel face has nothing a colour could
//     honestly stand for.
//   - Mirror reflections are environment reflections, not the scene's mirror
//     reflectivity.
//   - A source's glow is its light's display colour, not a radiance.
//   - A detector's hit map is a picture of the f64 readout's cells, painted on
//     the face they belong to; the numbers stay in the readout.
//
// A role comes from the Rust scene (render_view.rs); nothing here writes back.

import * as THREE from "three/webgpu";
import { abs, dot, float, mix, normalView, positionLocal, positionViewDirection, texture, uniform, vec2, vec3 } from "three/tsl";
import { FaceGroup, type CoatColors, type MaterialRole, type RenderObjectDescriptor } from "./types";
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

  /** A source's emitting face: lit by its own light's colour. Keyed by colour. */
  private emitter(emit: [number, number, number] | null): THREE.Material {
    const [r, g, b] = emit ?? [1, 1, 1];
    const key = `source_emitter:${r.toFixed(2)},${g.toFixed(2)},${b.toFixed(2)}`;
    return this.get(key, () => {
      const light = new THREE.Color(r, g, b);
      return new THREE.MeshStandardNodeMaterial({
        color: light.clone().multiplyScalar(0.2),
        emissive: light,
        emissiveIntensity: 1.4,
        metalness: 0,
        roughness: 0.6,
      });
    });
  }

  /** A coated plate: a splitter reads as a half-silvered window, a dichroic
   *  as clear glass tinted by what it passes, glowing faintly in what it
   *  reflects. Keyed by its colours, quantized so a dragged edge reuses
   *  materials; the colours are uniforms, so every coated plate of a role
   *  shares one shader. */
  private coated(role: "beamsplitter" | "dichroic", coat: CoatColors | null): THREE.Material {
    const quant = (c: number[] | undefined) => (c ?? [0.5, 0.5, 0.5]).map((v) => Math.round(Math.min(1, Math.max(0, v)) * 50) / 50);
    const r = quant(coat?.r);
    const t = quant(coat?.t);
    return this.get(`${role}:${r.join(",")}|${t.join(",")}`, () => {
      const mean = (c: number[]) => (c[0] + c[1] + c[2]) / 3;
      const peak = (c: number[]) => {
        const p = Math.max(c[0], c[1], c[2], 1e-6);
        return new THREE.Color(c[0] / p, c[1] / p, c[2] / p);
      };
      let m: THREE.MeshPhysicalNodeMaterial;
      if (role === "beamsplitter") {
        m = new THREE.MeshPhysicalNodeMaterial({
          color: 0xffffff,
          // The reflected share shows as metal, the rest stays see-through.
          metalness: Math.min(0.85, mean(r)),
          roughness: 0.12,
          transmission: 1,
          ior: 1.5,
          thickness: 0.5,
        });
        m.emissiveNode = fresnelEdge(0.2);
      } else {
        const reflect = peak(r);
        m = new THREE.MeshPhysicalNodeMaterial({
          color: mean(t) > 0.02 ? peak(t) : new THREE.Color(0.2, 0.2, 0.22),
          metalness: 0,
          roughness: 0.05,
          transmission: 1,
          ior: 1.5,
          thickness: 0.5,
          specularIntensity: 1,
          specularColor: reflect,
        });
        const glow = uniform(reflect.clone().multiplyScalar(Math.min(1, mean(r) * 1.5)));
        const grazing = float(1).sub(abs(dot(normalView, positionViewDirection))).pow(2);
        m.emissiveNode = glow.mul(grazing.mul(0.5).add(0.12));
      }
      m.depthWrite = false;
      return m;
    });
  }

  private material(role: ShadingRole, d: RenderObjectDescriptor, group: FaceGroup): THREE.Material {
    switch (role) {
      case "optical_glass":
        return this.glass(d.nd);
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
      case "source_emitter":
        return this.emitter(d.emit);
      case "source_housing":
        // Anodized lamp housing: dark, satin, a little metallic.
        return this.get("source_housing", () => new THREE.MeshStandardNodeMaterial({
          color: new THREE.Color(0.1, 0.105, 0.12),
          metalness: 0.5,
          roughness: 0.45,
        }));
      case "beamsplitter":
      case "dichroic":
        return this.coated(role, group === FaceGroup.Back ? d.coat?.back ?? null : d.coat?.front ?? null);
      case "polarizing":
        // A polarizer or a waveplate: a smoky plate, so it reads as an element
        // that does something to the light rather than as a clear window. The
        // f32 preview cannot show what it does, so the plate saying "I am not a
        // window" is the whole of what the 3D view can honestly convey.
        return this.get("polarizing", () => {
          const m = new THREE.MeshPhysicalNodeMaterial({
            color: new THREE.Color(0.30, 0.31, 0.38),
            metalness: 0,
            roughness: 0.25,
            transmission: 0.55,
            ior: 1.5,
            thickness: 0.4,
            transparent: true,
            opacity: 0.85,
          });
          m.depthWrite = false;
          return m;
        });
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
      case FaceGroup.SourceEmitter:
        return "source_emitter";
      case FaceGroup.SourceHousing:
        return "source_housing";
      default:
        return d.role;
    }
  }

  /** Materials for an object, indexed by face group. Shared instances. Only
   *  the groups an object's kind can have are resolved; the rest are unused
   *  slots and get a shared placeholder. */
  forObject(d: RenderObjectDescriptor): THREE.Material[] {
    const out: THREE.Material[] = [];
    for (let g = 0; g < FACE_GROUP_SLOTS; g++) {
      const used = d.kind === "source" ? g >= FaceGroup.SourceEmitter : g < FaceGroup.SourceEmitter;
      out.push(used ? this.material(MaterialLibrary.roleFor(d, g), d, g) : this.material("error", d, g));
    }
    return out;
  }

  /** A detector face showing a hit map: the detector's own finish, with the
   *  map's cells lit on top. One per detector (it holds that detector's
   *  texture); the caller owns and disposes it. `visible` fades the map without
   *  a new shader. */
  static hitMap(map: THREE.DataTexture, halfW: number, halfH: number) {
    const half = uniform(new THREE.Vector2(halfW, halfH));
    const visible = uniform(1);
    const m = new THREE.MeshPhysicalNodeMaterial({
      metalness: 0.6,
      roughness: 0.38,
      clearcoat: 0.25,
      clearcoatRoughness: 0.2,
    });
    // The face is the local XY plane, so a fragment's local (x, y) is its
    // detector (u, v), the coordinates the cells are binned in.
    const uv = positionLocal.xy.div(half.mul(2)).add(0.5);
    const cell = texture(map, vec2(uv.x, uv.y));
    const lit = cell.a.mul(visible);
    m.colorNode = mix(vec3(0.04, 0.13, 0.16), vec3(0.02, 0.02, 0.025), lit);
    m.emissiveNode = cell.rgb.mul(lit).mul(1.3);
    m.name = "detector_hit_map";
    return { material: m, half, visible };
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
