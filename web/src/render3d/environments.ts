// Lighting environments and backgrounds for the 3D view.
//
// Each preset is a small procedural studio (a box room with soft emissive
// panels), prefiltered once into a PMREM for image-based lighting and cached.
// Switching presets swaps a cached texture; it never rebuilds geometry, and
// never regenerates a PMREM that already exists. No image files are loaded.
//
// The background is chosen independently of the lighting.
//
// All of this is display lighting. It makes glass edges, mirror reflections,
// and curvature readable. It has no relation to the optical source in the scene.

import * as THREE from "three/webgpu";
import type { ScaffoldTheme } from "./scaffold";
import type { BackgroundId, EnvironmentId } from "./types";

type Panel = {
  pos: [number, number, number];
  scale: [number, number, number];
  intensity: number;
};

type RoomLayout = {
  /** Wall albedo (grey level). */
  wall: number;
  /** Point light in the room (lights the walls and boxes). */
  lamp: number;
  panels: Panel[];
  /** Blocks in the room that break up the reflections. */
  boxes: boolean;
};

export type EnvironmentPreset = {
  id: EnvironmentId;
  label: string;
  room: RoomLayout;
  /** Strength of the image-based lighting. */
  intensity: number;
  /** The background when "match environment" is chosen. */
  background: number;
  /** Extra direct lights, as [colour, intensity, direction toward the light]. */
  lights: [number, number, [number, number, number]][];
  scaffold: { grid: number; gridOpacity: number; ground: number | null };
};

export const ENVIRONMENTS: EnvironmentPreset[] = [
  {
    // The app's dark identity: a dim room, soft panels, restrained highlights.
    id: "dark_studio",
    label: "Dark studio",
    room: {
      wall: 0.22,
      lamp: 500,
      boxes: true,
      panels: [
        { pos: [0, 20, 0], scale: [10, 0.1, 10], intensity: 60 },
        { pos: [-16, 14, 6], scale: [0.1, 8, 10], intensity: 40 },
        { pos: [15, 12, -3], scale: [0.1, 6, 8], intensity: 18 },
        { pos: [0, 9, 14.5], scale: [8, 6, 0.1], intensity: 26 },
      ],
    },
    intensity: 1.0,
    background: 0x07090e,
    lights: [],
    scaffold: { grid: 0x7896bf, gridOpacity: 0.32, ground: null },
  },
  {
    // A bright, soft studio. Ray colours switch to their light-background style.
    id: "light_studio",
    label: "Light studio",
    room: {
      wall: 0.75,
      lamp: 900,
      boxes: false,
      panels: [
        { pos: [0, 20, 0], scale: [14, 0.1, 14], intensity: 12 },
        { pos: [-16, 12, 0], scale: [0.1, 10, 18], intensity: 7 },
        { pos: [0, 10, 14.5], scale: [18, 10, 0.1], intensity: 6 },
      ],
    },
    intensity: 1.0,
    background: 0xd9dbde,
    lights: [[0xffffff, 0.6, [-1, 2, 1.5]]],
    scaffold: { grid: 0x5d6673, gridOpacity: 0.28, ground: null },
  },
  {
    // A dark bench under an overhead strip, with a visible ground.
    id: "optical_bench",
    label: "Optical bench",
    room: {
      wall: 0.16,
      lamp: 250,
      boxes: false,
      panels: [
        { pos: [0, 20, 0], scale: [22, 0.1, 3], intensity: 55 },
        { pos: [-16, 10, 0], scale: [0.1, 3, 10], intensity: 5 },
        { pos: [15, 10, 0], scale: [0.1, 3, 10], intensity: 5 },
      ],
    },
    intensity: 0.8,
    background: 0x141619,
    lights: [[0xfff4e6, 1.6, [0.15, 1, 0.35]]],
    scaffold: { grid: 0x8a8f96, gridOpacity: 0.22, ground: 0x1b1d21 },
  },
  {
    // Shape reading: narrow bright strips on dark walls, so curvature and tilt
    // show up as bent, broken highlight lines.
    id: "inspection",
    label: "Inspection",
    room: {
      wall: 0.05,
      lamp: 60,
      boxes: false,
      panels: [
        { pos: [-16, 12, -9], scale: [0.1, 18, 0.9], intensity: 70 },
        { pos: [-16, 12, -3], scale: [0.1, 18, 0.9], intensity: 70 },
        { pos: [-16, 12, 3], scale: [0.1, 18, 0.9], intensity: 70 },
        { pos: [-16, 12, 9], scale: [0.1, 18, 0.9], intensity: 70 },
        { pos: [15, 12, -6], scale: [0.1, 18, 0.9], intensity: 50 },
        { pos: [15, 12, 0], scale: [0.1, 18, 0.9], intensity: 50 },
        { pos: [15, 12, 6], scale: [0.1, 18, 0.9], intensity: 50 },
        { pos: [0, 20, 0], scale: [20, 0.1, 0.9], intensity: 60 },
      ],
    },
    intensity: 1.1,
    background: 0x6f7378,
    lights: [[0xffffff, 1.2, [-1, 0.6, -1]]],
    scaffold: { grid: 0x2a2e33, gridOpacity: 0.3, ground: null },
  },
];

export const BACKGROUNDS: { id: BackgroundId; label: string; color: number | null }[] = [
  { id: "match", label: "Match environment", color: null },
  { id: "black", label: "Black", color: 0x000000 },
  { id: "charcoal", label: "Dark charcoal", color: 0x1a1c20 },
  { id: "mid_gray", label: "Medium grey", color: 0x6f7378 },
  { id: "light_gray", label: "Light grey", color: 0xd9dbde },
];

export function environmentPreset(id: EnvironmentId): EnvironmentPreset {
  return ENVIRONMENTS.find((e) => e.id === id) ?? ENVIRONMENTS[0];
}

/** Relative luminance of an sRGB hex colour, 0..1. */
export function luminance(hex: number): number {
  const c = new THREE.Color(hex); // converted to linear working space
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

/** A procedural studio room for PMREM capture. */
function buildRoom(layout: RoomLayout): { scene: THREE.Scene; dispose: () => void } {
  const scene = new THREE.Scene();
  scene.position.y = -3.5;
  const box = new THREE.BoxGeometry();
  box.deleteAttribute("uv");
  const materials: THREE.Material[] = [];
  const wall = new THREE.MeshStandardNodeMaterial({ color: new THREE.Color().setScalar(layout.wall), side: THREE.BackSide });
  materials.push(wall);
  const room = new THREE.Mesh(box, wall);
  room.position.set(-0.757, 13.219, 0.717);
  room.scale.set(31.713, 28.305, 28.591);
  scene.add(room);
  const lamp = new THREE.PointLight(0xffffff, layout.lamp, 28, 2);
  lamp.position.set(0.418, 16.199, 0.3);
  scene.add(lamp);
  if (layout.boxes) {
    const bm = new THREE.MeshStandardNodeMaterial({ color: new THREE.Color().setScalar(Math.min(1, layout.wall * 3)) });
    materials.push(bm);
    for (const [p, r, s] of [
      [[-10.9, 2.0, 1.8], -0.2, [2.3, 7.9, 4.7]],
      [[6.2, 0.9, 7.8], 0.56, [3.9, 6.3, 3.7]],
      [[-2.2, -0.4, -5.5], 0.52, [3.9, 3.5, 3.0]],
    ] as [[number, number, number], number, [number, number, number]][]) {
      const m = new THREE.Mesh(box, bm);
      m.position.set(...p);
      m.rotation.y = r;
      m.scale.set(...s);
      scene.add(m);
    }
  }
  for (const panel of layout.panels) {
    // An emissive-only surface: black diffuse, bright emission.
    const pm = new THREE.MeshBasicNodeMaterial({ color: new THREE.Color().setScalar(panel.intensity) });
    materials.push(pm);
    const m = new THREE.Mesh(box, pm);
    m.position.set(...panel.pos);
    m.scale.set(...panel.scale);
    scene.add(m);
  }
  return {
    scene,
    dispose: () => {
      box.dispose();
      for (const m of materials) m.dispose();
    },
  };
}

/** Builds, caches, and applies environments to a scene. */
export class EnvironmentManager {
  private readonly pmrem: THREE.PMREMGenerator;
  private readonly cache = new Map<EnvironmentId, THREE.RenderTarget>();
  private readonly lights = new THREE.Group();
  /** PMREMs generated since start (each preset at most once). */
  builds = 0;
  current: EnvironmentPreset = ENVIRONMENTS[0];
  background: BackgroundId = "match";

  constructor(
    renderer: THREE.WebGPURenderer,
    private readonly scene: THREE.Scene,
  ) {
    this.pmrem = new THREE.PMREMGenerator(renderer);
    this.lights.name = "environment-lights";
    scene.add(this.lights);
  }

  setEnvironment(id: EnvironmentId) {
    const preset = environmentPreset(id);
    let target = this.cache.get(preset.id);
    if (!target) {
      const room = buildRoom(preset.room);
      target = this.pmrem.fromScene(room.scene, 0.04);
      room.dispose();
      this.cache.set(preset.id, target);
      this.builds++;
    }
    this.current = preset;
    this.scene.environment = target.texture;
    this.scene.environmentIntensity = preset.intensity;

    this.clearLights();
    for (const [color, intensity, dir] of preset.lights) {
      // Aimed at the world origin (the default target).
      const light = new THREE.DirectionalLight(color, intensity);
      light.position.set(...dir).normalize().multiplyScalar(100);
      this.lights.add(light);
    }
    this.applyBackground();
  }

  setBackground(id: BackgroundId) {
    this.background = id;
    this.applyBackground();
  }

  /** The background colour in effect, as sRGB hex. */
  backgroundHex(): number {
    const choice = BACKGROUNDS.find((b) => b.id === this.background);
    return choice?.color ?? this.current.background;
  }

  private clearLights() {
    for (const l of [...this.lights.children]) {
      this.lights.remove(l);
      if (l instanceof THREE.Light) l.dispose();
    }
  }

  private applyBackground() {
    const hex = this.backgroundHex();
    if (this.scene.background instanceof THREE.Color) this.scene.background.setHex(hex);
    else this.scene.background = new THREE.Color(hex);
  }

  /** The grid and ground look for the current environment and background. */
  scaffoldTheme(): ScaffoldTheme {
    const s = this.current.scaffold;
    // A grid drawn for a dark room would vanish on a light background, and the
    // other way round; follow the background actually shown.
    const lightBg = luminance(this.backgroundHex()) > 0.25;
    const grid = lightBg && luminance(s.grid) > 0.2 ? 0x4a525d : !lightBg && luminance(s.grid) < 0.05 ? 0x7896bf : s.grid;
    return {
      grid: new THREE.Color(grid),
      gridOpacity: s.gridOpacity,
      ground: s.ground === null ? null : new THREE.Color(s.ground),
    };
  }

  dispose() {
    for (const t of this.cache.values()) t.dispose();
    this.cache.clear();
    this.pmrem.dispose();
    this.clearLights();
  }
}
