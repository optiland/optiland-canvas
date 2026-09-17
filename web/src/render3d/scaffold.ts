// The 3D view's scaffold: world axes, a ground plane, and a grid on it.
//
// Display aids only. The grid spacing is the shared grid control's, so a drag
// that snaps to 5 mm lands on a line drawn 5 mm apart. The ground sits below the
// lowest component rather than through the optical axis, so lenses stand on it
// visually instead of being sliced by it. It has no optical meaning.

import * as THREE from "three/webgpu";
import { attribute, float, length, positionLocal, smoothstep, uniform } from "three/tsl";
import type { Bounds3 } from "./camera";

export type ScaffoldTheme = {
  /** Grid line colour and opacity. */
  grid: THREE.Color;
  gridOpacity: number;
  /** Ground plane colour, or null for no ground plane in this environment. */
  ground: THREE.Color | null;
};

const colorUniform = (c: THREE.Color) => uniform(c);
const floatUniform = (v: number) => uniform(v);
type ColorUniform = ReturnType<typeof colorUniform>;
type FloatUniform = ReturnType<typeof floatUniform>;

const AXIS_COLORS: [number, number, number][] = [
  [0.59, 0.69, 0.82], // +X, the optical axis (labelled "Z")
  [0.59, 0.82, 0.69], // +Y, vertical ("Y")
  [0.82, 0.69, 0.59], // +Z, out of plane ("X")
];

export class Scaffold {
  readonly root = new THREE.Group();
  private axes: THREE.LineSegments;
  private grid: THREE.LineSegments | null = null;
  private ground: THREE.Mesh;
  private gridColor = colorUniform(new THREE.Color(0.47, 0.59, 0.75));
  private gridOpacity = floatUniform(0.35);
  private gridExtent = floatUniform(100);
  private gridKey = "";
  private groundMaterial: THREE.MeshStandardNodeMaterial;
  /** Length of the drawn axes, mm. */
  axisLength = 18;
  groundY = 0;

  constructor() {
    this.root.name = "scaffold";
    this.axes = makeAxes();
    this.root.add(this.axes);

    this.groundMaterial = new THREE.MeshStandardNodeMaterial({ roughness: 0.92, metalness: 0 });
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.groundMaterial);
    this.ground.name = "ground";
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.renderOrder = -2;
    this.ground.visible = false;
    this.root.add(this.ground);
  }

  /** Place the ground, grid, and axes for an assembly that occupies `bounds`,
   *  centred near `centre`. `spacing` is the grid pitch in mm. */
  layout(bounds: Bounds3 | null, spacing: number, gridOn: boolean, groundOn: boolean, theme: ScaffoldTheme) {
    const lo = bounds?.min ?? { x: -40, y: -20, z: -20 };
    const hi = bounds?.max ?? { x: 40, y: 20, z: 20 };
    const size = Math.max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 20);
    const cx = (lo.x + hi.x) / 2, cz = (lo.z + hi.z) / 2;
    // Below the lowest point, with a margin that grows with the assembly.
    this.groundY = lo.y - Math.max(2, 0.15 * size);
    this.axisLength = Math.min(200, Math.max(18, 0.2 * size));
    this.axes.scale.setScalar(this.axisLength);

    const sp = Math.max(0.1, spacing);
    // A patch that covers the assembly generously, capped so a fine pitch on a
    // large scene does not become a million lines.
    let half = Math.ceil((0.9 * size + 20) / sp);
    half = Math.max(8, Math.min(200, half));
    const ox = Math.round(cx / sp) * sp, oz = Math.round(cz / sp) * sp;
    const key = `${sp}|${half}|${ox}|${oz}`;
    if (key !== this.gridKey) {
      this.gridKey = key;
      if (this.grid) {
        this.root.remove(this.grid);
        this.grid.geometry.dispose();
        (this.grid.material as THREE.Material).dispose();
      }
      this.grid = makeGrid(sp, half, ox, oz, this.gridColor, this.gridOpacity, this.gridExtent);
      this.root.add(this.grid);
    }
    this.gridExtent.value = half * sp;
    this.grid!.position.y = this.groundY;
    // A small lift keeps the lines off the ground plane without z-fighting;
    // polygon offset alone is not reliable across both backends for lines.
    this.grid!.position.y += Math.max(0.01, 1e-4 * size);
    this.grid!.visible = gridOn;

    this.gridColor.value.copy(theme.grid);
    this.gridOpacity.value = theme.gridOpacity;

    const extent = 2 * half * sp + 4 * size;
    this.ground.scale.set(extent, extent, 1);
    this.ground.position.set(ox, this.groundY, oz);
    this.ground.visible = groundOn && theme.ground !== null;
    if (theme.ground) this.groundMaterial.color.copy(theme.ground);
  }

  /** World-space end points of the three axes, for the overlay labels. */
  axisEnds(): { x: number; y: number; z: number }[] {
    const L = this.axisLength;
    return [
      { x: L, y: 0, z: 0 },
      { x: 0, y: L, z: 0 },
      { x: 0, y: 0, z: L },
    ];
  }

  dispose() {
    this.axes.geometry.dispose();
    (this.axes.material as THREE.Material).dispose();
    if (this.grid) {
      this.grid.geometry.dispose();
      (this.grid.material as THREE.Material).dispose();
    }
    this.ground.geometry.dispose();
    this.groundMaterial.dispose();
  }
}

function makeAxes(): THREE.LineSegments {
  const pos = new Float32Array(18);
  const col = new Float32Array(18);
  for (let a = 0; a < 3; a++) {
    pos[a * 6 + 3 + a] = 1; // unit axis, scaled by the object
    for (let v = 0; v < 2; v++) col.set(AXIS_COLORS[a], a * 6 + v * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const m = new THREE.LineBasicNodeMaterial({ vertexColors: true, depthTest: false, depthWrite: false, transparent: true });
  const axes = new THREE.LineSegments(g, m);
  axes.name = "axes";
  axes.renderOrder = 20;
  axes.frustumCulled = false;
  return axes;
}

function makeGrid(
  sp: number,
  half: number,
  ox: number,
  oz: number,
  color: ColorUniform,
  opacity: FloatUniform,
  extent: FloatUniform,
): THREE.LineSegments {
  const n = 2 * half + 1;
  const pos = new Float32Array(n * 2 * 6);
  const emph = new Float32Array(n * 2 * 2);
  let k = 0, e = 0;
  const L = half * sp;
  for (let i = -half; i <= half; i++) {
    const x = ox + i * sp;
    const z = oz + i * sp;
    // Lines along Z (constant x), then along X (constant z), in grid-local
    // coordinates relative to (ox, oz) so the shader fade is centred.
    pos.set([i * sp, 0, -L, i * sp, 0, L], k); k += 6;
    pos.set([-L, 0, i * sp, L, 0, i * sp], k); k += 6;
    // The world axes read brighter, as in the 2D grid.
    const onX = Math.abs(x) < 1e-9 ? 1 : 0;
    const onZ = Math.abs(z) < 1e-9 ? 1 : 0;
    emph.set([onX, onX], e); e += 2;
    emph.set([onZ, onZ], e); e += 2;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("emph", new THREE.BufferAttribute(emph, 1));
  const m = new THREE.LineBasicNodeMaterial({ transparent: true, depthWrite: false });
  // Fade with distance from the patch centre so the grid never ends in a hard
  // edge; the world axis lines are brighter.
  const fade = smoothstep(extent, extent.mul(0.35), length(positionLocal.xz));
  const emphasis = float(1).add(attribute("emph", "float").mul(1.6));
  m.colorNode = color;
  m.opacityNode = fade.mul(opacity).mul(emphasis).min(1);
  const grid = new THREE.LineSegments(g, m);
  grid.name = "grid";
  grid.position.set(ox, 0, oz);
  grid.renderOrder = -1;
  grid.frustumCulled = false;
  return grid;
}
