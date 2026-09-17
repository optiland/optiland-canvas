// The 3D view's scaffold: world axes, a ground plane, and a grid on it.
//
// Display aids only. The grid spacing is the shared grid control's, so a drag
// that snaps to 5 mm lands on a line drawn 5 mm apart. The ground sits below the
// lowest component rather than through the optical axis, so lenses stand on it
// visually instead of being sliced by it. It has no optical meaning.
//
// The grid's material is built once. Its geometry depends only on the pitch and
// the patch size (rounded to a power of two), so moving the assembly moves the
// grid without rebuilding anything.

import * as THREE from "three/webgpu";
import { abs, attribute, float, length, mix, positionLocal, positionWorld, smoothstep, step, uniform, vertexColor } from "three/tsl";
import type { Bounds3 } from "./camera";

export type ScaffoldTheme = {
  /** Grid line colour and opacity. */
  grid: THREE.Color;
  gridOpacity: number;
  /** Ground plane colour, or null for no ground plane in this environment. */
  ground: THREE.Color | null;
  /** The background shown is light (darker axis lines read on it). */
  lightBackground: boolean;
};

const AXIS_COLORS: [number, number, number][] = [
  [0.59, 0.69, 0.82], // +X, the optical axis (labelled "Z")
  [0.59, 0.82, 0.69], // +Y, vertical ("Y")
  [0.82, 0.69, 0.59], // +Z, out of plane ("X")
];

export class Scaffold {
  readonly root = new THREE.Group();
  private axes: THREE.LineSegments;
  private grid: THREE.LineSegments;
  private ground: THREE.Mesh;
  private gridColor = uniform(new THREE.Color(0.47, 0.59, 0.75));
  private gridOpacity = uniform(0.35);
  private gridExtent = uniform(100);
  private gridPitch = uniform(5);
  /** Axis line brightness: full on dark grounds, darkened on light ones. */
  private axisTone = uniform(1);
  private gridKey = "";
  private gridMaterial: THREE.LineBasicNodeMaterial;
  private groundMaterial: THREE.MeshStandardNodeMaterial;
  /** Grid geometries built since start (diagnostics). */
  gridBuilds = 0;
  /** Length of the drawn axes, mm. */
  axisLength = 18;
  groundY = 0;

  constructor() {
    this.root.name = "scaffold";
    this.axes = makeAxes();
    (this.axes.material as THREE.LineBasicNodeMaterial).colorNode = vertexColor().rgb.mul(this.axisTone);
    this.root.add(this.axes);

    this.gridMaterial = new THREE.LineBasicNodeMaterial({ transparent: true, depthWrite: false });
    // Fade with distance from the patch centre so the grid never ends in a hard
    // edge. A line lying on a world axis (x = 0 for lines along z, z = 0 for
    // lines along x) is drawn brighter, as in the 2D grid.
    const fade = smoothstep(this.gridExtent, this.gridExtent.mul(0.35), length(positionLocal.xz));
    const offAxis = mix(abs(positionWorld.x), abs(positionWorld.z), attribute("alongX", "float"));
    const onAxis = step(offAxis, this.gridPitch.mul(0.01));
    this.gridMaterial.colorNode = this.gridColor;
    this.gridMaterial.opacityNode = fade.mul(this.gridOpacity).mul(float(1).add(onAxis.mul(1.6))).min(1);
    this.gridMaterial.name = "grid";
    this.grid = new THREE.LineSegments(new THREE.BufferGeometry(), this.gridMaterial);
    this.grid.name = "grid";
    this.grid.renderOrder = -1;
    this.grid.frustumCulled = false;
    this.root.add(this.grid);

    // The ground fades out toward its edge, so it never ends in a visible line.
    this.groundMaterial = new THREE.MeshStandardNodeMaterial({ roughness: 0.92, metalness: 0, transparent: true });
    this.groundMaterial.depthWrite = false;
    this.groundMaterial.opacityNode = float(1).sub(smoothstep(float(0.08), float(0.5), length(positionLocal.xy)));
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.groundMaterial);
    this.ground.name = "ground";
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.renderOrder = -2;
    this.ground.visible = false;
    this.root.add(this.ground);
  }

  /** Place the ground, grid, and axes for an assembly that occupies `bounds`.
   *  `spacing` is the grid pitch in mm. */
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
    // large scene does not become a million lines. A power of two, so small
    // changes in the assembly's size do not rebuild the grid.
    const need = Math.ceil((0.9 * size + 20) / sp);
    const half = Math.max(8, Math.min(256, 2 ** Math.ceil(Math.log2(need))));
    const key = `${sp}|${half}`;
    if (key !== this.gridKey) {
      this.gridKey = key;
      this.grid.geometry.dispose();
      this.grid.geometry = makeGridGeometry(sp, half);
      this.gridBuilds++;
    }
    this.gridExtent.value = half * sp;
    this.gridPitch.value = sp;
    // Snapped to the pitch, so the lines stay on world multiples of it. A small
    // lift keeps the lines off the ground plane without z-fighting.
    this.grid.position.set(Math.round(cx / sp) * sp, this.groundY + Math.max(0.01, 1e-4 * size), Math.round(cz / sp) * sp);
    this.grid.visible = gridOn;

    this.gridColor.value.copy(theme.grid);
    this.gridOpacity.value = theme.gridOpacity;
    this.axisTone.value = theme.lightBackground ? 0.35 : 1;

    // A wide floor: the fade (in unit plane coordinates) happens far out.
    const extent = 12 * Math.max(half * sp, 2 * size);
    this.ground.scale.set(extent, extent, 1);
    this.ground.position.set(this.grid.position.x, this.groundY, this.grid.position.z);
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
    this.grid.geometry.dispose();
    this.gridMaterial.dispose();
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
  // The colour node (set by Scaffold) reads the vertex colours itself.
  const m = new THREE.LineBasicNodeMaterial({ depthTest: false, depthWrite: false, transparent: true });
  const axes = new THREE.LineSegments(g, m);
  axes.name = "axes";
  axes.renderOrder = 20;
  axes.frustumCulled = false;
  return axes;
}

/** Grid lines in the patch's local frame, centred on its origin. */
function makeGridGeometry(sp: number, half: number): THREE.BufferGeometry {
  const n = 2 * half + 1;
  const pos = new Float32Array(n * 2 * 6);
  const alongX = new Float32Array(n * 2 * 2);
  let k = 0, e = 0;
  const L = half * sp;
  for (let i = -half; i <= half; i++) {
    pos.set([i * sp, 0, -L, i * sp, 0, L], k); k += 6; // along z
    alongX.set([0, 0], e); e += 2;
    pos.set([-L, 0, i * sp, L, 0, i * sp], k); k += 6; // along x
    alongX.set([1, 1], e); e += 2;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("alongX", new THREE.BufferAttribute(alongX, 1));
  return g;
}
