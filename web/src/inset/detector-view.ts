// The detector's editor in the component view: its face (shape, physical
// size, pixels), what it measures, and the picture that measure makes, with
// a panel beside it for cross-sections through the picture or for how the
// light on it differs by wavelength.
//
// The face is drawn to scale and the pixels are drawn as the grid they are,
// so a coarse grid looks coarse. Power and irradiance are the same face
// histogram with different divisors, so both draw the face. Intensity is a
// different histogram, binned by the direction light arrived from: on a grid
// of two angles by default, the far field with straight axes, or as the polar
// map a goniometer draws. The numbers and the pictures come from one
// `detectorResultJSON` reply, the same one the dock reads, and the dock
// re-reads this view whenever it re-reads itself, so the flux printed here is
// the flux printed there.
//
// Cross-sections are lines on the picture, dragged to where they should cut;
// the plot beside it follows as they move, from numbers already in hand, so a
// drag traces nothing. The colour panel plots, line by line, where each
// wavelength lands, which way it travels, how large its spot is and where it
// focuses: what chromatic aberration is left, measured on this detector.

import {
  formatFlux,
  groupLines,
  raysPerLitPixel,
  type Box,
  type DetectorFace,
  type DetectorReadout,
  type FaceBox,
  type SpectralLine,
} from "../detector-face";
import {
  CUT_COLOURS,
  cellGrid,
  cutCentre,
  drawAxes,
  drawCellGrid,
  fwhm,
  profile,
  type CellGrid,
  type Cut,
} from "../detector-plots";
import { drawPolarMap } from "../polarmap";
import { button, fmt, label, numberField, segmented, type Editor, type EditorContext, type Handle, type NumberField } from "./ui";

type Side = "none" | "cuts" | "colour";
type Metric = "angle" | "centroid" | "spot" | "focus";

/** What a detector's view shows beside its picture, kept for the session so
 *  selecting the detector again finds its cuts where they were left. */
type Memory = { side: Side; cuts: Cut[]; cutsUnit: "mm" | "°" | null; metric: Metric | null; exposure: number };
const memory = new Map<number, Memory>();

const MAX_CUTS = CUT_COLOURS.length;
/** Quick picks for the analysis rays; the field beside them takes any count. */
const RAY_PRESETS = [
  { value: "0", text: "scene", title: "the rays the ray-density control sets" },
  { value: "1e5", text: "100k", title: "a hundred thousand launch rays" },
  { value: "1e6", text: "1M", title: "a million launch rays" },
  { value: "1e7", text: "10M", title: "ten million launch rays" },
] as const;
/** The most the tracer takes (`MAX_ANALYSIS_RAYS` in the wasm crate). */
const MAX_RAYS = 1e9;

/** A ray count as typed: `250000`, `250 000`, `250,000`, `2.5e5`, `250k`,
 *  `2.5M`, `1G`; empty or `scene` for the ray density's own (0). Null for
 *  anything else. */
export function parseRayCount(text: string): number | null {
  const t = text.trim().toLowerCase().replace(/[\s,_']/g, "");
  if (t === "" || t === "scene" || t === "0") return 0;
  const m = /^(\d+(?:\.\d*)?(?:e\d+)?)([kmg]?)$/.exec(t);
  if (!m) return null;
  const v = Number(m[1]) * { "": 1, k: 1e3, m: 1e6, g: 1e9 }[m[2] as "" | "k" | "m" | "g"];
  return Number.isFinite(v) ? Math.round(v) : null;
}

/** A ray count for the field: `250k`, `2.5M`, or the digits below 10 000. */
const shortCount = (n: number) =>
  n >= 1e9 ? `${+(n / 1e9).toPrecision(4)}G`
  : n >= 1e6 ? `${+(n / 1e6).toPrecision(4)}M`
  : n >= 1e4 ? `${+(n / 1e3).toPrecision(4)}k`
  : String(Math.round(n));

/** Whether a line arrives as a cone converging to (or diverging from) a
 *  focus, rather than as a near-parallel beam. Every line has a plane of
 *  least spot size, but in light a tube's walls have mixed, or in a
 *  collimator's output, that plane is arithmetic and not a focus. The cone
 *  tells them apart: 20 mrad RMS is about f/25, slower than any lens anyone
 *  focuses with and several times the spread of sunlight a collimator
 *  squeezes. */
const converging = (l: SpectralLine) => l.focus != null && (l.spread ?? 0) > 0.02;

/** One standard error of a line's mean direction in each plane, radians, and
 *  of its centroid in each direction, mm: the spread over the square root of
 *  the rays behind it, per axis (the RMS figures are over both axes). Two
 *  lines closer than a few of these differ by sampling, not by colour. */
const noise = (l: SpectralLine) => {
  const k = 1 / Math.sqrt(2 * Math.max(1, l.hits));
  return { angle: (l.spread ?? 0) * k, centroid: (l.rms ?? 0) * k };
};

/** A count of rays for display: 4.6 k, 1.2 M. */
const count = (n: number) =>
  n >= 1e6 ? `${fmt(n / 1e6, 2)} M` : n >= 1e4 ? `${fmt(n / 1e3, 1)} k` : String(Math.round(n));

export function createDetectorView(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  const mem = memory.get(id) ?? { side: "none", cuts: [], cutsUnit: null, metric: null, exposure: 0 };
  memory.set(id, mem);

  type Measure = DetectorReadout["measure"];
  type Shape = DetectorFace["shape"];
  let info: DetectorReadout | null = null;
  let face: DetectorFace | null = null;
  // The grid the picture is made of, and where it was last drawn. `mapIsFace`
  // says whether that picture is the face (the face's handles belong on it)
  // or the far field (they do not).
  let grid: CellGrid | null = null;
  let box: FaceBox | null = null;
  let mapIsFace = false;
  // The scale held while a face handle is dragged, so the face does not
  // rescale under the pointer.
  let held: number | null = null;

  // --- the bar -------------------------------------------------------------
  const pick = segmented<Measure>(
    ctx.bar,
    [
      { value: "power", text: "power", title: "flux landing in each pixel" },
      { value: "irradiance", text: "irradiance", title: "each pixel's flux over its area on the face, per mm²" },
      {
        value: "intensity",
        text: "intensity",
        title: "flux per steradian, binned by the direction light arrived from",
      },
      {
        value: "illuminance",
        text: "illuminance",
        title: "lux, as a light meter reads them, and the face drawn in the colours a person would see (CIE 1931 observer)",
      },
    ],
    (m) => {
      if (!scene.setDetectorMeasure(id, m, info?.coneDeg ?? 30)) {
        ctx.note("this detector cannot measure that", true);
        return;
      }
      ctx.note("");
      ctx.edited(true);
    },
  );
  label(ctx.bar, "face");
  const shapePick = segmented<Shape>(
    ctx.bar,
    [
      { value: "rectangle", text: "rect", title: "a rectangular face, as a camera sensor has" },
      { value: "disc", text: "disc", title: "a round face, as a power meter head or a photodiode has" },
      { value: "annulus", text: "ring", title: "a round face with a round hole in it" },
    ],
    (s) => reshape(s),
  );
  const layoutRow = document.createElement("span");
  layoutRow.className = "cv-group";
  ctx.bar.appendChild(layoutRow);
  label(layoutRow, "far field");
  const layoutPick = segmented<"grid" | "polar">(
    layoutRow,
    [
      { value: "grid", text: "grid", title: "two angles on straight axes: the far field as a camera at a lens's focus sees it, and cross-sections cut through it" },
      { value: "polar", text: "polar", title: "polar angle outward and azimuth around, as a goniometer plots it; one sector is the I(θ) curve" },
    ],
    (l) => {
      if (!scene.setDetectorDirectionLayout(id, l)) return;
      ctx.edited(true);
    },
  );

  const sideRow = document.createElement("span");
  sideRow.className = "cv-group";
  ctx.bar.appendChild(sideRow);
  label(sideRow, "beside");
  const sidePick = segmented<Side>(
    sideRow,
    [
      { value: "none", text: "nothing", title: "the picture alone" },
      { value: "cuts", text: "cross-sections", title: "profiles along lines you drag across the picture" },
      { value: "colour", text: "colour", title: "each wavelength's position, direction, spot and focus: the chromatic aberration left at this detector" },
    ],
    (s) => setSide(s),
  );
  const cutButtons = document.createElement("span");
  cutButtons.className = "cv-group";
  ctx.bar.appendChild(cutButtons);
  button(cutButtons, "+ row", "a cross-section along a row: across the picture at one height", () => addCut("row"));
  button(cutButtons, "+ col", "a cross-section down a column: up the picture at one place across", () => addCut("col"));
  button(cutButtons, "− cut", "remove the last cross-section", () => {
    mem.cuts.pop();
    afterCuts();
  });
  const metricRow = document.createElement("span");
  metricRow.className = "cv-group";
  ctx.bar.appendChild(metricRow);
  const metricPick = segmented<Metric>(
    metricRow,
    [
      { value: "angle", text: "direction", title: "each line's mean arrival angle, from the mean of all: the colour a collimator leaves in its beam" },
      { value: "centroid", text: "position", title: "each line's centroid, from the mean of all: lateral colour" },
      { value: "spot", text: "spot", title: "each line's RMS spot radius on the face" },
      { value: "focus", text: "focus", title: "where along the normal each line's spot is smallest: axial colour" },
    ],
    (m) => {
      mem.metric = m;
      metricPick.set(m);
      ctx.redraw();
    },
  );

  const raysRow = document.createElement("span");
  raysRow.className = "cv-group";
  ctx.bar.appendChild(raysRow);
  label(raysRow, "analysis rays");
  const raysField = document.createElement("input");
  raysField.type = "text";
  raysField.className = "cv-rays";
  raysField.dataset.name = "analysisRays";
  raysField.spellcheck = false;
  raysField.placeholder = "scene";
  raysField.title =
    "how many launch rays to trace the numbers with, in the background: 250000, 2.5e5, 250k, 2.5M. " +
    "The picture keeps the ray density's own rays. Applies to every detector; empty for the ray density's.";
  raysRow.appendChild(raysField);
  const raysPick = segmented<(typeof RAY_PRESETS)[number]["value"]>(
    raysRow,
    RAY_PRESETS.map((p) => ({ ...p })),
    (v) => setRays(Number(v)),
  );
  /** Trace the numbers with `n` launch rays (0: the ray density's). */
  function setRays(n: number) {
    const kept = scene.setAnalysisRays(n);
    showRays(kept);
    ctx.retrace();
    writeReadout();
  }
  function showRays(n: number) {
    if (document.activeElement !== raysField) raysField.value = n > 0 ? shortCount(n) : "";
    raysField.classList.remove("bad");
    raysPick.set(RAY_PRESETS.find((p) => Number(p.value) === n)?.value ?? null);
  }
  const commitRays = () => {
    const n = parseRayCount(raysField.value);
    if (n == null) {
      raysField.classList.add("bad");
      return;
    }
    if (n === scene.analysisRays()) {
      showRays(n);
      return;
    }
    setRays(Math.min(n, MAX_RAYS));
  };
  raysField.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      commitRays();
      raysField.blur();
    } else if (e.key === "Escape") {
      raysField.blur();
    }
    e.stopPropagation();
  });
  raysField.addEventListener("blur", () => {
    commitRays();
    showRays(scene.analysisRays());
  });

  // --- the fields ----------------------------------------------------------
  const fWidth = numberField(ctx.fields, "width", "mm", {
    name: "detWidth", step: 0.5, digits: 3,
    title: "the face's full width, along its local u axis",
    commit: (v) => face && writeFace({ ...face, width: v }, true),
  });
  const fHeight = numberField(ctx.fields, "height", "mm", {
    name: "detHeight", step: 0.5, digits: 3,
    title: "the face's full height, along its local v axis",
    commit: (v) => face && writeFace({ ...face, height: v }, true),
  });
  const fDiameter = numberField(ctx.fields, "⌀", "mm", {
    name: "detDiameter", step: 0.5, digits: 3,
    title: "the face's diameter",
    commit: (v) => face && writeFace({ ...face, width: v, height: v }, true),
  });
  const fHole = numberField(ctx.fields, "hole ⌀", "mm", {
    name: "detHole", step: 0.5, digits: 3,
    title: "the diameter of the hole in the ring",
    commit: (v) => face && writeFace({ ...face, inner: v }, true),
  });
  const fAcross = numberField(ctx.fields, "px across", "", {
    name: "detPixelsU", step: 1, digits: 0,
    title: "pixels across the face (u); the pitch is the width over this",
    commit: (v) => face && writePixels(v, face.pixels[1]),
  });
  const fDown = numberField(ctx.fields, "px down", "", {
    name: "detPixelsV", step: 1, digits: 0,
    title: "pixels down the face (v); the pitch is the height over this",
    commit: (v) => face && writePixels(face.pixels[0], v),
  });
  const cone = numberField(ctx.fields, "cone", "°", {
    name: "cone",
    step: 5,
    digits: 1,
    title: "the half angle the histogram spans, from the face normal (on the grid, in each of the two angles)",
    commit: (v) => {
      if (!scene.setDetectorMeasure(id, "intensity", v)) {
        ctx.note("a cone has to be inside (0, 90] degrees", true);
        return;
      }
      ctx.note("");
      ctx.edited(true);
    },
  });
  const exposure = numberField(ctx.fields, "exposure", "EV", {
    name: "detExposure", step: 0.5, digits: 1,
    title: "brightness of the true-colour picture in stops: 0 draws the brightest part of the face (its 99th percentile) at full white, as an eye adapted to it sees it",
    commit: (v) => {
      mem.exposure = Math.max(-10, Math.min(10, v));
      if (info) grid = cellGrid(info, face, mem.exposure);
      ctx.redraw();
    },
  });
  const fRings = numberField(ctx.fields, "rings", "", {
    name: "detRings", step: 1, digits: 0,
    title: "polar rings out to the cone: the resolution of I(θ)",
    commit: (v) => writeDirections(v, face?.directionBins?.[1] ?? 1),
  });
  const fSectors = numberField(ctx.fields, "sectors", "", {
    name: "detSectors", step: 1, digits: 0,
    title: "azimuth sectors around the normal; one sector is the I(θ) curve",
    commit: (v) => writeDirections(face?.directionBins?.[0] ?? 24, v),
  });

  const show = (f: { input: HTMLInputElement }, on: boolean) =>
    (f.input.closest(".cv-field") as HTMLElement | null)?.toggleAttribute("hidden", !on);
  const relabel = (f: NumberField, text: string, title: string) => {
    const row = f.input.closest(".cv-field") as HTMLElement | null;
    const span = row?.querySelector("span");
    if (span) span.textContent = text;
    if (row) row.title = title;
  };

  // --- edits ---------------------------------------------------------------

  /** Send a face to the scene. `settled` false while a handle is held: the
   *  fast picture follows, and the numbers wait for the release. A settled
   *  edit re-reads this view on its way through the dock (main.ts,
   *  `refresh3dDetector`), so nothing here traces a second time. */
  function writeFace(next: DetectorFace, settled: boolean): boolean {
    const ok =
      next.shape === "rectangle" ? scene.setDetectorRectangle(id, next.width, next.height)
      : next.shape === "disc" ? scene.setDetectorDisc(id, next.width)
      : scene.setDetectorAnnulus(id, next.inner, next.width);
    if (!ok) {
      ctx.note(
        next.shape === "annulus"
          ? "a ring needs a positive size and a hole smaller than itself"
          : "a face needs a positive size",
        true,
      );
      return false;
    }
    ctx.note("");
    ctx.edited(settled);
    if (!settled) readFace();
    ctx.redraw();
    return true;
  }

  /** Switch the face's shape, keeping it the size it was: the largest disc
   *  inside a rectangle, the square around a disc, and a ring whose hole is a
   *  third of it. The pixel counts stay. */
  function reshape(to: Shape) {
    if (!face || face.shape === to) return;
    const d = Math.min(face.width, face.height);
    const next: DetectorFace =
      to === "rectangle" ? { ...face, shape: to, width: face.width, height: face.height }
      : to === "disc" ? { ...face, shape: to, width: d, height: d }
      : { ...face, shape: to, width: d, height: d, inner: face.inner > 0 && face.inner < d ? face.inner : d / 3 };
    writeFace(next, true);
  }

  function writePixels(across: number, down: number) {
    if (!Number.isInteger(across) || !Number.isInteger(down)) {
      ctx.note("a pixel count is a whole number", true);
      return;
    }
    if (!scene.setDetectorPixels(id, across, down)) {
      const most = face?.maxCells ?? 0;
      ctx.note(`pixel counts start at 1, and a face holds at most ${most.toLocaleString("en")} pixels`, true);
      return;
    }
    ctx.note("");
    ctx.edited(true);
  }

  function writeDirections(a: number, b: number) {
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
      ctx.note("cell counts are whole numbers", true);
      return;
    }
    if (!scene.setDetectorDirectionBins(id, a, b)) {
      ctx.note("cell counts start at 1", true);
      return;
    }
    ctx.note("");
    ctx.edited(true);
  }

  // --- the panel beside the picture ----------------------------------------

  function setSide(s: Side) {
    mem.side = s;
    ctx.widen(s !== "none");
    if (s === "cuts") ensureCuts();
    showSideControls();
    writeReadout();
    ctx.redraw();
  }

  function showSideControls() {
    sidePick.set(mem.side);
    cutButtons.hidden = mem.side !== "cuts" || !grid;
    metricRow.hidden = mem.side !== "colour";
    metricPick.set(metric());
  }

  /** The metric the colour panel plots: the one picked, else focus for a
   *  beam still converging on the detector and direction for one that is
   *  collimated, which is what a focusing lens and a collimator each leave
   *  as colour. */
  function metric(): Metric {
    if (mem.metric) return mem.metric;
    const lines = groupLines(info?.spectral ?? []);
    return lines.length > 0 && lines.every(converging) ? "focus" : "angle";
  }

  /** Cuts through the brightest cell, when there are none yet, or when the
   *  picture changed between the face and the far field and the old cuts'
   *  units mean nothing on the new one. */
  function ensureCuts() {
    if (!grid) return;
    if (mem.cutsUnit === grid.unit && mem.cuts.length) return;
    mem.cutsUnit = grid.unit;
    const [u, v] = brightest(grid);
    mem.cuts = [{ axis: "row", at: v }, { axis: "col", at: u }];
  }

  /** The centre of the brightest cell, in the grid's unit. */
  function brightest(g: CellGrid): [number, number] {
    let best = -1, top = 0;
    for (let k = 0; k < g.values.length; k++) {
      if (g.values[k] > top) {
        top = g.values[k];
        best = k;
      }
    }
    if (best < 0) return [0, 0];
    const i = best % g.nu, j = Math.floor(best / g.nu);
    return [-g.width / 2 + ((i + 0.5) * g.width) / g.nu, -g.height / 2 + ((j + 0.5) * g.height) / g.nv];
  }

  function addCut(axis: Cut["axis"]) {
    if (!grid) return;
    if (mem.cuts.length >= MAX_CUTS) {
      ctx.note(`at most ${MAX_CUTS} cross-sections`, true);
      return;
    }
    ensureCuts();
    const [u, v] = brightest(grid);
    mem.cuts.push({ axis, at: axis === "row" ? v : u });
    afterCuts();
  }

  function afterCuts() {
    ctx.note("");
    writeReadout();
    ctx.redraw();
  }

  // --- reading -------------------------------------------------------------

  /** The face alone: cheap, so a drag can follow it. */
  function readFace() {
    const raw = JSON.parse(scene.detectorFaceJSON(id));
    face = raw && typeof raw === "object" ? (raw as DetectorFace) : null;
    if (!face) return;
    shapePick.set(face.shape);
    const round = face.shape !== "rectangle";
    show(fWidth, !round);
    show(fHeight, !round);
    show(fDiameter, round);
    show(fHole, face.shape === "annulus");
    fWidth.set(face.width);
    fHeight.set(face.height);
    fDiameter.set(face.width);
    fHole.set(face.inner);
    fAcross.set(face.pixels[0]);
    fDown.set(face.pixels[1]);
    const angular = face.directionBins != null;
    show(fRings, angular);
    show(fSectors, angular);
    if (face.directionBins) {
      fRings.set(face.directionBins[0]);
      fSectors.set(face.directionBins[1]);
    }
  }

  /** The face and a fresh f64 readout of it. */
  function reread() {
    readFace();
    showRays(scene.analysisRays());
    const text = ctx.readout(id);
    // No reference traced yet: the last readout stays until one is.
    if (text !== null) {
      const raw = JSON.parse(text);
      info = raw && typeof raw === "object" ? (raw as DetectorReadout) : null;
    }
    pick.set(info?.measure ?? null);
    const intensity = info?.measure === "intensity";
    show(cone, intensity);
    show(exposure, info?.measure === "illuminance");
    exposure.set(mem.exposure);
    layoutRow.hidden = !intensity;
    layoutPick.set(info?.directionLayout ?? null);
    if (info?.directionLayout === "grid") {
      relabel(fRings, "across", "cells across the far field, over ±cone in the u angle");
      relabel(fSectors, "down", "cells down the far field, over ±cone in the v angle");
    } else {
      relabel(fRings, "rings", "polar rings out to the cone: the resolution of I(θ)");
      relabel(fSectors, "sectors", "azimuth sectors around the normal; one sector is the I(θ) curve");
    }
    if (info?.coneDeg != null) cone.set(info.coneDeg);
    grid = info ? cellGrid(info, face, mem.exposure) : null;
    if (mem.side === "cuts") ensureCuts();
    showSideControls();
    writeReadout();
  }

  const length = (mm: number) => (mm < 1 ? `${fmt(mm * 1000, 1)} µm` : `${fmt(mm, 3)} mm`);

  function writeReadout() {
    ctx.read.replaceChildren();
    ctx.read.classList.toggle("tall", mem.side !== "none");
    const line = (html: string, warn = false) => {
      const div = document.createElement("div");
      div.innerHTML = html;
      if (warn) div.className = "warn";
      ctx.read.appendChild(div);
    };
    if (!info || !face) return;

    // The flux on the whole face first: the number the dock quotes, in the
    // same digits, whatever the measure divides it by.
    const rays = `${info.hits} ray${info.hits === 1 ? "" : "s"}`;
    if (info.hits === 0) {
      line("flux <b>0</b> · no rays on this face yet");
    } else if (info.measure === "power") {
      line(`flux <b>${formatFlux(info.totalIncident)}</b> on the face from ${rays} · brightest pixel <b>${formatFlux(info.peak)}</b>`);
    } else if (info.measure === "illuminance") {
      const mean = info.cells.length ? info.cells.reduce((s, v) => s + v, 0) / info.cells.length : null;
      line(
        `flux <b>${formatFlux(info.totalIncident)}</b> from ${rays} · peak <b>${formatFlux(info.peak)}</b> ${info.unit}` +
          (mean != null ? ` · pixel mean <b>${formatFlux(mean)}</b> ${info.unit}` : ""),
      );
    } else if (info.measure === "irradiance") {
      line(
        `flux <b>${formatFlux(info.totalIncident)}</b> from ${rays} · peak <b>${formatFlux(info.peak)}</b> ${info.unit}` +
          ` · face mean <b>${formatFlux(info.meanIrradiance)}</b>`,
      );
    } else {
      const [a, b] = info.coneBins;
      const share = info.totalIncident > 0 ? info.outsideCone / info.totalIncident : 0;
      const past = share > 1e-9 ? `, ${(share * 100).toFixed(1)}% arrived past it` : "";
      const cells = info.directionLayout === "grid"
        ? `${a} × ${b} cells over ±${fmt(info.coneDeg ?? 0, 1)}°`
        : `${a} ring${a === 1 ? "" : "s"} × ${b} sector${b === 1 ? "" : "s"} to ${fmt(info.coneDeg ?? 0, 0)}°`;
      line(`flux <b>${formatFlux(info.totalIncident)}</b> from ${rays} · peak <b>${formatFlux(info.peak)}</b> ${info.unit} over ${cells}${past}`);
    }

    if (mem.side === "cuts") writeCuts(line);
    if (mem.side === "colour") writeColour(line);

    // The instrument: its size, its pixels, and what a pixel is.
    const [nu, nv] = face.pixels;
    const [pu, pv] = face.pitch;
    const size =
      face.shape === "rectangle" ? `${fmt(face.width, 3)} × ${fmt(face.height, 3)} mm`
      : face.shape === "disc" ? `⌀ ${fmt(face.width, 3)} mm`
      : `⌀ ${fmt(face.width, 3)} mm, hole ⌀ ${fmt(face.inner, 3)} mm`;
    const pitch = Math.abs(pu - pv) <= 1e-9 * Math.max(pu, pv) ? length(pu) : `${length(pu)} × ${length(pv)}`;
    const rim = face.shape === "rectangle" ? "" : " · rim pixels read over their part on the face";
    line(`${size} · ${nu} × ${nv} px of ${pitch}${rim}`);

    // Whether a pixel's reading means much: the rays behind it. With k rays a
    // random source puts one standard deviation at 1/√k, and a grid source
    // can gain or lose a row and a column of rays, which is 2/√k; the larger
    // is the one quoted, since the app's sources sample on a grid.
    const k = raysPerLitPixel(info);
    const [own, traced, passes] = Array.from(scene.launchRayCounts());
    const more = traced < MAX_RAYS / 2 ? "; raise the analysis rays" : "";
    if (k != null && info.measure !== "intensity") {
      const per = k < 10 ? fmt(k, 1) : String(Math.round(k));
      if (k < 10) {
        line(`${per} rays per lit pixel: single pixels are sampling noise${more} or use fewer pixels`, true);
      } else {
        line(`${per} rays per lit pixel: sampling alone can move one pixel by about ±${+(200 / Math.sqrt(k)).toPrecision(2)}%`);
      }
    }

    // How many rays the numbers rest on, against what the picture draws.
    const inPasses = passes > 1 ? ` in ${passes} passes` : "";
    line(
      traced > own
        ? `numbers from <b>${count(traced)}</b> launch rays${inPasses}; the picture keeps the ray density's ${count(own)}`
        : `numbers from <b>${count(own)}</b> launch rays, the ray density's; set the analysis rays to trace more in the background`,
    );
  }

  /** Each cut's line, in its colour: where it is and what it shows. */
  function writeCuts(line: (html: string, warn?: boolean) => void) {
    if (!grid) {
      line("cross-sections cut a grid: set the far field to <b>grid</b>");
      return;
    }
    const g = grid;
    mem.cuts.forEach((cut, i) => {
      const p = profile(g, cut);
      const peak = p.reduce((m, q) => Math.max(m, q.y), 0);
      const w = fwhm(p);
      const where = cut.axis === "row" ? `v = ${fmt(cutCentre(g, cut), 3)}` : `u = ${fmt(cutCentre(g, cut), 3)}`;
      const colour = CUT_COLOURS[i % CUT_COLOURS.length];
      line(
        `<span style="color:${colour}">■</span> ${cut.axis} at ${where} ${g.unit} · peak <b>${formatFlux(peak)}</b> ${g.reads}` +
          (w != null ? ` · FWHM <b>${fmt(w, 4)}</b> ${g.unit}` : ""),
      );
    });
  }

  /** The chromatic numbers: how far apart the lines are against how spread
   *  out each one is. */
  function writeColour(line: (html: string, warn?: boolean) => void) {
    const bins = info?.spectral ?? [];
    const lines = groupLines(bins);
    if (lines.length < 2) {
      line(lines.length ? "one wavelength on this detector: no colour to compare" : "no light on this detector");
      return;
    }
    const nm =
      `${fmt(lines[0].nm, 0)}–${fmt(lines[lines.length - 1].nm, 0)} nm, ` +
      (lines.length < bins.length ? `${bins.length} bins in ${lines.length} groups of equal flux` : `${lines.length} lines`);
    const span = (f: (l: SpectralLine) => number | null) => {
      const v = lines.map(f).filter((x): x is number => x != null && Number.isFinite(x));
      return v.length > 1 ? Math.max(...v) - Math.min(...v) : null;
    };
    const dir = Math.hypot(span((l) => l.angle[0]) ?? 0, span((l) => l.angle[1]) ?? 0);
    const spread = Math.max(0, ...lines.map((l) => l.spread ?? 0));
    // The typical sampling error of one line's mean: the numbers above are
    // only colour where they stand clear of it.
    const typical = (f: (l: SpectralLine) => number) => {
      const v = lines.map(f).sort((a, b) => a - b);
      return v[Math.floor(v.length / 2)];
    };
    const dirNoise = typical((l) => noise(l).angle);
    line(
      `${nm} · direction differs by <b>${fmt(dir * 1e3, 3)}</b> mrad between lines` +
        ` (sampling ±${fmt(dirNoise * 1e3, 2)} mrad a line);` +
        ` each line spreads up to <b>${fmt(spread * 1e3, 3)}</b> mrad RMS`,
      dir < 4 * dirNoise,
    );
    if (dir < 4 * dirNoise) {
      line("the difference is within a few sampling errors: raise the analysis rays before reading it as colour", true);
    }
    const pos = Math.hypot(span((l) => l.centroid[0]) ?? 0, span((l) => l.centroid[1]) ?? 0);
    const posNoise = typical((l) => noise(l).centroid);
    const focus = lines.every(converging) ? span((l) => l.focus) : null;
    const spot = Math.max(0, ...lines.map((l) => l.rms ?? 0));
    line(
      `lateral colour <b>${length(pos)}</b> between centroids (sampling ±${length(posNoise)}; spots up to ${length(spot)} RMS)` +
        (focus != null
          ? ` · axial colour: best focus spans <b>${fmt(focus, 4)}</b> mm`
          : " · rays near parallel here (under 20 mrad RMS): read the direction, not a focus"),
    );
    if (info && info.spectralUntracked > 0) {
      line(`${formatFlux(info.spectralUntracked)} of the flux is on lines past the first 64 and is left out here`, true);
    }
  }

  // --- drawing -------------------------------------------------------------

  /** The picture and the panel beside it: side by side in a wide window,
   *  one above the other in a tall one. */
  function layout(w: number, h: number): { map: Box; side: Box | null } {
    if (mem.side === "none") return { map: { x: 0, y: 0, w, h }, side: null };
    if (w >= h * 1.25) {
      const mw = Math.floor(w * 0.48);
      return { map: { x: 0, y: 0, w: mw, h }, side: { x: mw + 6, y: 0, w: w - mw - 6, h } };
    }
    const mh = Math.floor(h * 0.55);
    return { map: { x: 0, y: 0, w, h: mh }, side: { x: 0, y: mh + 4, w, h: h - mh - 4 } };
  }

  /** The face with nothing on it, for a handle held or no readout yet. */
  const emptyFace = (f: DetectorFace): CellGrid => ({
    values: [], nu: f.pixels[0], nv: f.pixels[1], width: f.width, height: f.height,
    unit: "mm", face: f, reads: "", top: null,
  });

  function drawMap(c: CanvasRenderingContext2D, area: Box) {
    if (!face) return;
    const polar = info?.measure === "intensity" && info.directionLayout !== "grid";
    if (held == null && polar && info) {
      drawPolarMap(
        c,
        { cells: info.cells, rings: info.coneBins[0], sectors: info.coneBins[1], coneDeg: info.coneDeg ?? 0 },
        area,
      );
      box = null;
      mapIsFace = false;
      return;
    }
    // While a handle is held the face is new and its numbers are not in yet,
    // so it is drawn empty rather than with the old face's pixels.
    const g = held != null || !grid ? emptyFace(face) : grid;
    box = drawCellGrid(c, g, area, held ?? undefined);
    mapIsFace = g.unit === "mm";
    if (mem.side === "cuts" && grid && held == null) drawCutLines(c);
  }

  /** Where a cut's line is on screen. */
  function cutLine(cut: Cut): { x0: number; y0: number; x1: number; y1: number } | null {
    if (!box || !grid) return null;
    const at = cutCentre(grid, cut);
    if (cut.axis === "row") {
      const y = box.y + box.h - (at + grid.height / 2) * box.scale;
      return { x0: box.x, y0: y, x1: box.x + box.w, y1: y };
    }
    const x = box.x + (at + grid.width / 2) * box.scale;
    return { x0: x, y0: box.y, x1: x, y1: box.y + box.h };
  }

  function drawCutLines(c: CanvasRenderingContext2D) {
    c.save();
    c.lineWidth = 1.5;
    c.setLineDash([5, 3]);
    mem.cuts.forEach((cut, i) => {
      const l = cutLine(cut);
      if (!l) return;
      c.strokeStyle = CUT_COLOURS[i % CUT_COLOURS.length];
      c.beginPath();
      c.moveTo(l.x0, l.y0);
      c.lineTo(l.x1, l.y1);
      c.stroke();
    });
    c.restore();
  }

  function drawCuts(c: CanvasRenderingContext2D, area: Box) {
    if (!grid) {
      note(c, area, "a polar histogram has no rows to cut: set the far field to grid");
      return;
    }
    const g = grid;
    const profiles = mem.cuts.map((cut) => profile(g, cut));
    if (!profiles.length) {
      note(c, area, "add a row or a column to cut the picture");
      return;
    }
    const xs = mem.cuts.map((cut) => (cut.axis === "row" ? g.width : g.height) / 2);
    const half = Math.max(...xs);
    const top = Math.max(1e-300, ...profiles.map((p) => p.reduce((m, q) => Math.max(m, q.y), 0)));
    const axes = drawAxes(
      c, area, [-half, half], [0, top * 1.05],
      g.unit === "°" ? "angle from the normal (°)" : "position on the face (mm)",
      g.reads,
    );
    c.save();
    c.beginPath();
    c.rect(axes.inner.x, axes.inner.y, axes.inner.w, axes.inner.h);
    c.clip();
    profiles.forEach((p, i) => {
      c.strokeStyle = CUT_COLOURS[i % CUT_COLOURS.length];
      c.lineWidth = 1.5;
      c.beginPath();
      p.forEach((q, k) => (k ? c.lineTo(axes.X(q.x), axes.Y(q.y)) : c.moveTo(axes.X(q.x), axes.Y(q.y))));
      c.stroke();
    });
    c.restore();
  }

  function drawColour(c: CanvasRenderingContext2D, area: Box) {
    const lines = groupLines(info?.spectral ?? []);
    if (lines.length === 0) {
      note(c, area, "no light on this detector yet");
      return;
    }
    const m = metric();
    const w = lines.reduce((s, l) => s + l.flux, 0) || 1;
    const mean = (f: (l: SpectralLine) => number) => lines.reduce((s, l) => s + l.flux * f(l), 0) / w;
    // Each metric as one or two series of (nm, value).
    type Series = { name: string; mark: "dot" | "diamond"; points: { nm: number; v: number; e?: number }[] };
    let unit = "";
    let series: Series[] = [];
    if (m === "angle") {
      const [mu, mv] = [mean((l) => l.angle[0]), mean((l) => l.angle[1])];
      unit = "mrad from the mean direction";
      series = [
        { name: "u", mark: "dot", points: lines.map((l) => ({ nm: l.nm, v: (l.angle[0] - mu) * 1e3, e: noise(l).angle * 1e3 })) },
        { name: "v", mark: "diamond", points: lines.map((l) => ({ nm: l.nm, v: (l.angle[1] - mv) * 1e3, e: noise(l).angle * 1e3 })) },
      ];
    } else if (m === "centroid") {
      const [mu, mv] = [mean((l) => l.centroid[0]), mean((l) => l.centroid[1])];
      const big = lines.some((l) => Math.abs(l.centroid[0] - mu) > 1 || Math.abs(l.centroid[1] - mv) > 1);
      const k = big ? 1 : 1e3;
      unit = `${big ? "mm" : "µm"} from the mean centroid`;
      series = [
        { name: "u", mark: "dot", points: lines.map((l) => ({ nm: l.nm, v: (l.centroid[0] - mu) * k, e: noise(l).centroid * k })) },
        { name: "v", mark: "diamond", points: lines.map((l) => ({ nm: l.nm, v: (l.centroid[1] - mv) * k, e: noise(l).centroid * k })) },
      ];
    } else if (m === "spot") {
      const big = lines.some((l) => (l.rms ?? 0) > 1);
      const k = big ? 1 : 1e3;
      unit = `RMS spot radius (${big ? "mm" : "µm"})`;
      series = [{ name: "rms", mark: "dot", points: lines.filter((l) => l.rms != null).map((l) => ({ nm: l.nm, v: l.rms! * k })) }];
    } else {
      unit = "best focus along the normal, + past the face (mm)";
      series = [{ name: "focus", mark: "dot", points: lines.filter(converging).map((l) => ({ nm: l.nm, v: l.focus! })) }];
    }
    const all = series.flatMap((s) => s.points);
    if (!all.length) {
      note(c, area, m === "focus" ? "no line converges to a focus near this detector" : "nothing to plot");
      return;
    }
    const nmLo = Math.min(...all.map((p) => p.nm)), nmHi = Math.max(...all.map((p) => p.nm));
    let lo = Math.min(...all.map((p) => p.v - (p.e ?? 0))), hi = Math.max(...all.map((p) => p.v + (p.e ?? 0)));
    if (m === "angle" || m === "centroid") {
      lo = Math.min(lo, 0);
      hi = Math.max(hi, 0);
    }
    const pad = (hi - lo) * 0.12 || Math.max(Math.abs(hi), 1e-6) * 0.5;
    const axes = drawAxes(c, area, [nmLo - 15, nmHi + 15], [lo - pad, hi + pad], "wavelength (nm)", unit);
    c.save();
    for (const s of series) {
      c.strokeStyle = "rgba(205,214,228,0.35)";
      c.lineWidth = 1;
      c.setLineDash(s.mark === "diamond" ? [3, 3] : []);
      c.beginPath();
      s.points.forEach((p, k) => (k ? c.lineTo(axes.X(p.nm), axes.Y(p.v)) : c.moveTo(axes.X(p.nm), axes.Y(p.v))));
      c.stroke();
      c.setLineDash([]);
      for (const p of s.points) {
        const x = axes.X(p.nm), y = axes.Y(p.v);
        if (p.e) {
          // ±1 standard error of the mean: the sampling, not the optics.
          c.strokeStyle = "rgba(205,214,228,0.45)";
          c.beginPath();
          c.moveTo(x, axes.Y(p.v - p.e));
          c.lineTo(x, axes.Y(p.v + p.e));
          c.stroke();
        }
        c.fillStyle = scene.wavelengthCssColor(p.nm);
        c.beginPath();
        if (s.mark === "diamond") {
          c.moveTo(x, y - 4.5);
          c.lineTo(x + 4.5, y);
          c.lineTo(x, y + 4.5);
          c.lineTo(x - 4.5, y);
          c.closePath();
        } else {
          c.arc(x, y, 3.5, 0, Math.PI * 2);
        }
        c.fill();
      }
    }
    if (series.length > 1) {
      c.fillStyle = "rgba(205,214,228,0.6)";
      c.font = "10px ui-monospace, monospace";
      c.textAlign = "right";
      c.fillText("● u   ◆ v", axes.inner.x + axes.inner.w - 4, axes.inner.y + 11);
    }
    c.restore();
  }

  function note(c: CanvasRenderingContext2D, area: Box, text: string) {
    c.fillStyle = "rgba(205,214,228,0.5)";
    c.font = "11px ui-monospace, monospace";
    c.fillText(text, area.x + 8, area.y + 18, area.w - 16);
  }

  reread();
  if (mem.side !== "none") ctx.widen(true);

  return {
    hint: "face on, to scale · drag the corner or the rim to size the face · drag a cross-section's line to move it",
    refresh: reread,
    draw(c, w, h) {
      if (!face) return;
      const L = layout(w, h);
      drawMap(c, L.map);
      if (!L.side) return;
      if (mem.side === "cuts") drawCuts(c, L.side);
      else drawColour(c, L.side);
    },
    handles: () => {
      const out: Handle[] = [];
      if (face && box && mapIsFace) {
        const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
        if (face.shape === "rectangle") {
          out.push({ key: "corner", x: box.x + box.w, y: box.y, shape: "square", color: "#7fd6ff", cursor: "nesw-resize",
            title: `${fmt(face.width, 2)} × ${fmt(face.height, 2)} mm: drag the corner` });
        } else {
          out.push({ key: "rim", x: box.x + box.w, y: cy, shape: "ring", color: "#7fd6ff", cursor: "ew-resize",
            title: `⌀ ${fmt(face.width, 2)} mm: drag the rim` });
          if (face.shape === "annulus") {
            out.push({ key: "hole", x: cx + (face.inner / 2) * box.scale, y: cy, shape: "diamond", color: "#cdd6e4", cursor: "ew-resize",
              title: `hole ⌀ ${fmt(face.inner, 2)} mm: drag` });
          }
        }
      }
      // Each cut's grip, just outside the picture at the line's start.
      if (mem.side === "cuts" && grid && box && held == null) {
        mem.cuts.forEach((cut, i) => {
          const l = cutLine(cut);
          if (!l) return;
          out.push({
            key: `cut-${i}`,
            x: cut.axis === "row" ? l.x0 - 1 : l.x0,
            y: cut.axis === "row" ? l.y0 : l.y0 - 1,
            shape: "triangle",
            color: CUT_COLOURS[i % CUT_COLOURS.length],
            cursor: cut.axis === "row" ? "ns-resize" : "ew-resize",
            title: `${cut.axis} cross-section: drag the line anywhere along it`,
          });
        });
      }
      return out;
    },
    pick(x, y) {
      if (mem.side !== "cuts" || !grid || !box || held != null) return null;
      for (let i = mem.cuts.length - 1; i >= 0; i--) {
        const cut = mem.cuts[i];
        const l = cutLine(cut);
        if (!l) continue;
        const near = cut.axis === "row"
          ? Math.abs(y - l.y0) <= 4 && x >= l.x0 && x <= l.x1
          : Math.abs(x - l.x0) <= 4 && y >= l.y0 && y <= l.y1;
        if (near) {
          return {
            key: `cut-${i}`, x, y, shape: "triangle", color: CUT_COLOURS[i % CUT_COLOURS.length],
            cursor: cut.axis === "row" ? "ns-resize" : "ew-resize",
            title: `${cut.axis} cross-section: drag to move it`,
          };
        }
      }
      return null;
    },
    drag(hd, x, y, phase) {
      if (hd.key.startsWith("cut-")) {
        // A cut moves over numbers already in hand: nothing is traced.
        const cut = mem.cuts[Number(hd.key.slice(4))];
        if (!cut || !grid || !box) return;
        if (phase !== "end") {
          const g = grid;
          cut.at = cut.axis === "row"
            ? Math.min(g.height / 2, Math.max(-g.height / 2, (box.y + box.h - y) / box.scale - g.height / 2))
            : Math.min(g.width / 2, Math.max(-g.width / 2, (x - box.x) / box.scale - g.width / 2));
        }
        writeReadout();
        ctx.redraw();
        return;
      }
      if (!face || !box) return;
      if (phase === "start") {
        held = box.scale;
        ctx.redraw();
        return;
      }
      if (phase === "end") {
        held = null;
        ctx.edited(true);
        ctx.redraw();
        return;
      }
      const s = held ?? box.scale;
      const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
      const du = Math.max(0.05, (2 * Math.abs(x - cx)) / s);
      const dv = Math.max(0.05, (2 * Math.abs(y - cy)) / s);
      if (hd.key === "corner") writeFace({ ...face, width: du, height: dv }, false);
      else if (hd.key === "rim") writeFace({ ...face, width: Math.max(du, face.inner + 0.05), height: Math.max(du, face.inner + 0.05) }, false);
      else writeFace({ ...face, inner: Math.min(du, face.width - 0.05) }, false);
    },
  };
}
