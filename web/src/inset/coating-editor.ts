// The coating editor: a flat surface's reflectance and transmittance across the
// visible band, for a mirror, a beam splitter, a dichroic filter, a clear
// plate, or an absorber.
//
// The curves are sampled from Rust (`surfaceRtF64`), so they are the tables the
// tracer reads. The source's wavelengths are marked on them. Handles:
//   splitter: ● R and ● T levels, dragged up or down; the absorbed share
//     (1 - R - T) stays as it was;
//   dichroic: ◆ the edge, dragged sideways; ● its two ends set the transition
//     width; ● the reflect and pass plateaus set the peak R and T.
// A `film` surface is different in kind: a real stack of layers, evaluated by
// the transfer matrix at the angle the scene's light actually meets the surface
// at. Its curve is plotted at that angle, with the on-axis curve behind it in
// grey so the band shift is visible, and the stack itself is drawn under the
// plot with every layer boundary draggable.
//
// Kind buttons and the fields apply to both faces. Everything but `film` is
// scalar: no angle of incidence and no polarization anywhere.

import {
  fmt,
  label,
  numberField,
  segmented,
  type DragPhase,
  type Editor,
  type EditorContext,
  type Handle,
} from "./ui";

type Dichroic = { edgeNm: number; widthNm: number; longPass: boolean; rMax: number; tMax: number };
type FaceCoat = { kind: string; r: number | null; t: number | null; dichroic: Dichroic | null };
type Coat = { front: FaceCoat; back: FaceCoat };
type FilmLayer = { name: string; n: number; k: number; thicknessNm: number };
type FilmDesign = { kind: string; lambdaNm: number; aoiDeg: number; periods: number; nHigh: number; nLow: number };
type Film = {
  aoiDeg: number;
  design: FilmDesign | null;
  substrateN: number | null;
  layers: FilmLayer[];
  edgeNm: number | null;
  peakR: number;
};

const NM0 = 380;
const NM1 = 780;
const STEP = 2;
const R_COLOR = "#7fd6ff";
const T_COLOR = "#ffd678";

type Kind = "mirror" | "split" | "dichroic" | "film" | "clear" | "absorb";

export function createCoatingEditor(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  const grid = new Float64Array(Math.round((NM1 - NM0) / STEP) + 1).map((_, i) => NM0 + i * STEP);
  let coat: Coat = JSON.parse(scene.surfaceCoatingJSON(id));
  let curve: Float64Array<ArrayBufferLike> = new Float64Array(0);
  // The same stack on axis, drawn behind the working curve: the gap between the
  // two is the whole reason this editor exists.
  let curveOnAxis: Float64Array<ArrayBufferLike> | null = null;
  let film: Film | null = null;
  let lines: { nm: number; color: string }[] = [];
  let atLines: Float64Array<ArrayBufferLike> = new Float64Array(0);
  const bandColor = Array.from(grid, (nm) => scene.wavelengthCssColor(nm));

  // Plot frame (CSS px), set by `draw`.
  const box = { l: 34, r: 10, t: 14, b: 30, w: 0, h: 0 };
  const X = (nm: number) => box.l + ((nm - NM0) / (NM1 - NM0)) * (box.w - box.l - box.r);
  const Y = (v: number) => box.t + (1 - v) * (box.h - box.t - box.b);
  const nmAt = (x: number) => NM0 + ((x - box.l) / (box.w - box.l - box.r)) * (NM1 - NM0);
  const valueAt = (y: number) => 1 - (y - box.t) / (box.h - box.t - box.b);
  const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

  // --- controls -------------------------------------------------------------
  const kindSeg = segmented<Kind>(ctx.bar, [
    { value: "mirror", text: "mirror", title: "a 95% mirror on both faces" },
    { value: "split", text: "splitter", title: "a 50:50 beam splitter" },
    { value: "dichroic", text: "dichroic", title: "a long-pass dichroic at 550 nm, one table in wavelength, the same at every angle" },
    { value: "film", text: "film", title: "a real thin-film stack: angle-dependent, solved for the angle this surface is used at" },
    { value: "clear", text: "clear", title: "fully transmitting" },
    { value: "absorb", text: "absorb", title: "a beam block" },
  ], (k) => {
    if (k === "film") {
      // Start from a long-pass at 550 nm, solved for the angle this surface is
      // actually used at rather than for normal incidence.
      const aoi = scene.surfaceAoiDeg(id);
      const ok = scene.setSurfaceCoatingDesign(id, "long-pass", 550, Number.isFinite(aoi) ? aoi : 45, 8, 2.35, 1.46);
      report(ok, "refused: this surface cannot take a film", true);
      return;
    }
    if (scene.setSurfaceResponse(id, k, k)) {
      ctx.edited(true);
      rebuildFields();
      reread();
      ctx.redraw();
    }
  });
  label(ctx.bar, "");
  const passSeg = segmented<"long" | "short">(ctx.bar, [
    { value: "long", text: "long-pass", title: "transmit above the edge, reflect below" },
    { value: "short", text: "short-pass", title: "transmit below the edge, reflect above" },
  ], (p) => {
    if (film?.design) {
      writeDesign({ ...film.design, kind: p === "long" ? "long-pass" : "short-pass" });
      return;
    }
    const d = coat.front.dichroic;
    if (!d) return;
    writeDichroic({ ...d, longPass: p === "long" }, true);
  });

  let fieldKind = "";
  function rebuildFields() {
    const kind = coat.front.kind;
    if (kind === fieldKind) return;
    fieldKind = kind;
    ctx.fields.replaceChildren();
    fields = {};
    if (kind === "split") {
      fields.r = numberField(ctx.fields, "R", "%", { name: "R", step: 1, commit: (v) => writeSplit(v / 100, null, true), title: "reflectance" });
      fields.t = numberField(ctx.fields, "T", "%", { name: "T", step: 1, commit: (v) => writeSplit(null, v / 100, true), title: "transmittance" });
    } else if (kind === "film") {
      const d = (): FilmDesign =>
        film?.design ?? { kind: "long-pass", lambdaNm: 550, aoiDeg: film?.aoiDeg ?? 45, periods: 8, nHigh: 2.35, nLow: 1.46 };
      fields.edge = numberField(ctx.fields, "edge", "nm", { name: "edge", step: 5, commit: (v) => writeDesign({ ...d(), lambdaNm: v }), title: "where the stack should cross 50%, at the design angle" });
      fields.aoi = numberField(ctx.fields, "design ∠", "°", { name: "aoi", step: 1, commit: (v) => writeDesign({ ...d(), aoiDeg: v }), title: "the angle of incidence the stack is solved for" });
      fields.periods = numberField(ctx.fields, "periods", "", { name: "periods", step: 1, commit: (v) => writeDesign({ ...d(), periods: Math.round(v) }), title: "repeats of the layer pair: more is a deeper, steeper band" });
      fields.nHigh = numberField(ctx.fields, "n high", "", { name: "nHigh", step: 0.05, commit: (v) => writeDesign({ ...d(), nHigh: v }), title: "index of the high-index layer (TiO₂ is 2.35)" });
      fields.nLow = numberField(ctx.fields, "n low", "", { name: "nLow", step: 0.05, commit: (v) => writeDesign({ ...d(), nLow: v }), title: "index of the low-index layer (SiO₂ is 1.46)" });
    } else if (kind === "dichroic") {
      const d = () => coat.front.dichroic!;
      fields.edge = numberField(ctx.fields, "edge", "nm", { name: "edge", step: 5, commit: (v) => writeDichroic({ ...d(), edgeNm: v }, true), title: "the wavelength where the transition is half way" });
      fields.width = numberField(ctx.fields, "width", "nm", { name: "width", step: 1, commit: (v) => writeDichroic({ ...d(), widthNm: v }, true), title: "the full width of the transition" });
      fields.rMax = numberField(ctx.fields, "R max", "%", { name: "rMax", step: 1, commit: (v) => writeDichroic({ ...d(), rMax: v / 100 }, true), title: "reflectance in the reflect band" });
      fields.tMax = numberField(ctx.fields, "T max", "%", { name: "tMax", step: 1, commit: (v) => writeDichroic({ ...d(), tMax: v / 100 }, true), title: "transmittance in the pass band" });
    }
  }
  let fields: Record<string, ReturnType<typeof numberField>> = {};

  function writeSplit(r: number | null, t: number | null, settled: boolean): boolean {
    const f = coat.front;
    if (f.r == null || f.t == null) return false;
    const rr = clamp01(r ?? f.r);
    const tt = clamp01(t ?? f.t);
    const ok = scene.setSurfaceSplit(id, rr, tt);
    report(ok, "refused: reflectance and transmittance add up to more than 100%", settled);
    return ok;
  }

  function writeDesign(d: FilmDesign, settled = true): boolean {
    const kind = d.kind === "short-pass" ? "short-pass" : "long-pass";
    const ok = scene.setSurfaceCoatingDesign(id, kind, d.lambdaNm, d.aoiDeg, Math.max(1, Math.round(d.periods)), d.nHigh, d.nLow);
    report(ok, "refused: not a stack that can be built", settled);
    return ok;
  }

  function writeLayer(index: number, nm: number, settled: boolean): boolean {
    const ok = scene.setCoatingLayerThickness(id, index, Math.max(0, nm));
    report(ok, "refused: not a thickness a layer can have", settled);
    return ok;
  }

  function writeDichroic(d: Dichroic, settled: boolean): boolean {
    const ok = scene.setSurfaceDichroic(id, d.edgeNm, d.widthNm, d.longPass, d.rMax, d.tMax);
    report(ok, "refused: not a number the coating accepts", settled);
    return ok;
  }

  function report(ok: boolean, why: string, settled: boolean) {
    if (!ok) {
      ctx.note(why, true);
      return;
    }
    ctx.note("");
    ctx.edited(settled);
    reread();
    ctx.redraw();
  }

  function reread() {
    // The film first: a coated plate is a glass body, so it has no surface
    // response to read and would otherwise look like nothing at all.
    film = JSON.parse(scene.surfaceFilmJSON(id));
    const blank = { kind: "film", r: null, t: null, dichroic: null };
    const c = JSON.parse(scene.surfaceCoatingJSON(id)) ?? (film ? { front: { ...blank }, back: { ...blank } } : null);
    if (!c) return;
    coat = c;
    if (film) coat.front.kind = "film";
    rebuildFields();
    const aoi = film?.aoiDeg ?? 0;
    curve = scene.surfaceRtF64(id, false, grid, aoi);
    curveOnAxis = film && aoi > 0.5 ? scene.surfaceRtF64(id, false, grid, 0) : null;
    const src = JSON.parse(scene.firstSourceWavelengthsJSON()) as { nm: number }[] | null;
    lines = (src ?? []).map((l) => ({ nm: l.nm, color: scene.wavelengthCssColor(l.nm) }));
    atLines = scene.surfaceRtF64(id, false, new Float64Array(lines.map((l) => l.nm)), film?.aoiDeg ?? 0);

    const f = coat.front;
    kindSeg.set(["mirror", "split", "dichroic", "film", "clear", "absorb"].includes(f.kind) ? (f.kind as Kind) : null);
    const pass = film?.design ? (film.design.kind === "short-pass" ? "short" : "long") : f.dichroic ? (f.dichroic.longPass ? "long" : "short") : null;
    passSeg.set(pass);
    ctx.bar.querySelectorAll<HTMLButtonElement>("button[data-value='long'], button[data-value='short']")
      .forEach((b) => { b.style.display = pass ? "" : "none"; });
    if (film?.design) {
      fields.edge?.set(film.design.lambdaNm);
      fields.aoi?.set(film.design.aoiDeg);
      fields.periods?.set(film.design.periods);
      fields.nHigh?.set(film.design.nHigh);
      fields.nLow?.set(film.design.nLow);
    }
    fields.r?.set(f.r == null ? null : f.r * 100);
    fields.t?.set(f.t == null ? null : f.t * 100);
    if (f.dichroic) {
      fields.edge?.set(f.dichroic.edgeNm);
      fields.width?.set(f.dichroic.widthNm);
      fields.rMax?.set(f.dichroic.rMax * 100);
      fields.tMax?.set(f.dichroic.tMax * 100);
    }

    const parts = lines.map((l, i) => {
      const r = atLines[2 * i], t = atLines[2 * i + 1];
      return `<span style="color:${l.color}">${fmt(l.nm, 1)}</span> R <b>${pct(r)}</b> T <b>${pct(t)}</b>`;
    });
    const back = coat.back.kind !== f.kind && !film ? `<br>back face: ${coat.back.kind} (the buttons set both faces)` : "";
    const mirror = f.kind === "mirror" ? `reflectivity <b>${pct(f.r)}</b> · ` : "";
    if (film) {
      // The measured edge, not the requested one: when the stack cannot deliver
      // what it was asked for, these two disagree and the editor says so.
      const asked = film.design?.lambdaNm;
      const got = film.edgeNm;
      const miss = asked != null && got != null && Math.abs(got - asked) > 2;
      const edge = got == null ? "no edge in band" : `cuts at <b>${fmt(got, 0)} nm</b>`;
      const off = miss ? ` <span style="color:#ffb454">(asked ${fmt(asked!, 0)})</span>` : "";
      ctx.read.innerHTML =
        `${film.layers.length} layers · used at <b>${fmt(film.aoiDeg, 1)}°</b> · ${edge}${off} · peak R <b>${pct(film.peakR)}</b>` +
        `<br>${parts.length ? parts.join(" · ") : "no source wavelengths"}`;
      return;
    }
    ctx.read.innerHTML =
      `${mirror}${parts.length ? `at the source lines: ${parts.join(" · ")}` : "no source wavelengths"}${back}`;
  }

  const pct = (v: number | null) => (v == null || Number.isNaN(v) ? "—" : `${fmt(v * 100, 1)}%`);

  // Where the stack diagram sits, and how many px a nanometre of film is worth.
  // Set from the plot frame, which `draw` fills in, so the handles and the
  // drawing agree by construction.
  const STACK_H = 20;
  function stackGeom(w: number, h: number) {
    const x0 = box.l;
    const width = w - box.r - box.l;
    const total = film ? film.layers.reduce((a, l) => a + l.thicknessNm, 0) : 1;
    return { x0, w: width, y: h - box.b + 15, h: STACK_H, scale: width / Math.max(1, total) };
  }

  function draw(c: CanvasRenderingContext2D, w: number, h: number) {
    box.w = w;
    box.h = h;
    c.font = "10px ui-monospace, monospace";
    // Axes and grid.
    c.lineWidth = 1;
    c.strokeStyle = "rgba(120,150,190,0.12)";
    c.fillStyle = "rgba(205,214,228,0.55)";
    c.beginPath();
    for (const v of [0, 0.5, 1]) {
      c.moveTo(box.l, Math.round(Y(v)) + 0.5);
      c.lineTo(w - box.r, Math.round(Y(v)) + 0.5);
      c.fillText(`${v * 100}%`, 2, Y(v) + 3);
    }
    for (let nm = 400; nm <= NM1; nm += 50) {
      c.moveTo(Math.round(X(nm)) + 0.5, box.t);
      c.lineTo(Math.round(X(nm)) + 0.5, h - box.b);
      c.fillText(String(nm), X(nm) - 9, h - 4);
    }
    c.stroke();
    // The visible band under the plot.
    const stripY = h - box.b + 4;
    grid.forEach((nm, i) => {
      c.fillStyle = bandColor[i];
      c.fillRect(X(nm) - 1, stripY, Math.ceil(X(nm + STEP) - X(nm)) + 1, 8);
    });
    // The source's lines.
    lines.forEach((l) => {
      c.strokeStyle = l.color;
      c.globalAlpha = 0.55;
      c.beginPath(); c.moveTo(X(l.nm), box.t); c.lineTo(X(l.nm), h - box.b); c.stroke();
      c.globalAlpha = 1;
    });
    // The same stack on axis, behind everything: the distance between this and
    // the live curve is what tilting the part costs, drawn rather than claimed.
    if (curveOnAxis) {
      c.strokeStyle = "rgba(205,214,228,0.30)";
      c.lineWidth = 1.5;
      c.beginPath();
      let pen = false;
      grid.forEach((nm, i) => {
        const v = curveOnAxis![2 * i];
        if (Number.isNaN(v)) { pen = false; return; }
        if (pen) c.lineTo(X(nm), Y(v)); else c.moveTo(X(nm), Y(v));
        pen = true;
      });
      c.stroke();
    }
    // R and T. NaN (an uncoated or out-of-range face) leaves a gap.
    for (const [k, color, dash] of [[0, R_COLOR, []], [1, T_COLOR, [6, 3]]] as const) {
      c.strokeStyle = color;
      c.lineWidth = 2;
      c.setLineDash(dash as unknown as number[]);
      c.beginPath();
      let pen = false;
      grid.forEach((nm, i) => {
        const v = curve[2 * i + k];
        if (Number.isNaN(v)) { pen = false; return; }
        if (pen) c.lineTo(X(nm), Y(v));
        else c.moveTo(X(nm), Y(v));
        pen = true;
      });
      c.stroke();
      c.setLineDash([]);
    }
    // Where each line meets the curves.
    lines.forEach((l, i) => {
      for (const [k, color] of [[0, R_COLOR], [1, T_COLOR]] as const) {
        const v = atLines[2 * i + k];
        if (Number.isNaN(v)) continue;
        c.fillStyle = color;
        c.beginPath(); c.arc(X(l.nm), Y(v), 2.5, 0, Math.PI * 2); c.fill();
      }
    });
    // The stack itself, under the spectrum strip: layers left to right from the
    // incident medium inward, each as wide as it is thick. Boundaries drag.
    if (film && film.layers.length) {
      const g = stackGeom(w, h);
      c.fillStyle = "rgba(10,14,22,0.55)";
      c.fillRect(g.x0, g.y, g.w, g.h);
      let x = g.x0;
      film.layers.forEach((l, i) => {
        const dw = l.thicknessNm * g.scale;
        // High index dark, low index pale: the eye reads the period at a glance.
        const t = Math.min(1, Math.max(0, (l.n - 1.3) / 1.2));
        c.fillStyle = `rgba(${Math.round(150 - 90 * t)},${Math.round(190 - 110 * t)},${Math.round(235 - 120 * t)},0.85)`;
        c.fillRect(x, g.y, Math.max(1, dw - 0.6), g.h);
        x += dw;
        if (i === 0 && dw > 26) {
          c.fillStyle = "rgba(10,14,22,0.8)";
          c.fillText(l.name.slice(0, 3), g.x0 + 2, g.y + g.h - 3);
        }
      });
      c.strokeStyle = "rgba(205,214,228,0.45)";
      c.strokeRect(g.x0 + 0.5, g.y + 0.5, g.w - 1, g.h - 1);
      c.fillStyle = "rgba(205,214,228,0.55)";
      const total = film.layers.reduce((a, l) => a + l.thicknessNm, 0);
      c.fillText(`stack ${fmt(total, 0)} nm`, g.x0, g.y - 3);
      if (film.substrateN != null) {
        c.fillText(`n ${fmt(film.substrateN, 3)}`, g.x0 + g.w - 44, g.y - 3);
      }
    }
    // Legend.
    c.fillStyle = R_COLOR;
    c.fillText("R reflect", w - 130, 11);
    c.fillStyle = T_COLOR;
    c.fillText("T transmit", w - 68, 11);
    if (film) {
      c.fillStyle = "rgba(205,214,228,0.45)";
      c.fillText("on axis", w - 186, 11);
    }
    if (coat.front.kind === "refract") {
      c.fillStyle = "rgba(205,214,228,0.7)";
      c.fillText("uncoated: Fresnel reflection depends on the angle", box.l + 6, Y(0.5));
    }
  }

  function handles(): Handle[] {
    const f = coat.front;
    const out: Handle[] = [];
    if (f.kind === "split" && f.r != null && f.t != null) {
      out.push({ key: "r", x: X(470), y: Y(f.r), shape: "circle", color: R_COLOR, cursor: "ns-resize", title: `reflectance ${pct(f.r)}: drag up or down` });
      out.push({ key: "t", x: X(690), y: Y(f.t), shape: "circle", color: T_COLOR, cursor: "ns-resize", title: `transmittance ${pct(f.t)}: drag up or down` });
    }
    if (film && film.layers.length && box.w > 0) {
      // One handle per layer boundary. Dragging it changes the thickness of the
      // layer to its left, which is the only layer whose boundary that is.
      const g = stackGeom(box.w, box.h);
      let x = g.x0;
      film.layers.forEach((l, i) => {
        x += l.thicknessNm * g.scale;
        out.push({
          key: `f${i}`,
          x,
          y: g.y + g.h / 2,
          shape: "ring",
          color: "#cdd6e4",
          cursor: "ew-resize",
          title: `${l.name} ${fmt(l.thicknessNm, 1)} nm (n ${fmt(l.n, 3)}): drag to change its thickness`,
        });
      });
    }
    const d = f.dichroic;
    if (d) {
      const lo = d.edgeNm - d.widthNm / 2, hi = d.edgeNm + d.widthNm / 2;
      const inPlot = (nm: number) => Math.min(NM1 - 4, Math.max(NM0 + 4, nm));
      const reflectSide = d.longPass ? (NM0 + Math.max(NM0, lo)) / 2 : (NM1 + Math.min(NM1, hi)) / 2;
      const passSide = d.longPass ? (NM1 + Math.min(NM1, hi)) / 2 : (NM0 + Math.max(NM0, lo)) / 2;
      out.push({ key: "rmax", x: X(inPlot(reflectSide)), y: Y(d.rMax), shape: "circle", color: R_COLOR, cursor: "ns-resize", title: `peak reflectance ${pct(d.rMax)}: drag up or down` });
      out.push({ key: "tmax", x: X(inPlot(passSide)), y: Y(d.tMax), shape: "circle", color: T_COLOR, cursor: "ns-resize", title: `peak transmittance ${pct(d.tMax)}: drag up or down` });
      out.push({ key: "lo", x: X(inPlot(lo)), y: Y(0.5) + 16, shape: "ring", color: "#cdd6e4", cursor: "ew-resize", title: `transition ${fmt(d.widthNm, 1)} nm wide: drag sideways` });
      out.push({ key: "hi", x: X(inPlot(hi)), y: Y(0.5) + 16, shape: "ring", color: "#cdd6e4", cursor: "ew-resize", title: `transition ${fmt(d.widthNm, 1)} nm wide: drag sideways` });
      out.push({ key: "edge", x: X(inPlot(d.edgeNm)), y: Y(0.5), shape: "diamond", color: "#ffffff", cursor: "ew-resize", title: `edge at ${fmt(d.edgeNm, 1)} nm: drag sideways` });
    }
    return out;
  }

  let start: { r: number; t: number; loss: number } | null = null;
  function drag(h: Handle, x: number, y: number, phase: DragPhase) {
    const f = coat.front;
    if (phase === "start") {
      if (f.r != null && f.t != null) start = { r: f.r, t: f.t, loss: Math.max(0, 1 - f.r - f.t) };
      return;
    }
    if (phase === "end") {
      start = null;
      ctx.edited(true);
      reread();
      ctx.redraw();
      return;
    }
    if (h.key.startsWith("f") && film) {
      const i = Number(h.key.slice(1));
      const layer = film.layers[i];
      if (!layer) return;
      const g = stackGeom(box.w, box.h);
      // The boundary's own position, so the drag is absolute rather than
      // accumulating: everything left of this layer stays where it is.
      const left = g.x0 + film.layers.slice(0, i).reduce((a, l) => a + l.thicknessNm, 0) * g.scale;
      writeLayer(i, Math.max(0, (x - left) / g.scale), false);
      return;
    }
    const v = clamp01(valueAt(y));
    if ((h.key === "r" || h.key === "t") && start) {
      // Keep what the splitter absorbs; the other fraction takes up the change.
      const other = clamp01(1 - start.loss - v);
      if (h.key === "r") writeSplit(v, other, false);
      else writeSplit(other, v, false);
      return;
    }
    const d = f.dichroic;
    if (!d) return;
    const nm = nmAt(x);
    if (h.key === "edge") writeDichroic({ ...d, edgeNm: Math.min(NM1, Math.max(NM0, nm)) }, false);
    else if (h.key === "lo" || h.key === "hi") writeDichroic({ ...d, widthNm: Math.max(0.5, 2 * Math.abs(nm - d.edgeNm)) }, false);
    else if (h.key === "rmax") writeDichroic({ ...d, rMax: v }, false);
    else if (h.key === "tmax") writeDichroic({ ...d, tMax: v }, false);
  }

  reread();

  return {
    hint: "R solid, T dashed · drag the handles to shape the coating · scalar: no angle or polarization dependence",
    refresh: reread,
    draw,
    handles,
    drag,
  };
}
