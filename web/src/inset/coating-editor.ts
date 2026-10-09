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
//
// For a film, the angles the scene's rays actually meet the face at are
// histogrammed (`faceIncidenceJSON`, one f64 trace with the event log on) and
// drawn top right, and the R and T those rays see on average, the curves
// weighted by that histogram, are drawn dotted beside the single-angle ones.
//
// A film's stack is edited under the fields (coating-fidelity spec CF-11):
// each layer's material from the catalogue, its thickness and optical
// thickness, its place in the stack; what is behind the stack; and a report of
// the design record, every material without data at a launched wavelength and
// every refusal a Strict trace would meet. The plot shows the unpolarized mean
// or s and p apart, against wavelength or against angle. A measured table is
// imported from CSV through a dialog that asks for everything a CSV leaves out.

import { filmPicker } from "./film-picker";
import {
  button,
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

type StackLayer = {
  name: string;
  key: string | null;
  n: number;
  k: number;
  thicknessNm: number;
  quarterWaves: number;
  incoherent: boolean;
  covers: boolean;
  rangeNm: [number, number] | null;
};
type Stack = {
  layers: StackLayer[];
  backing: { kind: "medium" | "plate" | "opaque"; substrate: string | null; thicknessMm: number | null };
  design: string | null;
  lambdaNm: number;
  report: string[];
};
type Table = {
  lambda_um: number[];
  aoi_deg: number[];
  power_resolution: string;
  scope: string;
  side: string;
  provenance?: { source_kind: string; reference?: string; note?: string };
};

/** The angles an angle sweep samples, degrees. */
const ANGLES = Array.from({ length: 90 }, (_, i) => i);
const WARN = "#ffb454";

type IncidenceSide = { weights: number[]; flux: number; hits: number; meanDeg: number | null; r: number[] | null; t: number[] | null };
type Incidence = { binDeg: number; front: IncidenceSide; back: IncidenceSide };

export function createCoatingEditor(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  const grid = new Float64Array(Math.round((NM1 - NM0) / STEP) + 1).map((_, i) => NM0 + i * STEP);
  let coat: Coat = JSON.parse(scene.surfaceCoatingJSON(id));
  let curve: Float64Array<ArrayBufferLike> = new Float64Array(0);
  // The same stack on axis, drawn behind the working curve: the gap between the
  // two is the whole reason this editor exists.
  let curveOnAxis: Float64Array<ArrayBufferLike> | null = null;
  // The angles the scene's rays meet this face at, and the curves they see.
  let incidence: Incidence | null = null;
  let film: Film | null = null;
  let lines: { nm: number; color: string }[] = [];
  let atLines: Float64Array<ArrayBufferLike> = new Float64Array(0);
  // The stack as the editor shows it, a measured table, and the plot's mode.
  let stack: Stack | null = null;
  let table: Table | null = null;
  let plotMode: "lambda" | "angle" = "lambda";
  let spMode = false;
  let angleNm = 550;
  // `[Rs, Rp, Ts, Tp]` per sample: along the wavelength grid at the plotted
  // angle, or along `ANGLES` at `angleNm`.
  let spLambda: Float64Array<ArrayBufferLike> | null = null;
  let spAngle: Float64Array<ArrayBufferLike> | null = null;
  let stackBox: HTMLDivElement | null = null;
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

  label(ctx.bar, "");
  const plotSeg = segmented<"lambda" | "angle">(ctx.bar, [
    { value: "lambda", text: "vs λ", title: "R and T against wavelength, at the angle the surface is used at" },
    { value: "angle", text: "vs ∠", title: "R and T against the angle of incidence, at one wavelength" },
  ], (m) => { plotMode = m; reread(); ctx.redraw(); });
  const polSeg = segmented<"mean" | "sp">(ctx.bar, [
    { value: "mean", text: "mean", title: "unpolarized: the mean of the s and p powers" },
    { value: "sp", text: "s, p", title: "s (solid) and p (dotted) apart" },
  ], (m) => { spMode = m === "sp"; reread(); ctx.redraw(); });
  button(ctx.bar, "table…", "import measured or vendor R and T from a CSV file, with its angle, side and media", () => openTableDialog());

  let fieldKind = "";
  function rebuildFields() {
    const kind = coat.front.kind;
    if (kind === fieldKind) return;
    fieldKind = kind;
    ctx.fields.replaceChildren();
    fields = {};
    stackBox = null;
    if (kind === "film" || kind === "table") {
      fields.at = numberField(ctx.fields, "at", "nm", {
        name: "angleNm",
        step: 10,
        commit: (v) => { angleNm = v; if (plotMode === "angle") { reread(); ctx.redraw(); } },
        title: "the wavelength an angle sweep is plotted at",
      });
    }
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
      stackBox = document.createElement("div");
      stackBox.className = "cv-stack";
      stackBox.id = "cvStack";
      ctx.fields.appendChild(stackBox);
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
    table = coat.front.kind === "table" ? (JSON.parse(scene.surfaceTableJSON(id)) as Table | null) : null;
    stack = film ? (JSON.parse(scene.filmStackJSON(id)) as Stack | null) : null;
    rebuildFields();
    const aoi = film?.aoiDeg ?? table?.aoi_deg[0] ?? 0;
    curve = scene.surfaceRtF64(id, false, grid, aoi);
    const plottable = !!(film || table);
    if (!plottable) { plotMode = "lambda"; spMode = false; }
    ctx.bar.querySelectorAll<HTMLButtonElement>("button[data-value='lambda'], button[data-value='angle'], button[data-value='mean'], button[data-value='sp']")
      .forEach((b) => { b.style.display = plottable ? "" : "none"; });
    plotSeg.set(plotMode);
    polSeg.set(spMode ? "sp" : "mean");
    fields.at?.set(angleNm);
    spLambda = plottable && spMode && plotMode === "lambda"
      ? scene.coatingSpF64(id, false, grid, new Float64Array(grid.length).fill(aoi)) : null;
    spAngle = plottable && plotMode === "angle"
      ? scene.coatingSpF64(id, false, new Float64Array(ANGLES.length).fill(angleNm), new Float64Array(ANGLES)) : null;
    renderStack();
    curveOnAxis = film && aoi > 0.5 ? scene.surfaceRtF64(id, false, grid, 0) : null;
    incidence = null;
    if (film) {
      try {
        incidence = JSON.parse(scene.faceIncidenceJSON(id, grid)) as Incidence | null;
      } catch {
        incidence = null;
      }
    }
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
      const hit = incidence && incidence.front.hits > 0 ? incidence.front : null;
      const seen = hit
        ? `<br>the rays meet it at <b>${fmt(hit.meanDeg ?? NaN, 1)}°</b> on average (${hit.hits} hits; histogram top right, their R and T dotted)`
        : "";
      ctx.read.innerHTML =
        `${film.layers.length} layers · used at <b>${fmt(film.aoiDeg, 1)}°</b> · ${edge}${off} · peak R <b>${pct(film.peakR)}</b>` +
        `<br>${parts.length ? parts.join(" · ") : "no source wavelengths"}${seen}`;
      return;
    }
    if (table) {
      const n = table.aoi_deg.length;
      const at = n === 1 ? `at ${fmt(table.aoi_deg[0], 1)}° only` : `${fmt(table.aoi_deg[0], 1)}° to ${fmt(table.aoi_deg[n - 1], 1)}°`;
      const span = `${fmt(table.lambda_um[0] * 1000, 0)} to ${fmt(table.lambda_um[table.lambda_um.length - 1] * 1000, 0)} nm`;
      const src = table.provenance ? ` · ${table.provenance.source_kind}${table.provenance.reference ? `: ${table.provenance.reference}` : ""}` : "";
      const scope = table.scope === "FiniteAssembly" ? "whole part" : "one interface";
      ctx.read.innerHTML =
        `measured table <b>${at}</b>, ${span}, ${table.power_resolution}, ${scope}, ${table.side} side${src}` +
        `<br>${parts.length ? parts.join(" · ") : "no source wavelengths"}`;
      return;
    }
    ctx.read.innerHTML =
      `${mirror}${parts.length ? `at the source lines: ${parts.join(" · ")}` : "no source wavelengths"}${back}`;
  }

  /** A stack edit's reply: the reason it was refused, or a fresh read. */
  function edited(replyJson: string) {
    const r = JSON.parse(replyJson) as { ok?: boolean; error?: string };
    if (r.error) {
      ctx.note(`refused: ${r.error}`, true);
      renderStack();
      return;
    }
    ctx.note("");
    ctx.edited(true);
    reread();
    ctx.redraw();
  }

  function el<K extends keyof HTMLElementTagNameMap>(tag: K, parent: HTMLElement, cls?: string, text?: string): HTMLElementTagNameMap[K] {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    parent.appendChild(e);
    return e;
  }

  /** The stack panel: one row per layer, a row to add one, the backing, and
   *  the report. Rebuilt on every read. */
  function renderStack() {
    const box = stackBox;
    if (!box) return;
    box.replaceChildren();
    const st = stack;
    if (!st) return;
    const n = st.layers.length;
    st.layers.forEach((l, i) => {
      const row = el("div", box, "cv-stack-row");
      row.dataset.layer = String(i);
      if (!l.covers) row.classList.add("warn");
      el("span", row, "cv-stack-i", String(i + 1));
      const pick = filmPicker(row, scene, {
        title: "the layer's material, from the catalogue",
        placeholder: l.key ? undefined : `${l.name} (own)`,
        onPick: (key) => edited(scene.setCoatingLayerMaterial(id, i, key)),
      });
      pick.set(l.key);
      const t = el("input", row, "cv-stack-t");
      t.type = "number";
      t.step = "1";
      t.min = "0";
      t.value = String(Number(l.thicknessNm.toFixed(2)));
      t.title = `physical thickness, nm${l.incoherent ? " (an incoherent slab)" : ""}`;
      t.addEventListener("change", () => {
        const v = Number(t.value);
        if (!Number.isFinite(v) || !scene.setCoatingLayerThickness(id, i, v)) {
          ctx.note("refused: not a thickness a layer can have", true);
          renderStack();
          return;
        }
        ctx.note("");
        ctx.edited(true);
        reread();
        ctx.redraw();
      });
      el("span", row, "cv-stack-u", "nm");
      const qw = el("span", row, "cv-stack-qw", `${fmt(l.quarterWaves, 2)} QW`);
      qw.title = `optical thickness at ${fmt(st.lambdaNm, 0)} nm, in quarter waves (n ${fmt(l.n, 3)}${l.k > 0 ? `, k ${fmt(l.k, 3)}` : ""})`;
      const up = button(row, "↑", "move this layer one place towards the incident medium", () => edited(scene.moveCoatingLayer(id, i, i - 1)));
      up.disabled = i === 0;
      const down = button(row, "↓", "move this layer one place towards the substrate", () => edited(scene.moveCoatingLayer(id, i, i + 1)));
      down.disabled = i === n - 1;
      button(row, "×", "remove this layer", () => edited(scene.removeCoatingLayer(id, i)));
    });
    const add = el("div", box, "cv-stack-row");
    el("span", add, "cv-stack-i", "+");
    filmPicker(add, scene, {
      title: "add a layer next to the substrate, 100 nm thick",
      placeholder: "add a layer of…",
      onPick: (key) => edited(scene.insertCoatingLayer(id, n, key, 100)),
    }).el.id = "cvAddLayer";
    const back = el("div", box, "cv-stack-row");
    el("span", back, "cv-stack-i", "|");
    const kind = el("select", back, "cv-film-pick");
    kind.id = "cvBacking";
    for (const [v, text] of [["medium", "behind: the far medium"], ["plate", "behind: a plate"], ["opaque", "behind: an opaque substrate"]]) {
      el("option", kind, undefined, text).value = v;
    }
    kind.value = st.backing.kind;
    const subst = filmPicker(back, scene, {
      title: "the substrate's material",
      placeholder: st.backing.substrate ?? "substrate…",
      onPick: (key) => edited(scene.setCoatingBacking(id, kind.value, key, st.backing.thicknessMm ?? 3)),
    });
    subst.set(null);
    kind.addEventListener("change", () => {
      if (kind.value === "medium") edited(scene.setCoatingBacking(id, "medium", "", 0));
      else if (subst.el.value) edited(scene.setCoatingBacking(id, kind.value, subst.el.value, st.backing.thicknessMm ?? 3));
      else ctx.note("pick the substrate's material", true);
    });
    const rep = el("div", box, "cv-report");
    rep.id = "cvReport";
    for (const line of st.report) {
      const p = el("div", rep, undefined, line);
      if (!line.startsWith("design:")) p.style.color = WARN;
    }
  }

  /** The CSV import dialog: a measured or vendor table, with everything a CSV
   *  does not say asked for (coating-fidelity spec CF-09). */
  function openTableDialog() {
    document.getElementById("tableDialog")?.remove();
    const dlg = el("div", document.body, "cv-dialog");
    dlg.id = "tableDialog";
    dlg.innerHTML = `
      <div class="cv-dialog-title">import a measured table</div>
      <textarea id="tdText" rows="6" placeholder="paste the CSV here, or choose a file"></textarea>
      <input type="file" id="tdFile" accept=".csv,.txt,.dat" />
      <div class="cv-dialog-grid">
        <label>wavelength column <input id="tdLamCol" type="number" min="0" value="0" /></label>
        <label>unit <select id="tdLamUnit"><option value="Nm">nm</option><option value="Um">µm</option></select></label>
        <label>values <select id="tdValUnit"><option value="Percent">percent</option><option value="Fraction">fraction</option></select></label>
        <label>data <select id="tdKind">
          <option value="lossless">T only (R = 1 − T, lossless)</option>
          <option value="unpolarized">R, T unpolarized</option>
          <option value="scalar">R, T polarization not stated</option>
          <option value="sp">Rs, Rp, Ts, Tp</option></select></label>
        <label>value columns <input id="tdCols" type="text" value="1" title="zero-based, comma-separated, in the order the data choice names them" /></label>
        <label>angle <select id="tdAoiMode"><option value="fixed">measured at</option><option value="column">from column</option></select>
          <input id="tdAoi" type="number" value="45" /></label>
        <label>light from <select id="tdSide"><option value="Front">the front</option><option value="Back">the back</option><option value="Either">either side (declared)</option></select></label>
        <label>covers <select id="tdScope"><option value="FiniteAssembly">the whole part</option><option value="Interface">one interface</option></select></label>
        <label>n in <input id="tdNin" type="number" step="0.01" value="1" /></label>
        <label>n out <input id="tdNout" type="number" step="0.01" value="1" /></label>
        <label>outside the data <select id="tdDomain"><option value="">as the profile says</option><option value="Refuse">refuse</option><option value="Clamp">read the edge</option></select></label>
        <label>source <input id="tdSource" type="text" placeholder="datasheet, part number" /></label>
      </div>
      <div id="tdError" class="cv-dialog-error"></div>
      <div class="cv-dialog-buttons"><button class="m4btn" id="tdImport" type="button">import</button><button class="m4btn" id="tdCancel" type="button">cancel</button></div>`;
    const q = <T extends HTMLElement>(sel: string) => dlg.querySelector(sel) as T;
    q<HTMLInputElement>("#tdFile").addEventListener("change", async (e) => {
      const f = (e.target as HTMLInputElement).files?.[0];
      if (f) q<HTMLTextAreaElement>("#tdText").value = await f.text();
    });
    q<HTMLButtonElement>("#tdCancel").addEventListener("click", () => dlg.remove());
    q<HTMLButtonElement>("#tdImport").addEventListener("click", () => {
      const num = (s: string) => Number(q<HTMLInputElement>(s).value);
      const cols = q<HTMLInputElement>("#tdCols").value.split(",").map((v) => Number(v.trim()));
      const kind = q<HTMLSelectElement>("#tdKind").value;
      const need = kind === "sp" ? 4 : kind === "lossless" ? 1 : 2;
      const err = q<HTMLDivElement>("#tdError");
      if (cols.length !== need || cols.some((c) => !Number.isInteger(c) || c < 0)) {
        err.textContent = `the data choice needs ${need} column number${need > 1 ? "s" : ""}`;
        return;
      }
      const columns =
        kind === "sp" ? { Sp: { rs: cols[0], rp: cols[1], ts: cols[2], tp: cols[3] } }
        : kind === "lossless" ? { UnpolarizedLosslessT: { t: cols[0] } }
        : kind === "scalar" ? { Scalar: { r: cols[0], t: cols[1] } }
        : { Unpolarized: { r: cols[0], t: cols[1] } };
      const aoi = q<HTMLSelectElement>("#tdAoiMode").value === "fixed" ? { Fixed: num("#tdAoi") } : { Column: Math.round(num("#tdAoi")) };
      const source = q<HTMLInputElement>("#tdSource").value.trim();
      const domain = q<HTMLSelectElement>("#tdDomain").value;
      const map: Record<string, unknown> = {
        lambda_column: Math.round(num("#tdLamCol")),
        lambda_unit: q<HTMLSelectElement>("#tdLamUnit").value,
        aoi,
        columns,
        value_unit: q<HTMLSelectElement>("#tdValUnit").value,
        side: q<HTMLSelectElement>("#tdSide").value,
        scope: q<HTMLSelectElement>("#tdScope").value,
        media: { incident: num("#tdNin"), exit: num("#tdNout") },
        reflection: "Path",
      };
      if (domain) map.out_of_domain = domain;
      if (source) map.provenance = { source_kind: "Supplied", reference: source };
      const reply = JSON.parse(scene.importMeasuredCsv(id, "", q<HTMLTextAreaElement>("#tdText").value, JSON.stringify(map))) as { ok?: boolean; error?: string };
      if (reply.error) {
        err.textContent = reply.error;
        return;
      }
      dlg.remove();
      fieldKind = "";
      ctx.note("");
      ctx.edited(true);
      reread();
      ctx.redraw();
    });
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

  /** The stack under the plot: layers left to right from the incident medium
   *  inward, each as wide as it is thick. Boundaries drag. */
  function drawStack(c: CanvasRenderingContext2D, w: number, h: number) {
    // The stack itself, under the spectrum strip: layers left to right from the
    // incident medium inward, each as wide as it is thick. Boundaries drag.
    if (!film || !film.layers.length) return;
    {
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
  }

  /** R and T against the angle of incidence at `angleNm`: the unpolarized
   *  mean, or s solid and p dotted. */
  function drawAngle(c: CanvasRenderingContext2D, w: number, h: number) {
    const sp = spAngle;
    if (!sp) return;
    const XA = (deg: number) => box.l + (deg / 90) * (box.w - box.l - box.r);
    c.lineWidth = 1;
    c.strokeStyle = "rgba(120,150,190,0.12)";
    c.fillStyle = "rgba(205,214,228,0.55)";
    c.beginPath();
    for (const v of [0, 0.5, 1]) {
      c.moveTo(box.l, Math.round(Y(v)) + 0.5);
      c.lineTo(w - box.r, Math.round(Y(v)) + 0.5);
      c.fillText(`${v * 100}%`, 2, Y(v) + 3);
    }
    for (let d = 0; d <= 90; d += 15) {
      c.moveTo(Math.round(XA(d)) + 0.5, box.t);
      c.lineTo(Math.round(XA(d)) + 0.5, h - box.b);
      c.fillText(`${d}°`, XA(d) - 7, h - box.b + 12);
    }
    c.stroke();
    // The angle the surface is used at.
    const used = film?.aoiDeg ?? table?.aoi_deg[0];
    if (used != null) {
      c.strokeStyle = "rgba(205,214,228,0.4)";
      c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(XA(used), box.t); c.lineTo(XA(used), h - box.b); c.stroke();
      c.setLineDash([]);
    }
    const series = (pick: (i: number) => number, color: string, dash: number[], width: number) => {
      c.strokeStyle = color;
      c.lineWidth = width;
      c.setLineDash(dash);
      c.beginPath();
      let pen = false;
      ANGLES.forEach((d, i) => {
        const v = pick(i);
        if (Number.isNaN(v)) { pen = false; return; }
        if (pen) c.lineTo(XA(d), Y(v)); else c.moveTo(XA(d), Y(v));
        pen = true;
      });
      c.stroke();
      c.setLineDash([]);
    };
    if (spMode) {
      series((i) => sp[4 * i], R_COLOR, [], 2);
      series((i) => sp[4 * i + 1], R_COLOR, [2, 3], 2);
      series((i) => sp[4 * i + 2], T_COLOR, [], 1.5);
      series((i) => sp[4 * i + 3], T_COLOR, [2, 3], 1.5);
    } else {
      series((i) => 0.5 * (sp[4 * i] + sp[4 * i + 1]), R_COLOR, [], 2);
      series((i) => 0.5 * (sp[4 * i + 2] + sp[4 * i + 3]), T_COLOR, [6, 3], 2);
    }
    c.fillStyle = "rgba(205,214,228,0.7)";
    c.fillText(`at ${fmt(angleNm, 0)} nm`, box.l + 4, box.t + 10);
  }

  function draw(c: CanvasRenderingContext2D, w: number, h: number) {
    box.w = w;
    box.h = h;
    c.font = "10px ui-monospace, monospace";
    if (plotMode === "angle" && spAngle) {
      drawAngle(c, w, h);
      drawStack(c, w, h);
      c.fillStyle = R_COLOR;
      c.fillText(spMode ? "Rs ─ Rp ┄" : "R reflect", w - 130, 11);
      c.fillStyle = T_COLOR;
      c.fillText(spMode ? "Ts ─ Tp ┄" : "T transmit", w - 68, 11);
      return;
    }
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
    // s solid and p dotted, thinner, over the mean.
    if (spLambda) {
      const sp = spLambda;
      for (const [k, color, dash] of [[0, R_COLOR, []], [1, R_COLOR, [2, 3]], [2, T_COLOR, []], [3, T_COLOR, [2, 3]]] as const) {
        c.strokeStyle = color;
        c.globalAlpha = 0.8;
        c.lineWidth = 1;
        c.setLineDash(dash as unknown as number[]);
        c.beginPath();
        let pen = false;
        grid.forEach((nm, i) => {
          const v = sp[4 * i + k];
          if (Number.isNaN(v)) { pen = false; return; }
          if (pen) c.lineTo(X(nm), Y(v)); else c.moveTo(X(nm), Y(v));
          pen = true;
        });
        c.stroke();
        c.setLineDash([]);
        c.globalAlpha = 1;
      }
    }
    // The curves the rays actually see: R and T averaged over the angles
    // they meet the face at, dotted.
    const seen = incidence && incidence.front.hits > 0 ? incidence.front : null;
    if (seen && seen.r && seen.t) {
      for (const [vals, color] of [[seen.r, R_COLOR], [seen.t, T_COLOR]] as const) {
        c.strokeStyle = color;
        c.lineWidth = 1.5;
        c.setLineDash([2, 3]);
        c.beginPath();
        let pen = false;
        grid.forEach((nm, i) => {
          const v = vals[i];
          if (v == null || Number.isNaN(v)) { pen = false; return; }
          if (pen) c.lineTo(X(nm), Y(v)); else c.moveTo(X(nm), Y(v));
          pen = true;
        });
        c.stroke();
        c.setLineDash([]);
      }
      // The histogram of those angles, 0 to 90 degrees, top right.
      const hw = 90, hh = 24, hx = w - box.r - hw - 4, hy = box.t + 18;
      const peak = Math.max(...seen.weights, 1e-300);
      c.fillStyle = "rgba(10,14,22,0.6)";
      c.fillRect(hx - 2, hy - 2, hw + 4, hh + 14);
      c.fillStyle = "rgba(205,214,228,0.75)";
      const bw = hw / seen.weights.length;
      seen.weights.forEach((v, i) => {
        const bh = (v / peak) * hh;
        if (bh > 0) c.fillRect(hx + i * bw, hy + hh - bh, Math.max(1, bw - 0.5), bh);
      });
      c.fillStyle = "rgba(205,214,228,0.55)";
      c.fillText("0°", hx - 1, hy + hh + 10);
      c.fillText("90°", hx + hw - 16, hy + hh + 10);
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
    drawStack(c, w, h);
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
