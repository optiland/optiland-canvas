// The Fresnel lens editor: the lens's recipe (OpticStudio's "Fresnel 1") and
// the grooved profile it makes.
//
// A Fresnel lens is not edited by dragging its outline: every facet follows
// from the recipe (base radius, conic and aspheric terms, clear radius, pitch or
// groove depth, substrate, which way the grooves face, draft), so the fields
// are the editor and the drawing shows what they build. Each change goes to
// Rust, which rebuilds the profile or refuses the recipe and says why; a
// refused recipe leaves the lens as it was.
//
// The profile is drawn in the lens's own frame (optical axis to the right),
// with the axial direction stretched so grooves a fraction of a millimetre
// deep are visible next to a lens tens of millimetres tall.

import {
  fmt,
  label,
  numberField,
  segmented,
  type Editor,
  type EditorContext,
  type NumberField,
} from "./ui";

type Grooves = { ConstantPitch: { pitch_mm: number } } | { ConstantDepth: { depth_mm: number } };
type Recipe = {
  radius_mm: number;
  conic: number;
  aspheric: number[];
  clear_radius_mm: number;
  grooves: Grooves;
  substrate_mm: number;
  facing: "PlusX" | "MinusX";
  draft_deg: number;
};
type Info = {
  recipe: Recipe;
  nd: number | null;
  vd: number | null;
  n: number | null;
  efl: number | null;
  zones: number;
  outline: number[];
};

type GrooveMode = "pitch" | "depth";

export function createFresnelEditor(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  let info: Info = JSON.parse(scene.fresnelLensJSON(id));

  const modeOf = (g: Grooves): GrooveMode => ("ConstantPitch" in g ? "pitch" : "depth");
  const grooveValue = (g: Grooves) => ("ConstantPitch" in g ? g.ConstantPitch.pitch_mm : g.ConstantDepth.depth_mm);

  // --- controls -------------------------------------------------------------
  label(ctx.bar, "grooves");
  const grooveSeg = segmented<GrooveMode>(ctx.bar, [
    { value: "pitch", text: "pitch", title: "zones of equal width" },
    { value: "depth", text: "depth", title: "zones of equal groove depth" },
  ], (m) => {
    if (m === modeOf(info.recipe.grooves)) return;
    // Switch keeping the lens's look: the first zone's width becomes the
    // depth's zone, or the other way round, near enough.
    const r = info.recipe;
    const v = grooveValue(r.grooves);
    const next: Grooves = m === "pitch"
      ? { ConstantPitch: { pitch_mm: Math.max(0.05, Math.min(v * 10, r.clear_radius_mm / 4)) } }
      : { ConstantDepth: { depth_mm: Math.max(0.01, v / 10) } };
    commit({ grooves: next });
  });
  label(ctx.bar, "facing");
  const facingSeg = segmented<"PlusX" | "MinusX">(ctx.bar, [
    { value: "PlusX", text: "+X", title: "grooves on the +X face, flat face toward -X" },
    { value: "MinusX", text: "−X", title: "grooves on the -X face, flat face toward +X" },
  ], (f) => commit({ facing: f }));

  const fields: Record<string, NumberField> = {
    R: numberField(ctx.fields, "R", "mm", {
      name: "fresnelR",
      step: 1,
      title: "base radius of curvature; positive makes the grooved face convex (converging)",
      commit: (v) => commit({ radius_mm: v }),
    }),
    k: numberField(ctx.fields, "k", "", {
      name: "fresnelK",
      step: 0.1,
      title: "base conic constant",
      commit: (v) => commit({ conic: v }),
    }),
    A4: numberField(ctx.fields, "A4", "", {
      name: "fresnelA4",
      exponent: true,
      title: "4th-order aspheric term of the base (positive adds to the bulge)",
      commit: (v) => commit({ aspheric: terms(0, v) }),
    }),
    A6: numberField(ctx.fields, "A6", "", {
      name: "fresnelA6",
      exponent: true,
      title: "6th-order aspheric term of the base",
      commit: (v) => commit({ aspheric: terms(1, v) }),
    }),
    clear: numberField(ctx.fields, "clear r", "mm", {
      name: "fresnelClear",
      step: 0.5,
      title: "radius of the grooved region, which is the lens's radius",
      commit: (v) => commit({ clear_radius_mm: v }),
    }),
    groove: numberField(ctx.fields, "pitch", "mm", {
      name: "fresnelGroove",
      step: 0.05,
      title: "zone width (pitch) or groove depth, as the grooves setting says",
      commit: (v) => commit({
        grooves: modeOf(info.recipe.grooves) === "pitch"
          ? { ConstantPitch: { pitch_mm: v } }
          : { ConstantDepth: { depth_mm: v } },
      }),
    }),
    substrate: numberField(ctx.fields, "substrate", "mm", {
      name: "fresnelSubstrate",
      step: 0.1,
      title: "flat back face to the groove valleys",
      commit: (v) => commit({ substrate_mm: v }),
    }),
    draft: numberField(ctx.fields, "draft", "°", {
      name: "fresnelDraft",
      step: 0.5,
      title: "riser tilt from the optical axis (0: risers parallel to the axis)",
      commit: (v) => commit({ draft_deg: v }),
    }),
    nd: numberField(ctx.fields, "nd", "", {
      name: "fresnelNd",
      step: 0.01,
      title: "glass index at the d line (PMMA 1.4917)",
      commit: (v) => glass(v, info.vd ?? 57.4),
    }),
    vd: numberField(ctx.fields, "Vd", "", {
      name: "fresnelVd",
      step: 1,
      title: "glass Abbe number (PMMA 57.4)",
      commit: (v) => glass(info.nd ?? 1.4917, v),
    }),
  };
  const grooveLabel = fields.groove.input.parentElement!.querySelector("span")!;

  function terms(i: number, v: number): number[] {
    const a = [...info.recipe.aspheric];
    a[i] = v;
    return a;
  }

  function commit(change: Partial<Recipe>) {
    const why = scene.setFresnelLens(id, JSON.stringify({ ...info.recipe, ...change }));
    if (why) {
      ctx.note(`refused: ${why}`, true);
      return;
    }
    ctx.note("");
    ctx.edited(true);
    reread();
    ctx.redraw();
  }

  function glass(nd: number, vd: number) {
    if (!scene.setBodyGlassAbbe(id, nd, vd)) {
      ctx.note("refused: not a glass", true);
      return;
    }
    ctx.note("");
    ctx.edited(true);
    reread();
    ctx.redraw();
  }

  function reread() {
    const next = JSON.parse(scene.fresnelLensJSON(id));
    if (!next) return;
    info = next;
    const r = info.recipe;
    fields.R.set(r.radius_mm);
    fields.k.set(r.conic);
    fields.A4.set(r.aspheric[0]);
    fields.A6.set(r.aspheric[1]);
    fields.clear.set(r.clear_radius_mm);
    fields.groove.set(grooveValue(r.grooves));
    fields.substrate.set(r.substrate_mm);
    fields.draft.set(r.draft_deg);
    fields.nd.set(info.nd);
    fields.vd.set(info.vd);
    grooveLabel.textContent = modeOf(r.grooves) === "pitch" ? "pitch" : "depth";
    grooveSeg.set(modeOf(r.grooves));
    facingSeg.set(r.facing);
    ctx.read.innerHTML =
      `<b>${info.zones}</b> zones · f ≈ <b>${fmt(info.efl, 2)}</b> mm (R/(n−1), n ${fmt(info.n, 4)})`;
  }

  reread();
  return {
    hint: "the recipe sets every facet; each one is the base surface shifted along the axis",
    refresh: reread,
    draw(c, w, h) {
      const pts = info.outline;
      if (pts.length < 4) return;
      let x0 = Infinity, x1 = -Infinity, y1 = 0;
      for (let i = 0; i < pts.length; i += 2) {
        x0 = Math.min(x0, pts[i]);
        x1 = Math.max(x1, pts[i]);
        y1 = Math.max(y1, Math.abs(pts[i + 1]));
      }
      const pad = 16;
      const sy = (h - 2 * pad) / (2 * y1);
      // Stretch the axis (up to 6x, labelled) so the grooves can be seen.
      const sx = Math.min(6 * sy, (w - 2 * pad) / Math.max(x1 - x0, 1e-6));
      const ox = w / 2 - ((x0 + x1) / 2) * sx;
      const oy = h / 2;
      c.beginPath();
      for (let i = 0; i < pts.length; i += 2) {
        const X = ox + pts[i] * sx, Y = oy - pts[i + 1] * sy;
        if (i === 0) c.moveTo(X, Y);
        else c.lineTo(X, Y);
      }
      c.closePath();
      c.fillStyle = "rgba(127,214,255,0.14)";
      c.fill();
      c.strokeStyle = "#7fd6ff";
      c.lineWidth = 1;
      c.stroke();
      c.strokeStyle = "rgba(150,175,210,0.35)";
      c.setLineDash([4, 4]);
      c.beginPath();
      c.moveTo(pad, oy);
      c.lineTo(w - pad, oy);
      c.stroke();
      c.setLineDash([]);
      const stretch = sx / sy;
      if (stretch > 1.05) {
        c.fillStyle = "rgba(205,214,228,0.55)";
        c.font = "10px ui-monospace, monospace";
        c.fillText(`axial ×${fmt(stretch, 1)}`, 8, h - 6);
      }
    },
    handles: () => [],
    drag() {},
  };
}
