// The modulator editor: a DMD or a phase-only LCoS SLM, its pattern beside
// the diffraction orders it makes.
//
// Left, the pattern as the device shows it: a DMD's ON mirrors light and OFF
// dark, an SLM's phase as grey from 0 to 2π. Right, the propagating orders in
// direction cosines, the unit circle the limit of propagation: each dot an
// order, its size and brightness its share of the reflected power on a log
// scale, the cross the incident direction. The orders come from the pattern's
// unit cell in closed form (oc_core::modulator::unit_cell_orders, Harvey's
// renormalization) at the wavelength set here; a DMD is lit along the
// direction its ON mirrors send onto the normal, an SLM at normal incidence.
//
// In the scene the ray tracer and the preview take the geometric limit (each
// mirror a tilted reflector, each electrode a phase-gradient deflector); the
// orders drawn here are what the wave path carries.

import { button, fmt, label, numberField, segmented, type Editor, type EditorContext, type NumberField } from "./ui";

type Dmd = { tilt_deg: number; tilt_axis: "Diagonal" | "Side"; mirror_fill: number; reflectance: number; window_transmittance: number };
type Lcos = {
  director_deg: number;
  lut: unknown;
  design_lambda_um: number;
  pixel_fill: number;
  gap_amplitude: number;
  gap_phase_rad: number;
  fringing_um: number;
  reflectance: number;
};
type Device = { pixels: [number, number]; pitch_um: number; kind: { Dmd?: Dmd; Lcos?: Lcos }; pattern: Record<string, unknown> };
type RayModel = "Geometric" | { Orders: { max_orders: number; min_efficiency: number } };
type Modulator = { kind: "dmd" | "slm"; device: Device; widthMm: number; heightMm: number; rayModel: RayModel };
type Order = { m: number; n: number; alpha: number; beta: number; efficiency: number };
type Orders = {
  incidence: [number, number];
  orders: Order[];
  propagating: number;
  total: number;
  period: [number, number];
  note: string | null;
  pitchOverLambda: number;
  error?: string;
};
type Preset = { name: string; pitchUm: number; tiltDeg: number; axis: "Diagonal" | "Side"; fNumber: number; onStateFill: number };

type PatternKind = "on" | "off" | "binary" | "flat" | "blazed" | "lens" | "vortex" | "imported" | "other";

/** What kind of pattern the device shows, as the editor's buttons name it. */
function patternKind(m: Modulator): PatternKind {
  const p = m.device.pattern;
  if ("Uniform" in p) {
    const v = (p.Uniform as { value: number }).value;
    return m.kind === "dmd" ? (v === 0 ? "off" : "on") : "flat";
  }
  if ("Explicit" in p) return "imported";
  const g = p.Generated as Record<string, unknown>;
  if ("BinaryGrating" in g) return "binary";
  if ("BlazedGrating" in g) return "blazed";
  if ("FresnelLens" in g) return "lens";
  if ("Vortex" in g) return "vortex";
  return "other";
}

export function createModulatorEditor(ctx: EditorContext): Editor {
  const { scene, id } = ctx;
  let m: Modulator = JSON.parse(scene.modulatorJSON(id));
  const presets: Preset[] = JSON.parse(scene.modulatorPresetsJSON());
  let lambdaNm = m.kind === "slm" && m.device.kind.Lcos ? m.device.kind.Lcos.design_lambda_um * 1000 : 633;
  // The pattern's own numbers, remembered while another kind is shown.
  let period = 8, angle = 0, duty = 0.5, focal = 200, charge = 1;
  let orders: Orders | null = null;
  let picture: HTMLCanvasElement | null = null;

  const read = () => {
    const next = JSON.parse(scene.modulatorJSON(id));
    if (next) m = next;
    const g = (m.device.pattern.Generated ?? {}) as Record<string, Record<string, number>>;
    const b = g.BinaryGrating ?? g.BlazedGrating;
    if (b) {
      period = b.period_px;
      angle = b.angle_deg;
      if (g.BinaryGrating) duty = g.BinaryGrating.duty;
    }
    if (g.FresnelLens) focal = g.FresnelLens.focal_mm;
    if (g.Vortex) charge = g.Vortex.charge;
  };
  const answer = (reply: string): boolean => {
    const r = JSON.parse(reply);
    if (r.error) {
      ctx.note(`refused: ${r.error}`, true);
      return false;
    }
    ctx.note("");
    read();
    ctx.edited(true);
    refresh();
    return true;
  };
  const setDevice = (change: (d: Device) => void) => {
    const d: Device = JSON.parse(JSON.stringify(m.device));
    change(d);
    answer(scene.setModulatorDevice(id, JSON.stringify({ pixels: d.pixels, pitch_um: d.pitch_um, kind: d.kind })));
  };
  const setPattern = (k: PatternKind) => {
    const p =
      k === "on" ? { Uniform: { value: 1 } }
      : k === "off" ? { Uniform: { value: 0 } }
      : k === "flat" ? { Uniform: { value: 0 } }
      : k === "binary" ? { Generated: { BinaryGrating: { period_px: period, duty, angle_deg: angle, phase: 0 } } }
      : k === "blazed" ? { Generated: { BlazedGrating: { period_px: period, angle_deg: angle } } }
      : k === "lens" ? { Generated: { FresnelLens: { focal_mm: focal } } }
      : k === "vortex" ? { Generated: { Vortex: { charge: Math.round(charge) } } }
      : null;
    if (p) answer(scene.setModulatorPattern(id, JSON.stringify(p)));
  };

  // The bar: the pattern, and for a DMD the TI presets.
  label(ctx.bar, "pattern");
  const kinds: { value: PatternKind; text: string; title: string }[] = m.kind === "dmd"
    ? [
        { value: "on", text: "all ON", title: "every mirror tilted ON" },
        { value: "off", text: "all OFF", title: "every mirror tilted OFF" },
        { value: "binary", text: "grating", title: "a binary amplitude grating: ON where frac(u/period) < duty, u along the angle" },
      ]
    : [
        { value: "flat", text: "flat", title: "every pixel at grey level 0: a mirror for the modulated beam" },
        { value: "blazed", text: "blazed", title: "a blazed phase grating 2π·frac(u/period), u along the angle: steers the modulated beam" },
        { value: "lens", text: "lens", title: "a Fresnel lens −πr²/(λ₀f), wrapped" },
        { value: "vortex", text: "vortex", title: "a spiral phase ℓ·atan2(y, x): a beam with a dark core" },
      ];
  const kindPick = segmented<PatternKind>(ctx.bar, kinds, setPattern);
  // An image of your own, resampled to the device's pixels (nearest
  // neighbour, its top row the device's top): a DMD turns a mirror ON where
  // the luminance is at least half; an SLM maps the luminance onto its grey
  // levels. Stored packed in the file.
  const file = document.createElement("input");
  file.type = "file";
  file.accept = "image/*";
  file.style.display = "none";
  file.dataset.name = "modulatorImport";
  file.onchange = async () => {
    const f = file.files?.[0];
    file.value = "";
    if (!f) return;
    try {
      const bmp = await createImageBitmap(f);
      const s = Math.min(1, 4096 / Math.max(bmp.width, bmp.height));
      const w = Math.max(1, Math.round(bmp.width * s)), h = Math.max(1, Math.round(bmp.height * s));
      const cv = document.createElement("canvas");
      cv.width = w;
      cv.height = h;
      const c2 = cv.getContext("2d")!;
      c2.drawImage(bmp, 0, 0, w, h);
      const data = c2.getImageData(0, 0, w, h).data;
      answer(scene.importModulatorBitmap(id, w, h, new Uint8Array(data.buffer), 128));
    } catch (e) {
      ctx.note(`refused: the image did not load (${e})`, true);
    }
  };
  ctx.bar.appendChild(file);
  button(ctx.bar, "image…", "show an image of your own on the device, resampled to its pixels", () => file.click());
  // How the ray tracer treats the device: the geometric limit, or the
  // brightest diffraction orders of the local unit cell (the rest booked as
  // untraced orders). The f32 preview always draws the geometric limit.
  label(ctx.bar, "rays");
  const raysPick = segmented<"geometric" | "orders">(
    ctx.bar,
    [
      { value: "geometric", text: "geometric", title: "each mirror a tilted reflector, each electrode a phase-gradient deflector: no diffraction in the ray trace" },
      { value: "orders", text: "orders", title: "each hit spawns the 16 brightest diffraction orders of the local unit cell above 0.1% (scalar, Harvey renormalized); the rest is booked as untraced orders. The preview stays geometric" },
    ],
    (v) =>
      answer(scene.setModulatorRayModel(id, JSON.stringify(v === "geometric" ? "Geometric" : { Orders: { max_orders: 16, min_efficiency: 1e-3 } }))),
  );
  let presetSelect: HTMLSelectElement | null = null;
  if (m.kind === "dmd") {
    label(ctx.bar, "preset");
    presetSelect = document.createElement("select");
    presetSelect.className = "cv-select";
    presetSelect.dataset.name = "modulatorPreset";
    presetSelect.title = "pitch, tilt and hinge axis from TI DLPA083B Table 2-1; the mirror's linear fill is the square root of TI's on-state fill factor (docs/dmd.md)";
    presetSelect.append(new Option("—", ""), ...presets.map((p, i) => new Option(`${p.name} · ${p.pitchUm} µm · ${p.tiltDeg}°`, String(i))));
    presetSelect.onchange = () => {
      const p = presets[Number(presetSelect!.value)];
      if (!p) return;
      setDevice((d) => {
        d.pitch_um = p.pitchUm;
        d.kind.Dmd = { ...d.kind.Dmd!, tilt_deg: p.tiltDeg, tilt_axis: p.axis, mirror_fill: Math.sqrt(p.onStateFill) };
      });
    };
    ctx.bar.appendChild(presetSelect);
  }

  // The fields: the array, the optics, the pattern's numbers, the wavelength.
  const fNx = numberField(ctx.fields, "columns", "px", {
    name: "modulatorColumns", step: 16, digits: 0, title: "pixels along the device's local X (the count is yours to set; no device resolution is assumed)",
    commit: (v) => setDevice((d) => { d.pixels[0] = Math.max(1, Math.round(v)); }),
  });
  const fNy = numberField(ctx.fields, "rows", "px", {
    name: "modulatorRows", step: 16, digits: 0, title: "pixels along local Y",
    commit: (v) => setDevice((d) => { d.pixels[1] = Math.max(1, Math.round(v)); }),
  });
  const fPitch = numberField(ctx.fields, "pitch", "µm", {
    name: "modulatorPitch", step: 0.1, digits: 2, title: "the square pixels' pitch",
    commit: (v) => setDevice((d) => { d.pitch_um = v; }),
  });
  let fTilt: NumberField, fFill: NumberField, fRefl: NumberField, fExtra: NumberField;
  if (m.kind === "dmd") {
    fTilt = numberField(ctx.fields, "tilt", "°", {
      name: "modulatorTilt", step: 0.5, digits: 2, title: "the mirrors' tilt in the ON and OFF states (below 45°)",
      commit: (v) => setDevice((d) => { d.kind.Dmd!.tilt_deg = v; }),
    });
    fFill = numberField(ctx.fields, "fill", "", {
      name: "modulatorFill", step: 0.01, digits: 3, title: "a mirror's side over the pitch (linear fill); the rest is dead space, which absorbs",
      commit: (v) => setDevice((d) => { d.kind.Dmd!.mirror_fill = v; }),
    });
    fRefl = numberField(ctx.fields, "reflectance", "", {
      name: "modulatorReflectance", step: 0.01, digits: 3, title: "a mirror's power reflectance",
      commit: (v) => setDevice((d) => { d.kind.Dmd!.reflectance = v; }),
    });
    fExtra = numberField(ctx.fields, "window", "", {
      name: "modulatorWindow", step: 0.01, digits: 3, title: "the window's power transmittance per pass (light crosses it twice)",
      commit: (v) => setDevice((d) => { d.kind.Dmd!.window_transmittance = v; }),
    });
  } else {
    fTilt = numberField(ctx.fields, "director", "°", {
      name: "modulatorDirector", step: 5, digits: 1, title: "the liquid crystal's director in the device plane: only the light polarized along it is modulated",
      commit: (v) => setDevice((d) => { d.kind.Lcos!.director_deg = v; }),
    });
    fFill = numberField(ctx.fields, "fill", "", {
      name: "modulatorFill", step: 0.01, digits: 3, title: "an electrode's side over the pitch; the dead space reflects unmodulated",
      commit: (v) => setDevice((d) => { d.kind.Lcos!.pixel_fill = v; }),
    });
    fRefl = numberField(ctx.fields, "reflectance", "", {
      name: "modulatorReflectance", step: 0.01, digits: 3, title: "an electrode's power reflectance",
      commit: (v) => setDevice((d) => { d.kind.Lcos!.reflectance = v; }),
    });
    fExtra = numberField(ctx.fields, "design λ", "nm", {
      name: "modulatorDesign", step: 1, digits: 1, title: "the wavelength the phase table is calibrated at; elsewhere the phase scales as λ₀/λ (flagged)",
      commit: (v) => setDevice((d) => { d.kind.Lcos!.design_lambda_um = v / 1000; }),
    });
  }
  const fPeriod = numberField(ctx.fields, "period", "px", {
    name: "modulatorPeriod", step: 1, digits: 2, title: "the grating's period in pixels",
    commit: (v) => { period = v; setPattern(patternKind(m)); },
  });
  const fAngle = numberField(ctx.fields, "angle", "°", {
    name: "modulatorAngle", step: 5, digits: 1, title: "the direction the grating's phase or stripes run along, from local X",
    commit: (v) => { angle = v; setPattern(patternKind(m)); },
  });
  const fShape = numberField(ctx.fields, m.kind === "dmd" ? "duty" : "focal / ℓ", "", {
    name: "modulatorShape", step: m.kind === "dmd" ? 0.05 : 1, digits: 3,
    title: m.kind === "dmd" ? "the share of each period that is ON" : "the lens's focal length (mm), or the vortex's charge",
    commit: (v) => {
      const k = patternKind(m);
      if (k === "binary") duty = v;
      else if (k === "lens") focal = v;
      else if (k === "vortex") charge = v;
      setPattern(k);
    },
  });
  const fLambda = numberField(ctx.fields, "λ", "nm", {
    name: "modulatorLambda", step: 1, digits: 1, title: "the wavelength of the orders drawn here",
    commit: (v) => { lambdaNm = Math.max(200, Math.min(2000, v)); refresh(); },
  });

  function readOrders() {
    orders = JSON.parse(scene.modulatorOrdersJSON(id, lambdaNm / 1000));
  }

  function readPicture() {
    const [w, h] = scene.modulatorPreviewSize(id, 160);
    if (!w || !h) {
      picture = null;
      return;
    }
    const rgba = scene.modulatorPreviewRGBA(id, 160);
    const cv = picture ?? document.createElement("canvas");
    cv.width = w;
    cv.height = h;
    cv.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
    picture = cv;
  }

  function refresh() {
    read();
    const k = patternKind(m);
    kindPick.set(k);
    raysPick.set(m.rayModel === "Geometric" ? "geometric" : "orders");
    fNx.set(m.device.pixels[0]);
    fNy.set(m.device.pixels[1]);
    fPitch.set(m.device.pitch_um);
    if (m.kind === "dmd") {
      const d = m.device.kind.Dmd!;
      fTilt.set(d.tilt_deg);
      fFill.set(d.mirror_fill);
      fRefl.set(d.reflectance);
      fExtra.set(d.window_transmittance);
    } else {
      const l = m.device.kind.Lcos!;
      fTilt.set(l.director_deg);
      fFill.set(l.pixel_fill);
      fRefl.set(l.reflectance);
      fExtra.set(l.design_lambda_um * 1000);
    }
    const grating = k === "binary" || k === "blazed";
    fPeriod.set(grating ? period : null);
    fAngle.set(grating ? angle : null);
    fShape.set(k === "binary" ? duty : k === "lens" ? focal : k === "vortex" ? charge : null);
    fLambda.set(lambdaNm);
    readOrders();
    readPicture();
    writeReadout();
    ctx.redraw();
  }

  function writeReadout() {
    const lines: string[] = [];
    lines.push(`${m.device.pixels[0]} × ${m.device.pixels[1]} pixels of ${fmt(m.device.pitch_um, 2)} µm: ${fmt(m.widthMm, 2)} × ${fmt(m.heightMm, 2)} mm`);
    if (patternKind(m) === "imported") {
      const n = (m.device.pattern.Explicit as { count: number }).count;
      lines.push(`an imported bitmap of ${n} values`);
    }
    if (orders?.error) {
      lines.push(`<span class="warn">no orders: ${orders.error}</span>`);
    } else if (orders) {
      const best = orders.orders.reduce<Order | null>((a, o) => (!a || o.efficiency > a.efficiency ? o : a), null);
      const total = orders.propagating > 0 ? orders.propagating : 1;
      if (best) {
        lines.push(
          `at ${fmt(lambdaNm, 1)} nm: ${orders.orders.length} orders propagate; the brightest, (${best.m}, ${best.n}), carries <b>${fmt((100 * best.efficiency) / total, 1)}%</b> of the reflected power`,
        );
      }
      if (orders.note) lines.push(`<span class="warn">${orders.note}</span>`);
      if (orders.pitchOverLambda < 14) {
        lines.push(
          `<span class="warn">the pitch is ${fmt(orders.pitchOverLambda, 1)}λ, below 14λ: scalar theory may be off by more than ±5% (Pommet et al. 1994)</span>`,
        );
      }
    }
    if (m.rayModel === "Geometric") {
      lines.push(
        m.kind === "dmd"
          ? "in the ray trace each mirror is a tilted reflector (the geometric limit); the wave path carries these orders"
          : "in the ray trace the director's half is steered by the phase gradient, the rest reflects (the geometric limit); the wave path carries these orders",
      );
    } else {
      const o = m.rayModel.Orders;
      lines.push(`in the ray trace each hit spawns the ${o.max_orders} brightest of these orders above ${fmt(100 * o.min_efficiency, 2)}%; the rest is booked as untraced orders (the preview draws the geometric limit)`);
    }
    ctx.read.innerHTML = lines.map((l) => `<div>${l}</div>`).join("");
  }

  function draw(c: CanvasRenderingContext2D, w: number, h: number) {
    c.fillStyle = "#050709";
    c.fillRect(0, 0, w, h);
    const pad = 12;
    const half = w / 2;
    // The pattern, nearest-neighbour, to the device's aspect.
    if (picture) {
      const s = Math.min((half - 2 * pad) / picture.width, (h - 2 * pad - 14) / picture.height);
      const pw = picture.width * s, ph = picture.height * s;
      const x0 = (half - pw) / 2, y0 = (h - ph) / 2 + 6;
      c.imageSmoothingEnabled = false;
      c.drawImage(picture, x0, y0, pw, ph);
      c.strokeStyle = "rgba(159,179,200,0.6)";
      c.lineWidth = 1;
      c.strokeRect(x0 - 0.5, y0 - 0.5, pw + 1, ph + 1);
    }
    c.fillStyle = "rgba(205,214,228,0.7)";
    c.font = "11px ui-monospace, monospace";
    c.fillText(m.kind === "dmd" ? "mirrors (ON light)" : "phase (0 → 2π)", pad, 16);
    c.fillText("orders, direction cosines", half + pad, 16);
    // The orders, in the unit circle.
    const r = Math.min(half - 2 * pad, h - 2 * pad - 14) / 2;
    const cx = half + half / 2, cy = h / 2 + 6;
    c.strokeStyle = "rgba(127,214,255,0.35)";
    c.beginPath();
    c.arc(cx, cy, r, 0, Math.PI * 2);
    c.stroke();
    c.setLineDash([2, 4]);
    c.beginPath();
    c.moveTo(cx - r, cy);
    c.lineTo(cx + r, cy);
    c.moveTo(cx, cy - r);
    c.lineTo(cx, cy + r);
    c.stroke();
    c.setLineDash([]);
    if (orders && !orders.error) {
      const total = orders.propagating > 0 ? orders.propagating : 1;
      for (const o of orders.orders) {
        const t = Math.max(0, 1 + Math.log10(o.efficiency / total) / 6); // 1e-6 → 0, 1 → 1
        const size = 1 + 4 * t;
        const g = Math.round(80 + 175 * t);
        c.fillStyle = `rgb(${g},${Math.round(g * 0.85)},255)`;
        c.beginPath();
        c.arc(cx + o.alpha * r, cy - o.beta * r, size, 0, Math.PI * 2);
        c.fill();
      }
      // The incident direction, mirrored to where its specular reflection
      // would leave a flat array: the order (0, 0).
      const [ia, ib] = orders.incidence;
      c.strokeStyle = "#ffd678";
      c.lineWidth = 1.5;
      const x = cx + ia * r, y = cy - ib * r;
      c.beginPath();
      c.moveTo(x - 5, y - 5);
      c.lineTo(x + 5, y + 5);
      c.moveTo(x + 5, y - 5);
      c.lineTo(x - 5, y + 5);
      c.stroke();
    }
  }

  refresh();
  return {
    hint: m.kind === "dmd"
      ? "a DMD: the pattern, and the orders it sends (log scale; × the flat-array specular direction)"
      : "an LCoS SLM: the phase, and the orders of its modulated beam (log scale; × the specular direction)",
    refresh,
    draw,
    handles: () => [],
    drag: () => {},
  };
}
