// The fidelity panel: how much coating physics the f64 trace pays for.
//
// Coating-fidelity spec §5. One preset selector (Fast, Standard, Reference,
// or custom), one control per switch with what it costs, the stamp every
// result carries, what the f32 preview makes of the profile, and the final
// check: one trace under Reference, beside the profile being edited, without
// changing it. The profile lives in the scene, so it is saved with it.
//
// The costs are read from `fidelity-costs.json`, which the throughput bench
// writes from measured runs. A switch value with no measured row says "not
// measured"; nothing here invents a number.

type Profile = {
  preset?: string;
  coating_model: string;
  film_materials: string;
  layer_absorption: string;
  plate_model: string;
  temperature: string;
  numerics: string;
  ghosts: number | null;
};

type Stamp = {
  profile: Profile;
  preset: string | null;
  text: string;
  approximations: string[];
  solverVersion: string;
};

type FidelityState = {
  profile: Profile;
  preset: string | null;
  stamp: Stamp;
  preview: { message: string | null; refused: string[] };
  presets: Record<string, Profile>;
};

type CostRow = { switch: string; value: string; relative: number; against: string };

type CheckReply = {
  current: Stamp;
  reference: Stamp;
  detectors: { id: number; flux: [number, number]; visible: [number, number]; dop: [number, number] }[];
  objects: { id: number; absorbed: [number, number] }[];
  onlyInCurrent: string[];
  onlyInReference: string[];
};

export type FidelityDeps = {
  scene: {
    fidelityJSON(): string;
    setFidelity(json: string): void;
    setFidelityPreset(name: string): void;
    fidelityCostsJSON(): string;
    finalCheckJSON(): string;
    ghostsEnabled(): boolean;
    maxGhostBounces(): number;
    sceneTemperatureC(): number;
    setSceneTemperature(c: number): boolean;
  };
  /** After the profile changed: re-trace, redraw, and refresh the reference. */
  edited: () => void;
  /** The profile may have moved the ghost toggle (Fast turns ghosts off). */
  ghostsChanged: (on: boolean, bounces: number) => void;
};

export type FidelityPanel = {
  /** Re-read the profile (after a load, an undo, a ghost toggle). */
  sync: () => void;
};

/** The switches, their values in order, and what each value means. */
const SWITCHES: { key: keyof Profile; label: string; values: [string, string][] }[] = [
  {
    key: "coating_model",
    label: "coatings",
    values: [
      ["Nominal", "design-point R and T at every wavelength and angle (nominal, not verified)"],
      ["Unpolarized", "full transfer matrix, s and p averaged"],
      ["Polarized", "full transfer matrix with phase, Stokes transport"],
    ],
  },
  {
    key: "film_materials",
    label: "film materials",
    values: [
      ["Nominal", "each film at its nominal constant"],
      ["Dispersive", "each film's n(λ), k(λ)"],
    ],
  },
  {
    key: "layer_absorption",
    label: "absorption",
    values: [
      ["Off", "absorbed power leaves the ray; no deposit is recorded"],
      ["PerFace", "one deposit per coated face"],
      ["PerLayer", "one deposit per layer, from the Poynting flux"],
    ],
  },
  {
    key: "plate_model",
    label: "coated plates",
    values: [
      ["PhaseFree", "phase-free Mueller matrix (approximate for polarized light)"],
      ["Incoherent", "incoherent Mueller series of both faces"],
    ],
  },
  {
    key: "temperature",
    label: "temperature",
    values: [
      ["Off", "films at their reference temperature"],
      ["Declared", "first-order thermo-optic shift at the declared temperature (declared, not validated)"],
    ],
  },
  {
    key: "numerics",
    label: "numerics",
    values: [
      ["Permissive", "out-of-range tables clamp, and the result is flagged"],
      ["Strict", "out-of-range and degenerate inputs are refused with a reason"],
    ],
  },
];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function fmt(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) return v.toExponential(3);
  return v.toPrecision(4);
}

function rel(a: number, b: number): string {
  if (a === b) return "same";
  const base = Math.max(Math.abs(a), Math.abs(b));
  if (base === 0) return "same";
  return `${(((b - a) / base) * 100).toFixed(2)}%`;
}

export function initFidelityPanel(deps: FidelityDeps): FidelityPanel {
  const presetSel = $<HTMLSelectElement>("fidPreset");
  const switches = $<HTMLDivElement>("fidSwitches");
  const previewNote = $<HTMLDivElement>("fidPreview");
  const stampLine = $<HTMLDivElement>("dockStamp");
  const checkBtn = $<HTMLButtonElement>("fidCheck");
  const checkOut = $<HTMLDivElement>("fidCheckOut");
  const tempIn = $<HTMLInputElement>("fidTemp");

  let costs: CostRow[] = [];
  try {
    costs = (JSON.parse(deps.scene.fidelityCostsJSON()) as { rows: CostRow[] }).rows ?? [];
  } catch {
    costs = [];
  }
  const costOf = (key: string, value: string) => {
    const row = costs.find((r) => r.switch === key && r.value === value);
    return row ? `×${row.relative.toFixed(2)} time against ${row.against}` : "not measured";
  };

  // One select per switch, built once.
  const selects = new Map<string, HTMLSelectElement>();
  const hints = new Map<string, HTMLSpanElement>();
  for (const sw of SWITCHES) {
    const row = document.createElement("label");
    row.className = "row fid-row";
    const name = document.createElement("span");
    name.className = "fid-label";
    name.textContent = sw.label;
    const sel = document.createElement("select");
    sel.id = `fid-${sw.key}`;
    sel.className = "fid-select";
    for (const [value, help] of sw.values) {
      const o = document.createElement("option");
      o.value = value;
      o.textContent = value;
      o.title = help;
      sel.appendChild(o);
    }
    const hint = document.createElement("span");
    hint.className = "fid-cost";
    row.append(name, sel, hint);
    switches.appendChild(row);
    selects.set(sw.key, sel);
    hints.set(sw.key, hint);
    sel.addEventListener("change", () => {
      const state = read();
      const next: Profile = { ...state.profile, [sw.key]: sel.value };
      delete next.preset;
      deps.scene.setFidelity(JSON.stringify(next));
      after();
    });
  }

  // The scene's temperature: read only under `temperature: Declared`, which
  // Reference sets once there is one (CF-10).
  tempIn.addEventListener("change", () => {
    const text = tempIn.value.trim();
    const value = text === "" ? NaN : Number(text);
    if (!deps.scene.setSceneTemperature(value)) {
      sync();
      return;
    }
    after();
  });

  presetSel.addEventListener("change", () => {
    if (presetSel.value === "custom") return;
    deps.scene.setFidelityPreset(presetSel.value);
    after();
  });

  const read = (): FidelityState => JSON.parse(deps.scene.fidelityJSON()) as FidelityState;

  function after() {
    deps.ghostsChanged(deps.scene.ghostsEnabled(), deps.scene.maxGhostBounces());
    sync();
    checkOut.textContent = "";
    deps.edited();
  }

  function sync() {
    let state: FidelityState;
    try {
      state = read();
    } catch {
      return;
    }
    presetSel.value = state.preset ?? "custom";
    for (const sw of SWITCHES) {
      const sel = selects.get(sw.key)!;
      const value = String(state.profile[sw.key]);
      sel.value = value;
      const help = sw.values.find(([v]) => v === value)?.[1] ?? "";
      sel.title = help;
      hints.get(sw.key)!.textContent = costOf(sw.key, value);
    }
    const t = deps.scene.sceneTemperatureC();
    if (document.activeElement !== tempIn) tempIn.value = Number.isFinite(t) ? String(t) : "";
    previewNote.textContent = state.preview.message ?? "Preview: runs this profile as asked.";
    stampLine.textContent = state.stamp.text;
    stampLine.title = [state.stamp.solverVersion, ...state.stamp.approximations].join("\n");
  }

  checkBtn.addEventListener("click", () => {
    checkOut.textContent = "tracing under Reference…";
    checkBtn.disabled = true;
    // Let the message paint before the trace takes the thread.
    setTimeout(() => {
      try {
        const reply = JSON.parse(deps.scene.finalCheckJSON()) as CheckReply | null;
        if (!reply) {
          checkOut.textContent = "no scene";
          return;
        }
        renderCheck(reply);
      } catch (e) {
        checkOut.textContent = `final check failed: ${e}`;
      } finally {
        checkBtn.disabled = false;
      }
    }, 20);
  });

  function renderCheck(r: CheckReply) {
    checkOut.textContent = "";
    const head = document.createElement("div");
    head.className = "fid-check-head";
    head.textContent = `${r.current.preset ?? "custom"} against Reference`;
    checkOut.appendChild(head);
    const table = document.createElement("table");
    table.className = "fid-check";
    table.id = "fidCheckTable";
    const tr = (cells: string[], th = false) => {
      const row = document.createElement("tr");
      for (const c of cells) {
        const cell = document.createElement(th ? "th" : "td");
        cell.textContent = c;
        row.appendChild(cell);
      }
      table.appendChild(row);
    };
    tr(["", "current", "Reference", "Δ"], true);
    for (const d of r.detectors) {
      tr([`detector ${d.id} flux`, fmt(d.flux[0]), fmt(d.flux[1]), rel(d.flux[0], d.flux[1])]);
      tr([`  visible`, fmt(d.visible[0]), fmt(d.visible[1]), rel(d.visible[0], d.visible[1])]);
      tr([`  DOP`, fmt(d.dop[0]), fmt(d.dop[1]), (d.dop[1] - d.dop[0]).toFixed(4)]);
    }
    for (const o of r.objects) {
      if (o.absorbed[0] === 0 && o.absorbed[1] === 0) continue;
      tr([`object ${o.id} absorbed`, fmt(o.absorbed[0]), fmt(o.absorbed[1]), rel(o.absorbed[0], o.absorbed[1])]);
    }
    checkOut.appendChild(table);
    const notes = (title: string, list: string[]) => {
      if (!list.length) return;
      const p = document.createElement("div");
      p.className = "fid-check-notes";
      p.textContent = `${title}: ${list.join("; ")}`;
      checkOut.appendChild(p);
    };
    notes("only under the current profile", r.onlyInCurrent);
    notes("only under Reference", r.onlyInReference);
  }

  sync();
  return { sync };
}
