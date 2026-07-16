// The openUC2 mode (§15).
//
// Reached at `#openuc2`. The freeform Canvas is untouched and still the default.
//
// What lives here is UI state only: a search box, which filters and overlays are
// on, the viewport, and which pointer gesture is in flight. The project —
// catalogue, instances, placements, the selection, the undo stack, the
// materialized scene — lives in Rust behind the `OpenUc2` handle, and this file
// asks for a view whenever it needs to draw one (§0.3: no assembly state
// mirrored into JS). The card list is re-derived from `catalogView()` rather than
// kept as a second copy of the catalogue.
//
// Rays go through the same `RayLayer` the freeform route uses (§15.1: do not fork
// the renderer), and any displayed *number* comes from `detectorJson`, which is
// the f64 reference path (§4.1).
//
// # Two loops, on purpose
//
// `frame()` draws. It is the interactive f32 path and runs on every pointer move
// of a drag. `refresh()` re-reads the lists and the reference numbers, and runs
// only when an operation settles. The split is §15.6: rays follow the drag, but
// a number is never shown from the f32 tracer, and the f64 trace is far too
// expensive to run per frame anyway.

import { OpenUc2 } from "../../../pkg/oc_wasm.js";
import { RayLayer, type V2 } from "../../rays";

/// One catalogue module, as the library list needs it. Mirrors `ModuleView`.
type ModuleCard = {
  id: string;
  uuid: string;
  version: string;
  name: string;
  category: string;
  group: string | null;
  description: string | null;
  tags: string[];
  color: string | null;
  capabilities: { layout: string; optical: string; cad: string; control: string };
  vendor: string;
  sku: string;
  price_eur: number;
  icon: string | null;
  footprint_cells: [number, number];
  cell_count: number;
  traceable: boolean;
  fragment_asset_missing: string | null;
};

type Coverage = {
  total: number;
  layout_ready: number;
  optical: Record<string, number>;
  cad: Record<string, number>;
  control: Record<string, number>;
  categories: Record<string, number>;
  traceable: number;
  declared_fragment_asset_missing: number;
  no_fragment_declared: number;
  missing_icon: number;
};

type Instance = {
  instance_id: string;
  name: string;
  module_id: string;
  color: string | null;
  anchor_grid: [number, number, number];
  rotation_quadrants: number;
  resolved: boolean;
  materialized: boolean;
};

type Warning = { kind: string; text: string; instance?: string };

/// One aggregated product. Mirrors `BomLine`.
type BomLine = {
  quantity: number;
  module_id: string;
  name: string;
  version: string;
  vendor: string;
  sku: string;
  unit_price_eur: number;
  subtotal_eur: number;
  resolved: boolean;
  readiness: string[];
};

type Bom = { lines: BomLine[]; total_eur: number; custom_object_count: number };

/// What a generated module's optics were built from. Mirrors `parametersJson`.
type Params = {
  generator: string;
  /// The parameters that actually reach the geometry. Only these are editable
  /// (§15.7); a field that changes nothing is worse than no field.
  editable: string[];
  values: Record<string, number | string>;
  overridden: string[];
  paraxial: Paraxial | null;
};

/// The lensmaker's equation for the current numbers, computed without the tracer.
type Paraxial = {
  efl_mm: number;
  bfd_mm: number;
  rear_focus_x_mm: number;
  nd: number;
  lambda_um: number;
};

type Footprint = {
  instance_id: string;
  color: string;
  layer: number;
  active: boolean;
  cells_mm: [number, number][];
  cell_size_mm: [number, number];
};

/// A placed port and what it found across from it. Mirrors `PlacedPort`.
type Port = {
  instance: string;
  module_id: string;
  port_id: string;
  kind: string;
  role: string;
  position_mm: [number, number];
  height_mm: number;
  direction: [number, number];
  layer: number;
  status:
    | { state: "connected"; instance: string; port: string }
    | { state: "misaligned"; instance: string; port: string; why: string }
    | { state: "free" };
};

/// The active grid profile. Mirrors `GridProfile`.
///
/// Read from the project rather than assumed: the pitch is catalogue data (§6.4),
/// and a UI that drew its own 50 would be wrong, silently, on any other profile.
type Grid = {
  id: string;
  pitch_mm: [number, number, number];
  cell_size_mm: [number, number, number];
  mounting_plane_offset_mm: number;
  allowed_rotations_deg: number[];
};

/// One piece of optical geometry, in world millimetres.
type Optic = {
  kind: "element" | "detector" | "source";
  id: number;
  instance: string;
  material?: string;
  points: number[];
  direction?: [number, number];
};

/// Where the static catalogue is served from. A base path, never a repository:
/// the browser must not be able to point itself at arbitrary live data (§15.3).
const CATALOG_BASE =
  (import.meta as { env?: Record<string, string> }).env?.VITE_OPENUC2_CATALOG ??
  "catalog/openuc2";

const EXPOSURE = 1.0;

/// A commit of all zeros is what an unpinned or fixture catalogue carries. A
/// real generated one is pinned to the module repository's sha (§5.4), so this
/// is the signal that what is on screen is placeholder data.
const UNPINNED = /^0+$/;

/// The overlays, each independently toggleable (§15.5).
///
/// `labels` stands in for the module icon: the development fixture ships no icon
/// assets, and the real artwork arrives with the module repository's compiler
/// (M1). It draws the instance id, which is what the fidelity warnings and the
/// port list name, so the panels and the canvas can be read against each other.
const OVERLAYS = [
  ["footprints", "footprints", true],
  ["labels", "labels", true],
  ["optics", "optical geometry", true],
  ["ports", "ports", false],
  ["rays", "rays", true],
  ["onion", "other layers", true],
] as const;

type OverlayKey = (typeof OVERLAYS)[number][0];

export async function initOpenUc2(): Promise<void> {
  const view = document.getElementById("view") as HTMLCanvasElement;
  const ctx = view.getContext("2d")!;
  const root = document.getElementById("uc2")!;
  root.style.display = "block";
  // Every panel that is not ours belongs to the freeform route. Selected by
  // structure rather than by an explicit id list, which would silently miss the
  // next panel someone adds — as it already did for the wavefront scrubber.
  for (const el of document.querySelectorAll<HTMLElement>(".panel")) {
    if (!el.closest("#uc2")) el.style.display = "none";
  }

  const uc2 = new OpenUc2();

  // --- catalogue ---------------------------------------------------------

  const status = document.getElementById("uc2Status")!;
  let cards: ModuleCard[] = [];
  let coverage: Coverage | null = null;

  try {
    const doc = await fetch(`${CATALOG_BASE}/catalog.json`).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.text();
    });
    uc2.loadCatalog(doc);

    // Fragments are separate assets, fetched once here rather than per frame.
    const declared = new Set(
      (JSON.parse(uc2.catalogView()) as ModuleCard[])
        .map((m) => m.fragment_asset_missing)
        .filter((p): p is string => !!p),
    );
    // `catalogView` reports what is *missing*, which before any attach is
    // exactly the set of declared fragments.
    await Promise.all(
      [...declared].map(async (path) => {
        try {
          const body = await fetch(`${CATALOG_BASE}/${path}`).then((r) => r.text());
          uc2.attachFragment(path, body);
        } catch (e) {
          // Leave it missing: the card will say so, which is the honest outcome.
          console.warn(`openuc2: fragment ${path} did not load:`, e);
        }
      }),
    );

    cards = JSON.parse(uc2.catalogView());
    coverage = JSON.parse(uc2.coverageJson());
    const profiles = JSON.parse(uc2.gridProfilesJson()) as { id: string }[];
    uc2.newProject(profiles[0]?.id ?? "openuc2-core-50");
    status.textContent = `${cards.length} modules`;
  } catch (e) {
    status.textContent = `catalogue did not load: ${e}`;
    status.className = "uc2-bad";
    return;
  }

  renderProvenance(uc2);
  renderCoverage(coverage);

  // The profile the project actually opened on. Fixed for the life of a setup
  // (§12.3), so it is read here rather than per frame — but re-read whenever a
  // different setup is opened, because that setup names its own profile and a
  // stale pitch would draw the grid and the drop preview in the wrong places.
  let grid = JSON.parse(uc2.gridJson()) as Grid;

  // --- library -----------------------------------------------------------

  const search = document.getElementById("uc2Search") as HTMLInputElement;
  const catSel = document.getElementById("uc2Category") as HTMLSelectElement;
  const capSel = document.getElementById("uc2Capability") as HTMLSelectElement;
  const list = document.getElementById("uc2List")!;

  for (const c of [...new Set(cards.map((m) => m.category))].sort()) {
    catSel.add(new Option(c, c));
  }

  /// Search covers name, id, vendor, SKU, and tags (§15.4).
  function matches(m: ModuleCard): boolean {
    const q = search.value.trim().toLowerCase();
    if (q) {
      const hay = [m.name, m.id, m.vendor, m.sku, ...m.tags, m.category]
        .join(" ")
        .toLowerCase();
      if (!hay.includes(q)) return false;
    }
    if (catSel.value !== "*" && m.category !== catSel.value) return false;
    switch (capSel.value) {
      case "traceable":
        return m.traceable;
      case "validated":
        return m.capabilities.optical === "validated";
      case "layout-only":
        return !m.traceable;
      default:
        return true;
    }
  }

  function renderList() {
    const shown = cards.filter(matches);
    list.replaceChildren(...shown.map(card));
    // Incomplete modules are never hidden by default (§15.4); say what is out of
    // view instead, so the gap stays visible.
    const hidden = cards.length - shown.length;
    status.textContent =
      hidden > 0 ? `${shown.length} of ${cards.length} modules` : `${cards.length} modules`;
  }

  function card(m: ModuleCard): HTMLElement {
    const el = document.createElement("div");
    el.className = "uc2-card";
    if (m.color) el.style.borderLeftColor = m.color;

    const title = document.createElement("div");
    title.className = "uc2-card-t";
    title.textContent = m.name;
    el.append(title);

    const meta = document.createElement("div");
    meta.className = "uc2-card-m";
    meta.textContent = `${m.category} · ${m.footprint_cells[0]}×${m.footprint_cells[1]}${
      m.sku ? ` · ${m.sku}` : ""
    }`;
    el.append(meta);

    const badges = document.createElement("div");
    badges.className = "uc2-badges";
    badges.append(badge(`optical: ${m.capabilities.optical}`, opticalTone(m.capabilities.optical)));
    badges.append(badge(`cad: ${m.capabilities.cad}`, "dim"));
    if (m.capabilities.control !== "absent") badges.append(badge(`control: ${m.capabilities.control}`, "dim"));
    el.append(badges);

    // A module can be perfectly good for layout and still contribute nothing to
    // the trace. Say which, and why, rather than letting it look simulated.
    if (m.fragment_asset_missing) {
      el.append(note(`optics declared but the asset did not load (${m.fragment_asset_missing}) — placing this adds no light`, "bad"));
    } else if (!m.traceable) {
      el.append(note("layout only: no Canvas optics, so it takes up grid space without affecting the rays", "warn"));
    } else if (m.capabilities.optical !== "validated") {
      el.append(note(`optics are ${m.capabilities.optical}, not validated against a reference`, "warn"));
    }

    // Drag onto the grid, or click to add. Click-to-add stays because it is the
    // only route that works from a keyboard or a touch screen without a drag.
    el.draggable = true;
    el.ondragstart = (e) => {
      dragCard = m;
      el.classList.add("uc2-dragging");
      e.dataTransfer?.setData("text/plain", m.id);
      if (e.dataTransfer) e.dataTransfer.effectAllowed = "copy";
    };
    el.ondragend = () => {
      dragCard = null;
      preview = null;
      el.classList.remove("uc2-dragging");
      frame();
    };

    const add = document.createElement("button");
    add.textContent = "add";
    add.title = "place on the first free cell of the active layer";
    add.onclick = () => place(m);
    el.append(add);
    return el;
  }

  function badge(text: string, tone: string): HTMLElement {
    const b = document.createElement("span");
    b.className = `uc2-badge uc2-${tone}`;
    b.textContent = text;
    return b;
  }

  function opticalTone(v: string): string {
    if (v === "validated") return "good";
    if (v === "idealized" || v === "placeholder") return "warn";
    return "dim";
  }

  function note(text: string, tone: string): HTMLElement {
    const n = document.createElement("div");
    n.className = `uc2-note uc2-${tone}`;
    n.textContent = text;
    return n;
  }

  search.oninput = renderList;
  catSel.onchange = renderList;
  capSel.onchange = renderList;
  renderList();

  // --- overlays ----------------------------------------------------------

  const overlayBox = document.getElementById("uc2Overlays")!;
  const shown: Record<string, boolean> = {};
  for (const [key, label, on] of OVERLAYS) {
    shown[key] = on;
    const l = document.createElement("label");
    l.className = "uc2-tog";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = on;
    cb.onchange = () => {
      shown[key] = cb.checked;
      frame();
    };
    l.append(cb, document.createTextNode(label));
    overlayBox.append(l);
  }
  const on = (k: OverlayKey) => shown[k];

  // --- placement ---------------------------------------------------------

  const err = document.getElementById("uc2Err")!;

  function fail(e: unknown) {
    // Structured, not a stack trace: a collision is a normal answer.
    err.textContent = String(e);
  }

  /// Click-to-add lands on the first free cell of the active layer.
  function place(m: ModuleCard) {
    const layer = uc2.activeLayer();
    let cell: [number, number, number] = [0, 0, layer];
    for (let x = 0; x < 64; x++) {
      if (!uc2.instanceAtCell(x, 0, layer)) {
        cell = [x, 0, layer];
        break;
      }
    }
    try {
      uc2.placeModule(m.uuid, m.version, cell[0], cell[1], cell[2], 0);
      err.textContent = "";
    } catch (e) {
      fail(e);
    }
    refresh();
  }

  // The folded-singlet demo (§16.4), the same layout the headless test builds.
  // Here so the mode can be checked in one click rather than by hand-placing four
  // modules onto the right cells.
  const demoBtn = document.getElementById("uc2Demo") as HTMLButtonElement;
  demoBtn.onclick = () => {
    const byRole = (id: string) => cards.find((c) => c.id === id);
    // The detector is turned 270°, not 90°: both lay its sensor across the folded
    // beam and trace identically, but only 270° points its -X window back at the
    // mirror. Turn on the ports overlay to see the difference.
    const plan: [string, [number, number, number], number][] = [
      ["source-collimated-3line-1x1", [0, 0, 0], 0],
      ["lens-pcx-nbk7-1x1", [1, 0, 0], 0],
      ["mirror-flat-45-1x1", [2, 0, 0], 0],
      ["detector-line-1x1", [2, -1, 0], 3],
    ];
    err.textContent = "";
    // One gesture, so rebuilding the demo is a single undo away.
    uc2.beginGesture();
    for (const i of JSON.parse(uc2.instancesView()) as Instance[]) {
      uc2.removeModule(i.instance_id);
    }
    if (uc2.activeLayer() !== 0) uc2.setActiveLayer(0);
    for (const [id, cell, rot] of plan) {
      const m = byRole(id);
      if (!m) {
        err.textContent = `the catalogue has no ${id}`;
        break;
      }
      try {
        uc2.placeModule(m.uuid, m.version, cell[0], cell[1], cell[2], rot);
      } catch (e) {
        fail(e);
      }
    }
    uc2.endGesture();
    refresh();
  };

  // --- module edits ------------------------------------------------------

  function rotate(id: string) {
    const inst = instances().find((i) => i.instance_id === id);
    if (!inst) return;
    try {
      uc2.rotateModule(id, (inst.rotation_quadrants + 1) % 4);
      err.textContent = "";
    } catch (e) {
      fail(e);
    }
    refresh();
  }

  function remove(id: string) {
    try {
      uc2.removeModule(id);
      err.textContent = "";
    } catch (e) {
      fail(e);
    }
    refresh();
  }

  /// Copy a module onto the first free cell near it.
  function duplicate(id: string) {
    const inst = instances().find((i) => i.instance_id === id);
    if (!inst) return;
    const [x, y, z] = inst.anchor_grid;
    for (let d = 1; d < 64; d++) {
      // The *copy* question, not the move question: a duplicate is a new module
      // and the original blocks it, so `blockerFor` would offer a cell the
      // original is standing on and `duplicateModule` would then refuse it.
      if (uc2.blockerForCopy(id, x + d, y, z)) continue;
      try {
        const made = uc2.duplicateModule(id, x + d, y, z);
        uc2.selectInstance(made);
        err.textContent = "";
      } catch (e) {
        fail(e);
      }
      refresh();
      return;
    }
    err.textContent = "no free cell to duplicate into";
  }

  /// Move the selected module by whole cells, for anyone not using a pointer.
  function nudge(dx: number, dy: number) {
    const id = uc2.selectedInstance();
    if (!id) return;
    const inst = instances().find((i) => i.instance_id === id);
    if (!inst) return;
    const [x, y, z] = inst.anchor_grid;
    try {
      uc2.moveModule(id, x + dx, y + dy, z);
      err.textContent = "";
    } catch (e) {
      fail(e);
    }
    refresh();
  }

  function undo() {
    try {
      uc2.undo();
      err.textContent = "";
    } catch (e) {
      fail(e);
    }
    refresh();
  }

  function redo() {
    try {
      uc2.redo();
      err.textContent = "";
    } catch (e) {
      fail(e);
    }
    refresh();
  }

  (document.getElementById("uc2Undo") as HTMLButtonElement).onclick = undo;
  (document.getElementById("uc2Redo") as HTMLButtonElement).onclick = redo;

  // --- files (§15.9) -----------------------------------------------------

  /// Hand the user a file. Nothing leaves the browser: every export is built in
  /// Rust and downloaded from memory, so a setup is never uploaded anywhere.
  function download(name: string, body: string, mime: string) {
    const url = URL.createObjectURL(new Blob([body], { type: mime }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }

  /// A filename from the setup's own name, or the demo's.
  function stem(): string {
    const name = (JSON.parse(uc2.saveSetup())?.metadata?.name as string) || "openuc2-setup";
    return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") ||
      "openuc2-setup";
  }

  (document.getElementById("uc2Save") as HTMLButtonElement).onclick = () => {
    // The setup, not the scene: this is the file that remembers which products
    // are assembled and where (§5.3). The .ocanvas export is the other button.
    download(`${stem()}.uc2canvas`, uc2.saveSetup(), "application/json");
  };

  const filePicker = document.getElementById("uc2File") as HTMLInputElement;
  (document.getElementById("uc2Open") as HTMLButtonElement).onclick = () => filePicker.click();
  filePicker.onchange = async () => {
    const f = filePicker.files?.[0];
    if (!f) return;
    try {
      uc2.loadSetup(await f.text());
      // The opened setup names its own grid profile, which need not be the one
      // the empty project started on.
      grid = JSON.parse(uc2.gridJson()) as Grid;
      err.textContent = "";
    } catch (e) {
      // A setup for a catalogue this build does not have is a normal answer, not
      // a crash: it names modules by pinned uuid and version (§12.3).
      fail(e);
    }
    // The picker keeps its value, so re-opening the same file would do nothing.
    filePicker.value = "";
    refresh();
  };

  (document.getElementById("uc2ExportOcanvas") as HTMLButtonElement).onclick = () => {
    try {
      const out = JSON.parse(uc2.exportFlatOcanvasJson()) as {
        ocanvas: string;
        warnings: Warning[];
      };
      download(`${stem()}.ocanvas`, out.ocanvas, "application/json");
      // §15.9: show the report *with* the export. The file cannot carry these —
      // .ocanvas v1 has nowhere to put them — so this is the only place they are
      // said, and a flattened scene's idealized parts must not pass for measured.
      err.textContent = "";
      exportReport(out.warnings);
    } catch (e) {
      fail(e);
    }
  };

  /// What the exported file does not tell you.
  function exportReport(warnings: Warning[]) {
    const box = document.getElementById("uc2Warn")!;
    const rows: HTMLElement[] = [
      note(
        warnings.length === 0
          ? "exported: every placed module's optics are validated"
          : `exported, with ${warnings.length} fidelity caveat${warnings.length > 1 ? "s" : ""} — the .ocanvas cannot carry these:`,
        warnings.length === 0 ? "good" : "warn",
      ),
      ...warnings.map((w) => note(w.text, "warn")),
    ];
    box.replaceChildren(...rows);
  }

  (document.getElementById("uc2BomCsv") as HTMLButtonElement).onclick = () =>
    download(`${stem()}-bom.csv`, uc2.bomCsv(), "text/csv");
  (document.getElementById("uc2BomJson") as HTMLButtonElement).onclick = () =>
    download(`${stem()}-bom.json`, uc2.bomJson(), "application/json");

  // --- layers ------------------------------------------------------------

  const layerNum = document.getElementById("uc2LayerNum")!;
  function setLayer(z: number) {
    try {
      uc2.setActiveLayer(z);
      err.textContent = "";
    } catch (e) {
      fail(e);
    }
    refresh();
  }
  (document.getElementById("uc2LayerUp") as HTMLButtonElement).onclick = () =>
    setLayer(uc2.activeLayer() + 1);
  (document.getElementById("uc2LayerDown") as HTMLButtonElement).onclick = () =>
    setLayer(uc2.activeLayer() - 1);

  // --- keyboard (§15.6) ---------------------------------------------------

  window.addEventListener("keydown", (e) => {
    if (location.hash !== "#openuc2") return;
    const t = e.target as HTMLElement | null;
    // Never steal a key from the search box.
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.isContentEditable)) return;

    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === "z") {
      e.preventDefault();
      return e.shiftKey ? redo() : undo();
    }
    if (mod && e.key.toLowerCase() === "y") {
      e.preventDefault();
      return redo();
    }

    const sel = uc2.selectedInstance();
    if (mod && e.key.toLowerCase() === "d") {
      e.preventDefault();
      if (sel) duplicate(sel);
      return;
    }
    if (mod) return;
    if (e.key === "Escape") {
      uc2.clearSelection();
      return refresh();
    }
    if (!sel) return;
    if (e.key.toLowerCase() === "r") {
      e.preventDefault();
      return rotate(sel);
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      return remove(sel);
    }
    // Arrows move by a cell. Dragging is the usual way, but it is a pointer-only
    // way, and a module added with the card's button would otherwise be stuck
    // where it landed.
    const step: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, 1],
      ArrowDown: [0, -1],
    };
    const d = step[e.key];
    if (d) {
      e.preventDefault();
      nudge(d[0], d[1]);
    }
  });

  // --- lists -------------------------------------------------------------

  const layout = document.getElementById("uc2Layout")!;
  const warnBox = document.getElementById("uc2Warn")!;
  const portBox = document.getElementById("uc2Ports")!;
  const nums = document.getElementById("uc2Nums")!;

  const instances = () => JSON.parse(uc2.instancesView()) as Instance[];

  function renderLayout(rows: Instance[]) {
    const sel = uc2.selectedInstance();
    layout.replaceChildren(
      ...rows.map((i) => {
        const row = document.createElement("div");
        row.className = i.instance_id === sel ? "uc2-row uc2-sel" : "uc2-row";
        const label = document.createElement("span");
        const off = i.anchor_grid[2] !== uc2.activeLayer();
        label.textContent = `${i.name}${off ? ` · layer ${i.anchor_grid[2]}` : ""}`;
        if (!i.resolved) label.className = "uc2-bad";
        else if (!i.materialized) label.className = "uc2-warn";
        row.append(label);
        // By name, not by cell: a footprint is not required to include its own
        // anchor cell, and addressing it that way would deselect such a module
        // instead of selecting it.
        row.onclick = () => {
          uc2.selectInstance(i.instance_id);
          refresh();
        };

        for (const [text, title, fn] of [
          ["⟳", "rotate 90° (R)", () => rotate(i.instance_id)],
          ["⧉", "duplicate (Ctrl+D)", () => duplicate(i.instance_id)],
          ["×", "remove (Del)", () => remove(i.instance_id)],
        ] as const) {
          const b = document.createElement("button");
          b.textContent = text;
          b.title = title;
          b.onclick = (e) => {
            e.stopPropagation();
            fn();
          };
          row.append(b);
        }
        return row;
      }),
    );
    if (rows.length === 0) layout.textContent = "nothing placed yet";
  }

  function renderPorts(ports: Port[]) {
    if (ports.length === 0) {
      portBox.textContent = "no module here declares a port";
      return;
    }
    const bad = ports.filter((p) => p.status.state === "misaligned");
    const free = ports.filter((p) => p.status.state === "free");
    const rows: HTMLElement[] = [];
    rows.push(
      note(
        `${ports.length - bad.length - free.length} of ${ports.length} connected`,
        bad.length ? "warn" : "dim",
      ),
    );
    // Misalignments first: a port that is nearly connected is a mistake, whereas
    // an open end is just the end of the chain.
    for (const p of bad) {
      const s = p.status as { instance: string; port: string; why: string };
      rows.push(note(`${p.instance} ${p.port_id} → ${s.instance} ${s.port}: ${s.why}`, "warn"));
    }
    for (const p of free) {
      rows.push(note(`${p.instance} ${p.port_id} (${p.role}): nothing across from it`, "dim"));
    }
    portBox.replaceChildren(...rows);
  }

  /// The module inspector (§15.7).
  ///
  /// Only a module with a typed generator gets editable fields, because only its
  /// parameters reach the geometry. For anything else the numbers are shown
  /// read-only and say so — a lens that reads 50 mm while tracing 100 would be
  /// worse than no field at all.
  function renderInspector() {
    const box = document.getElementById("uc2Inspector")!;
    const id = uc2.selectedInstance();
    if (!id) {
      box.replaceChildren(note("nothing selected", "dim"));
      return;
    }
    const inst = instances().find((i) => i.instance_id === id);
    if (!inst) {
      box.replaceChildren(note("nothing selected", "dim"));
      return;
    }

    const rows: HTMLElement[] = [];
    const head = document.createElement("div");
    head.textContent = `${inst.name}`;
    head.title = `${inst.module_id} · ${id}`;
    rows.push(head);
    rows.push(
      note(
        `cell [${inst.anchor_grid.join(", ")}] · rotated ${inst.rotation_quadrants * 90}°`,
        "dim",
      ),
    );

    const params = JSON.parse(uc2.parametersJson(id)) as Params | null;
    if (!params) {
      rows.push(
        note("its optics are a stored fragment, so there is nothing to configure", "dim"),
      );
      box.replaceChildren(...rows);
      return;
    }

    for (const key of params.editable) {
      rows.push(paramRow(id, key, params));
    }
    if (params.paraxial) rows.push(predict(params.paraxial));
    box.replaceChildren(...rows);
  }

  /// One editable parameter.
  function paramRow(id: string, key: string, params: Params): HTMLElement {
    const row = document.createElement("div");
    row.className = params.overridden.includes(key) ? "uc2-param uc2-over" : "uc2-param";
    const label = document.createElement("label");
    label.textContent = key.replace(/_mm$/, "").replace(/_/g, " ");
    row.append(label);

    const raw = params.values[key];
    if (typeof raw === "string") {
      // `material`, or a plane radius — JSON has no infinity, so `"inf"` is how a
      // flat surface travels. Not editable here: a glass picker and a
      // finite/plane toggle are their own controls, and a free-text field that
      // rejected most of what you typed would be a worse one.
      const v = document.createElement("span");
      v.textContent = raw;
      row.append(v);
      return row;
    }

    const input = document.createElement("input");
    input.type = "number";
    input.step = "1";
    input.value = String(raw);
    // `change`, not `input`: the geometry is rebuilt per edit, and a refused
    // value (a radius shorter than the semi-diameter) should be reported when you
    // finish typing, not after the first digit.
    input.onchange = () => {
      const n = Number(input.value);
      if (!Number.isFinite(n)) return;
      try {
        uc2.setParameter(id, key, JSON.stringify(n));
        err.textContent = "";
      } catch (e) {
        fail(e);
      }
      refresh();
    };
    row.append(input);

    if (params.overridden.includes(key)) {
      const reset = document.createElement("button");
      reset.textContent = "↺";
      reset.title = "back to the module's design value";
      reset.onclick = () => {
        try {
          uc2.clearParameter(id, key);
          err.textContent = "";
        } catch (e) {
          fail(e);
        }
        refresh();
      };
      row.append(reset);
    }
    return row;
  }

  /// What the lensmaker's equation says this lens should do.
  ///
  /// The point of showing it: it is computed from the numbers above without the
  /// tracer, so the rays on screen are an independent second answer. If the beam
  /// crosses the axis where this says it should, both are probably right — and if
  /// it does not, one of them is wrong and you can see which.
  function predict(p: Paraxial): HTMLElement {
    const el = document.createElement("div");
    el.className = "uc2-predict";
    const f = Number.isFinite(p.efl_mm) ? `${p.efl_mm.toFixed(2)} mm` : "∞ (no power)";
    el.innerHTML =
      `lensmaker: f = <b>${f}</b><br>` +
      `back focus <b>${Number.isFinite(p.bfd_mm) ? p.bfd_mm.toFixed(2) + " mm" : "∞"}</b> ` +
      `past the rear vertex<br>` +
      `n = ${p.nd.toFixed(5)} at ${(p.lambda_um * 1000).toFixed(1)} nm`;
    const why = document.createElement("div");
    why.className = "uc2-note uc2-dim";
    why.textContent =
      "computed from these numbers, not traced — the rays are the independent check";
    el.append(why);
    return el;
  }

  /// The module list (§15.8): what you would have to buy to build this.
  function renderBom(bom: Bom) {
    const box = document.getElementById("uc2Bom")!;
    const rows: HTMLElement[] = [];
    for (const l of bom.lines) {
      const row = document.createElement("div");
      row.className = "uc2-bom";
      const q = document.createElement("span");
      q.className = "uc2-bom-q";
      q.textContent = `${l.quantity}×`;
      const n = document.createElement("span");
      n.className = "uc2-bom-n";
      n.textContent = l.name;
      n.title = `${l.module_id} ${l.version}${l.sku ? ` · ${l.sku}` : ""}`;
      if (!l.resolved) n.className += " uc2-bad";
      const p = document.createElement("span");
      p.className = "uc2-bom-p";
      p.textContent = `€${l.subtotal_eur.toFixed(2)}`;
      row.append(q, n, p);
      rows.push(row);
      // Readiness rides with the line, because the module list is the one view
      // someone reads before spending money.
      for (const r of l.readiness) rows.push(note(r, "warn"));
    }
    if (bom.lines.length === 0) rows.push(note("nothing placed yet", "dim"));

    const total = document.createElement("div");
    total.className = "uc2-bom uc2-bom-t";
    const tl = document.createElement("span");
    tl.className = "uc2-bom-n";
    tl.textContent = "total";
    const tv = document.createElement("span");
    tv.className = "uc2-bom-p";
    tv.textContent = `€${bom.total_eur.toFixed(2)}`;
    total.append(tl, tv);
    rows.push(total);

    // Custom objects are not products and never enter the total (§5.7).
    if (bom.custom_object_count > 0) {
      rows.push(
        note(`custom / not in BOM: ${bom.custom_object_count} drawn object(s)`, "dim"),
      );
    }
    box.replaceChildren(...rows);
  }

  /// Re-read everything and settle the numbers. One settled operation, one call.
  function refresh() {
    const rows = instances();
    renderLayout(rows);
    layerNum.textContent = String(uc2.activeLayer());
    (document.getElementById("uc2Undo") as HTMLButtonElement).disabled = !uc2.canUndo();
    (document.getElementById("uc2Redo") as HTMLButtonElement).disabled = !uc2.canRedo();

    const warnings = JSON.parse(uc2.warningsJson()) as Warning[];
    warnBox.replaceChildren(
      ...warnings.map((w) => note(w.text, w.kind === "unresolved_module" ? "bad" : "warn")),
    );
    if (warnings.length === 0) warnBox.replaceChildren(note("nothing to report", "dim"));

    renderInspector();
    renderPorts(JSON.parse(uc2.portsJson()) as Port[]);
    renderBom(JSON.parse(uc2.bomJson()) as Bom);

    // Numbers come from the f64 reference path, never the interactive f32 one —
    // which is also why this is not in `frame()`.
    const det = uc2.detectorCount() > 0 ? JSON.parse(uc2.detectorJson(0)) : null;
    nums.textContent = det
      ? `detector 0 · ${det.hits} hits · flux ${det.flux.toFixed(4)} · centroid ${det.centroid.toFixed(3)}`
      : "no detector hits";
    frame();
  }

  // --- viewport ----------------------------------------------------------

  const rays = new RayLayer();
  let scale = 3.0; // px per mm
  const pan: V2 = { x: 0.25, y: 0.5 }; // fraction of the viewport
  const w = () => view.clientWidth;
  const h = () => view.clientHeight;
  const toScreen = (p: V2): V2 => ({
    x: p.x * scale + w() * pan.x,
    y: h() * pan.y - p.y * scale,
  });
  /// Screen pixels back to scene millimetres: what the pointer is actually over.
  const toWorld = (sx: number, sy: number): V2 => ({
    x: (sx - w() * pan.x) / scale,
    y: (h() * pan.y - sy) / scale,
  });
  const atPointer = (e: { clientX: number; clientY: number }): V2 => {
    const r = view.getBoundingClientRect();
    return toWorld(e.clientX - r.left, e.clientY - r.top);
  };
  const cellAt = (p: V2): [number, number, number] => {
    const c = uc2.cellAt(p.x, p.y);
    return [c[0], c[1], c[2]];
  };

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    view.width = Math.round(w() * dpr);
    view.height = Math.round(h() * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    frame();
  }
  window.addEventListener("resize", resize);

  view.addEventListener("wheel", (e) => {
    e.preventDefault();
    scale = Math.min(20, Math.max(0.5, scale * Math.exp(-e.deltaY * 0.001)));
    frame();
  });

  // --- pointer interaction -----------------------------------------------

  /// A drag in flight: either a module being moved, or the view being panned.
  type Grab =
    | { what: "module"; id: string; offset: [number, number]; moved: boolean }
    | { what: "pan"; last: V2; moved: boolean };
  let grab: Grab | null = null;

  /// The cell a drop would land on, and what would block it. Drawn, never applied.
  let preview: { cell: [number, number, number]; blocked: string } | null = null;
  /// The library card being dragged in, if any.
  let dragCard: ModuleCard | null = null;

  view.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    view.setPointerCapture(e.pointerId);
    const cell = cellAt(atPointer(e));
    const hit = uc2.instanceAtCell(cell[0], cell[1], cell[2]);

    if (hit) {
      // Selecting any part of a module selects the whole module (§14.7); the
      // grid cell is how the layout view addresses it.
      uc2.selectAtCell(cell[0], cell[1], cell[2]);
      const inst = instances().find((i) => i.instance_id === hit)!;
      // Grab offset, so a multi-cell module does not jump its anchor to the
      // pointer the moment the drag starts.
      grab = {
        what: "module",
        id: hit,
        offset: [cell[0] - inst.anchor_grid[0], cell[1] - inst.anchor_grid[1]],
        moved: false,
      };
      // Everything until pointerup is one undo step (§15.6).
      uc2.beginGesture();
      refresh();
    } else {
      grab = { what: "pan", last: { x: e.clientX, y: e.clientY }, moved: false };
    }
  });

  view.addEventListener("pointermove", (e) => {
    if (!grab) return;
    if (grab.what === "pan") {
      pan.x += (e.clientX - grab.last.x) / w();
      pan.y += (e.clientY - grab.last.y) / h();
      grab.last = { x: e.clientX, y: e.clientY };
      grab.moved = true;
      return frame();
    }

    const c = cellAt(atPointer(e));
    const target: [number, number, number] = [
      c[0] - grab.offset[0],
      c[1] - grab.offset[1],
      uc2.activeLayer(),
    ];
    // Ask before moving: a blocked cell shows as blocked and the module stays
    // where it was, rather than throwing once per pointer event.
    const blocker = uc2.blockerFor(grab.id, target[0], target[1], target[2]);
    preview = { cell: target, blocked: blocker };
    if (!blocker) {
      try {
        uc2.moveModule(grab.id, target[0], target[1], target[2]);
        grab.moved = true;
        err.textContent = "";
      } catch (ex) {
        fail(ex);
      }
    }
    // Rays follow the drag through the f32 path; the numbers wait for the drop.
    frame();
  });

  view.addEventListener("pointerup", () => {
    if (!grab) return;
    const g = grab;
    grab = null;
    preview = null;
    if (g.what === "pan") {
      // A click on empty grid, rather than a pan, clears the selection.
      if (!g.moved) {
        uc2.clearSelection();
        refresh();
      }
      return;
    }
    // Settle: one undo step for the whole drag, and f64 numbers now that the
    // module has stopped (§15.6).
    uc2.endGesture();
    refresh();
  });

  view.addEventListener("pointercancel", () => {
    if (grab?.what === "module") uc2.endGesture();
    grab = null;
    preview = null;
    frame();
  });

  // --- drag in from the library ------------------------------------------

  view.addEventListener("dragover", (e) => {
    if (!dragCard) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
    const cell = cellAt(atPointer(e));
    const blocked = uc2.instanceAtCell(cell[0], cell[1], cell[2]);
    preview = { cell, blocked };
    frame();
  });

  view.addEventListener("dragleave", () => {
    preview = null;
    frame();
  });

  view.addEventListener("drop", (e) => {
    if (!dragCard) return;
    e.preventDefault();
    const m = dragCard;
    dragCard = null;
    preview = null;
    const cell = cellAt(atPointer(e));
    try {
      uc2.placeModule(m.uuid, m.version, cell[0], cell[1], cell[2], 0);
      uc2.selectAtCell(cell[0], cell[1], cell[2]);
      err.textContent = "";
    } catch (ex) {
      fail(ex);
    }
    refresh();
  });

  // --- drawing -----------------------------------------------------------

  function frame() {
    const W = w(), H = h();
    const dpr = window.devicePixelRatio || 1;
    ctx.fillStyle = "#06080d";
    ctx.fillRect(0, 0, W, H);

    drawGrid(ctx, toScreen, scale, W, H, grid);

    const sel = uc2.selectedInstance();
    const prints = JSON.parse(uc2.footprintsJson()) as Footprint[];
    if (on("footprints")) {
      drawFootprints(ctx, prints, toScreen, scale, sel, on("onion"));
    }
    if (on("labels")) drawLabels(ctx, prints, toScreen, on("onion"));
    if (preview) drawPreview(ctx, preview, toScreen, scale, grid);

    if (on("optics")) {
      drawOptics(ctx, JSON.parse(uc2.opticalOverlayJson()) as Optic[], toScreen, sel);
    }

    if (on("rays")) {
      const segs = uc2.traceColored();
      rays.draw(segs, toScreen, { ghostOpacity: 0, exposure: EXPOSURE, dpr, width: W, height: H });
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalCompositeOperation = "lighter";
      ctx.drawImage(rays.canvas, 0, 0);
      ctx.restore();
    }

    // Last, so it stays readable: the ray layer composites with `lighter` and
    // washes out anything drawn under it, and a port marker is worth nothing if
    // it cannot be seen against the beam it is describing.
    if (on("ports")) {
      const ports = (JSON.parse(uc2.portsJson()) as Port[]).filter(
        (p) => p.layer === uc2.activeLayer(),
      );
      drawPorts(ctx, ports, toScreen);
    }
  }

  resize();
  refresh();
}

/// The grid the modules snap to, at the profile's own pitch.
function drawGrid(
  ctx: CanvasRenderingContext2D,
  toScreen: (p: V2) => V2,
  scale: number,
  W: number,
  H: number,
  grid: Grid,
) {
  const [px, py] = grid.pitch_mm;
  const [sx, sy] = [px * scale, py * scale];
  if (Math.min(sx, sy) < 6) return; // too dense to read
  // A cell origin is the cell's *centre* (§6.4), so the lines between cells sit
  // half a pitch off it. Drawn from the origin they would run straight through
  // every module instead of between them.
  const o = toScreen({ x: -px / 2, y: py / 2 });
  ctx.strokeStyle = "rgba(120,140,170,0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = o.x % sx; x < W; x += sx) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
  }
  for (let y = o.y % sy; y < H; y += sy) {
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
  }
  ctx.stroke();
}

/// Module footprints: where the cubes physically are.
///
/// Presentation only. The optical fragment is what the tracer reads (§15.5), so
/// a footprint is never allowed to imply that something is being simulated.
function drawFootprints(
  ctx: CanvasRenderingContext2D,
  rows: Footprint[],
  toScreen: (p: V2) => V2,
  scale: number,
  selected: string,
  onion: boolean,
) {
  for (const f of rows) {
    if (!f.active && !onion) continue;
    const [cw, ch] = f.cell_size_mm;
    const isSel = f.instance_id === selected;
    for (const [cx, cy] of f.cells_mm) {
      const a = toScreen({ x: cx - cw / 2, y: cy + ch / 2 });
      const wpx = cw * scale, hpx = ch * scale;
      ctx.fillStyle = f.active ? `${f.color}22` : `${f.color}0d`;
      ctx.strokeStyle = isSel ? "#7fd6ff" : f.active ? `${f.color}88` : `${f.color}33`;
      ctx.lineWidth = isSel ? 2 : 1;
      ctx.fillRect(a.x, a.y, wpx, hpx);
      ctx.strokeRect(a.x, a.y, wpx, hpx);
    }
  }
}

/// Module names, on their anchor cells.
///
/// This is the "physical overlay" of §15.5 standing in for artwork: the
/// development fixture ships no icons, and the real ones come with the module
/// repository's assets (M1). It is presentation either way.
function drawLabels(
  ctx: CanvasRenderingContext2D,
  rows: Footprint[],
  toScreen: (p: V2) => V2,
  onion: boolean,
) {
  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const f of rows) {
    if (!f.active && !onion) continue;
    const c = f.cells_mm[0];
    if (!c) continue;
    const p = toScreen({ x: c[0], y: c[1] });
    ctx.fillStyle = f.active ? "rgba(220,235,255,0.55)" : "rgba(220,235,255,0.20)";
    ctx.fillText(f.instance_id, p.x, p.y);
  }
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

/// Where a drop or a drag would land, and whether it can.
function drawPreview(
  ctx: CanvasRenderingContext2D,
  p: { cell: [number, number, number]; blocked: string },
  toScreen: (p: V2) => V2,
  scale: number,
  grid: Grid,
) {
  const [px, py] = grid.pitch_mm;
  const [cw, ch] = grid.cell_size_mm;
  const o = toScreen({ x: p.cell[0] * px - cw / 2, y: p.cell[1] * py + ch / 2 });
  ctx.save();
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 2;
  // Red is the whole point: collision feedback has to be visible mid-drag, not
  // reported afterwards in a text box.
  ctx.strokeStyle = p.blocked ? "#ff6b57" : "#7fe0a0";
  ctx.fillStyle = p.blocked ? "rgba(255,107,87,0.12)" : "rgba(127,224,160,0.10)";
  ctx.fillRect(o.x, o.y, cw * scale, ch * scale);
  ctx.strokeRect(o.x, o.y, cw * scale, ch * scale);
  ctx.restore();
}

/// The optical geometry: what the light actually meets.
function drawOptics(
  ctx: CanvasRenderingContext2D,
  optics: Optic[],
  toScreen: (p: V2) => V2,
  selected: string,
) {
  const tone: Record<string, string> = {
    glass: "#7fd6ff",
    mirror: "#cfd8e3",
    absorber: "#8a94a6",
    other: "#9fb3c8",
  };
  for (const o of optics) {
    const hot = o.instance !== "" && o.instance === selected;
    ctx.lineWidth = hot ? 2 : 1;

    if (o.kind === "source") {
      const p = toScreen({ x: o.points[0], y: o.points[1] });
      const d = o.direction ?? [1, 0];
      const t = toScreen({ x: o.points[0] + d[0] * 8, y: o.points[1] + d[1] * 8 });
      ctx.strokeStyle = hot ? "#7fd6ff" : "#e8c46a";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();
      continue;
    }

    ctx.strokeStyle = hot
      ? "#7fd6ff"
      : o.kind === "detector"
        ? "#e07fd6"
        : (tone[o.material ?? "other"] ?? tone.other);
    ctx.beginPath();
    for (let i = 0; i + 1 < o.points.length; i += 2) {
      const p = toScreen({ x: o.points[i], y: o.points[i + 1] });
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
  }
}

/// Ports, coloured by what they found across from them (§14.6).
///
/// Advisory: none of this reaches the tracer. It is the only view that can show a
/// module mounted facing away, which the footprint and the rays both hide.
function drawPorts(ctx: CanvasRenderingContext2D, ports: Port[], toScreen: (p: V2) => V2) {
  const STEM_MM = 8;
  for (const p of ports) {
    const s = p.status.state;
    const colour = s === "connected" ? "#7fe0a0" : s === "misaligned" ? "#ff9b8a" : "#8aa1b8";
    const o = toScreen({ x: p.position_mm[0], y: p.position_mm[1] });
    // A stem along the outward normal, so which way a port faces is visible. Two
    // mated ports draw their stems into each other's cubes, which is what a
    // connection looks like.
    const t = toScreen({
      x: p.position_mm[0] + p.direction[0] * STEM_MM,
      y: p.position_mm[1] + p.direction[1] * STEM_MM,
    });
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(o.x, o.y);
    ctx.lineTo(t.x, t.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(o.x, o.y, 4, 0, Math.PI * 2);
    // Filled reads as mated, hollow as open: legible without the legend.
    if (s === "connected") ctx.fill();
    else ctx.stroke();
  }
}

/// Say where the catalogue came from, and whether it is real.
function renderProvenance(uc2: OpenUc2) {
  const el = document.getElementById("uc2Src")!;
  const src = JSON.parse(uc2.catalogSourceJson()) as { repository: string; commit: string } | null;
  if (!src) {
    el.textContent = "no catalogue";
    return;
  }
  const short = src.commit.slice(0, 7);
  if (UNPINNED.test(src.commit)) {
    // Nobody should mistake placeholder modules, part numbers, or prices for
    // openUC2 products they can order.
    el.className = "uc2-bad";
    el.textContent = `${src.repository} · unpinned — development fixture, not real openUC2 module data`;
  } else {
    el.className = "";
    el.textContent = `${src.repository} @ ${short}`;
  }
}

/// The coverage report (§3, §11.4): what the catalogue does and does not cover.
function renderCoverage(c: Coverage | null) {
  const el = document.getElementById("uc2Cov")!;
  if (!c) {
    el.textContent = "";
    return;
  }
  const opt = (k: string) => c.optical[k] ?? 0;
  const rows = [
    `${c.total} modules · ${c.layout_ready} layout-ready`,
    `optical: ${opt("validated")} validated · ${opt("idealized")} idealized · ${opt("placeholder")} placeholder · ${opt("absent")} absent`,
    `${c.traceable} can contribute optics · ${c.no_fragment_declared} declare none`,
  ];
  if (c.declared_fragment_asset_missing > 0) {
    rows.push(`${c.declared_fragment_asset_missing} declare optics whose asset did not load`);
  }
  el.replaceChildren(
    ...rows.map((t) => {
      const d = document.createElement("div");
      d.textContent = t;
      return d;
    }),
  );
}
