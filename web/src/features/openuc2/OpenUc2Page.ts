// The openUC2 mode (§15).
//
// Reached at `#openuc2`. The freeform Canvas is untouched and still the default.
//
// What lives here is UI state only: a search box, which filters are on, the
// viewport. The project — catalogue, instances, placements, the materialized
// scene — lives in Rust behind the `OpenUc2` handle, and this file asks for a
// view whenever it needs to draw one (§0.3: no assembly state mirrored into JS).
// The card list is re-derived from `catalogView()` rather than kept as a second
// copy of the catalogue.
//
// Rays go through the same `RayLayer` the freeform route uses (§15.1: do not
// fork the renderer), and any displayed *number* comes from `detectorJson`,
// which is the f64 reference path (§4.1).

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

type Footprint = {
  instance_id: string;
  color: string;
  layer: number;
  active: boolean;
  cells_mm: [number, number][];
  cell_size_mm: [number, number];
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

  // --- placement (click-to-add) ------------------------------------------

  const err = document.getElementById("uc2Err")!;

  /// Click-to-add lands on the first free cell of the active layer. Drag-to-place
  /// and grid interaction are C4; §15.4 allows click-to-add to come first.
  function place(m: ModuleCard) {
    const taken = new Set(
      (JSON.parse(uc2.instancesView()) as Instance[]).map((i) => i.anchor_grid.join(",")),
    );
    const layer = uc2.activeLayer();
    let cell: [number, number, number] = [0, 0, layer];
    for (let x = 0; x < 64; x++) {
      if (!taken.has(`${x},0,${layer}`)) {
        cell = [x, 0, layer];
        break;
      }
    }
    try {
      uc2.placeModule(m.uuid, m.version, cell[0], cell[1], cell[2], 0);
      err.textContent = "";
    } catch (e) {
      // Structured, not a stack trace: a collision is a normal answer.
      err.textContent = String(e);
    }
    refresh();
  }

  // The folded-singlet demo (§16.4), the same layout the headless test builds.
  // Here so the mode can be checked in one click rather than by hand-placing
  // four modules onto the right cells.
  const demoBtn = document.getElementById("uc2Demo") as HTMLButtonElement;
  demoBtn.onclick = () => {
    const byRole = (id: string) => cards.find((c) => c.id === id);
    const plan: [string, [number, number, number], number][] = [
      ["source-collimated-3line-1x1", [0, 0, 0], 0],
      ["lens-pcx-nbk7-1x1", [1, 0, 0], 0],
      ["mirror-flat-45-1x1", [2, 0, 0], 0],
      ["detector-line-1x1", [2, -1, 0], 1],
    ];
    err.textContent = "";
    for (const i of JSON.parse(uc2.instancesView()) as Instance[]) {
      uc2.removeModule(i.instance_id);
    }
    for (const [id, cell, rot] of plan) {
      const m = byRole(id);
      if (!m) {
        err.textContent = `the catalogue has no ${id}`;
        break;
      }
      try {
        uc2.placeModule(m.uuid, m.version, cell[0], cell[1], cell[2], rot);
      } catch (e) {
        err.textContent = String(e);
      }
    }
    refresh();
  };

  const layout = document.getElementById("uc2Layout")!;
  const warnBox = document.getElementById("uc2Warn")!;
  const nums = document.getElementById("uc2Nums")!;

  function refresh() {
    const instances = JSON.parse(uc2.instancesView()) as Instance[];
    layout.replaceChildren(
      ...instances.map((i) => {
        const row = document.createElement("div");
        row.className = "uc2-row";
        const label = document.createElement("span");
        label.textContent = `${i.instance_id} ${i.name}`;
        if (!i.resolved) label.className = "uc2-bad";
        else if (!i.materialized) label.className = "uc2-warn";
        row.append(label);

        // Nudge, rather than drag: dragging on the grid is C4. Without any way
        // to reposition, though, click-to-add can only ever build a straight
        // line along +x, and a folded path would be unreachable.
        for (const [label, dx, dy] of [["←", -1, 0], ["→", 1, 0], ["↑", 0, 1], ["↓", 0, -1]] as const) {
          const b = document.createElement("button");
          b.textContent = label;
          b.title = `move one cell ${label}`;
          b.onclick = () => {
            try {
              uc2.moveModule(i.instance_id, i.anchor_grid[0] + dx, i.anchor_grid[1] + dy, i.anchor_grid[2]);
              err.textContent = "";
            } catch (e) {
              err.textContent = String(e);
            }
            refresh();
          };
          row.append(b);
        }

        const rot = document.createElement("button");
        rot.textContent = "⟳";
        rot.title = "rotate 90°";
        rot.onclick = () => {
          try {
            uc2.rotateModule(i.instance_id, ((i.rotation_quadrants + 1) % 4) as number);
            err.textContent = "";
          } catch (e) {
            err.textContent = String(e);
          }
          refresh();
        };
        const del = document.createElement("button");
        del.textContent = "×";
        del.title = "remove";
        del.onclick = () => {
          uc2.removeModule(i.instance_id);
          refresh();
        };
        row.append(rot, del);
        return row;
      }),
    );
    if (instances.length === 0) {
      layout.textContent = "nothing placed yet";
    }

    const warnings = JSON.parse(uc2.warningsJson()) as Warning[];
    warnBox.replaceChildren(
      ...warnings.map((w) => note(w.text, w.kind === "unresolved_module" ? "bad" : "warn")),
    );

    // Numbers come from the f64 reference path, never the interactive f32 one.
    const det = uc2.detectorCount() > 0 ? JSON.parse(uc2.detectorJson(0)) : null;
    nums.textContent = det
      ? `detector 0 · ${det.hits} hits · flux ${det.flux.toFixed(4)} · centroid ${det.centroid.toFixed(3)}`
      : "no detector hits";
    frame();
  }

  // --- viewport ----------------------------------------------------------

  const rays = new RayLayer();
  let scale = 3.0; // px per mm
  let pan: V2 = { x: 0.25, y: 0.5 }; // fraction of the viewport
  const w = () => view.clientWidth;
  const h = () => view.clientHeight;
  const toScreen = (p: V2): V2 => ({
    x: p.x * scale + w() * pan.x,
    y: h() * pan.y - p.y * scale,
  });

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
  let dragging = false;
  let last: V2 = { x: 0, y: 0 };
  view.addEventListener("pointerdown", (e) => {
    dragging = true;
    last = { x: e.clientX, y: e.clientY };
    view.setPointerCapture(e.pointerId);
  });
  view.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    pan.x += (e.clientX - last.x) / w();
    pan.y += (e.clientY - last.y) / h();
    last = { x: e.clientX, y: e.clientY };
    frame();
  });
  view.addEventListener("pointerup", () => (dragging = false));

  function frame() {
    const W = w(), H = h();
    const dpr = window.devicePixelRatio || 1;
    ctx.fillStyle = "#06080d";
    ctx.fillRect(0, 0, W, H);

    drawGrid(ctx, toScreen, scale, W, H);
    drawFootprints(ctx, JSON.parse(uc2.footprintsJson()) as Footprint[], toScreen, scale);

    const segs = uc2.traceColored();
    rays.draw(segs, toScreen, { ghostOpacity: 0, exposure: EXPOSURE, dpr, width: W, height: H });
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = "lighter";
    ctx.drawImage(rays.canvas, 0, 0);
    ctx.restore();
  }

  resize();
  refresh();
}

/// The 50 mm grid the modules snap to.
function drawGrid(
  ctx: CanvasRenderingContext2D,
  toScreen: (p: V2) => V2,
  scale: number,
  W: number,
  H: number,
) {
  const pitch = 50;
  const step = pitch * scale;
  if (step < 6) return; // too dense to read
  const o = toScreen({ x: 0, y: 0 });
  ctx.strokeStyle = "rgba(120,140,170,0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = o.x % step; x < W; x += step) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
  }
  for (let y = o.y % step; y < H; y += step) {
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
) {
  for (const f of rows) {
    const [cw, ch] = f.cell_size_mm;
    for (const [cx, cy] of f.cells_mm) {
      const a = toScreen({ x: cx - cw / 2, y: cy + ch / 2 });
      const wpx = cw * scale, hpx = ch * scale;
      ctx.fillStyle = f.active ? `${f.color}22` : `${f.color}0d`;
      ctx.strokeStyle = f.active ? `${f.color}88` : `${f.color}33`;
      ctx.lineWidth = 1;
      ctx.fillRect(a.x, a.y, wpx, hpx);
      ctx.strokeRect(a.x, a.y, wpx, hpx);
    }
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
