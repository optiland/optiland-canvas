// The gallery: ready-made designs to open on the canvas.
//
// Today it holds Optiland's sample lenses (`optiland.samples`), built as canvas
// scenes by `python -m optiland_canvas.gallery web/public/gallery`. That writes
// one `.ocanvas` per design and an `index.json` the cards are drawn from: each
// design's name, what it is, its first-order numbers, a drawing of its section,
// what the scene holds only approximately, and, for a design the canvas cannot
// build yet, why not. Opening a card loads its scene exactly as "open" loads a
// file, so a gallery design is an ordinary scene from then on.
//
// Every ported design is checked ray for ray against Optiland's own trace
// (`oc-trace/tests/gallery3.rs`); the cards say so, and say what is not exact.

export type GallerySpecs = {
  efl_mm: number;
  epd_mm: number;
  fno?: number;
  object_distance_mm?: number;
  object_na?: number;
  field: number;
  field_unit: "deg" | "mm";
  fields: number;
  wavelengths_um: number[];
  elements: number;
  surfaces: number;
};

export type GalleryDesign = {
  id: string;
  name: string;
  category: string;
  source: string;
  description: string;
  reference?: string;
  status: "ported" | "not_ported";
  file?: string;
  specs?: GallerySpecs;
  notes?: string[];
  preview?: { glass: string; max_index_error: number; band_um: [number, number] }[];
  outline?: { view_box: string; glass: string; stop: string; image: string };
  reasons?: string[];
};

export type GalleryIndex = { generator: string; optiland: string; designs: GalleryDesign[] };

export type Gallery = {
  /** Show the gallery (fetching its index the first time). */
  show: () => Promise<void>;
  hide: () => void;
  /** Open the design `id` on the canvas. False, with the reason shown, when it
   *  does not open. */
  open: (id: string) => Promise<boolean>;
};

/** Where the gallery is served from, next to the page. */
const BASE = "gallery/";

export function initGallery(opts: {
  /** Load a scene document; false when it does not load. */
  load: (name: string, text: string) => boolean;
  /** Frame the loaded scene in whichever view is showing. */
  frame: () => void;
  /** Say what was opened, and what in it is approximate. */
  opened: (design: GalleryDesign) => void;
}): Gallery {
  const overlay = document.getElementById("galleryOverlay")!;
  const sub = document.getElementById("gallerySub")!;
  const cats = document.getElementById("galleryCats")!;
  const grid = document.getElementById("galleryGrid")!;
  const out = document.getElementById("galleryOut")!;
  const err = document.getElementById("galleryErr")!;

  let index: GalleryIndex | null = null;
  let category = "All";

  async function fetchIndex(): Promise<GalleryIndex> {
    if (index) return index;
    const res = await fetch(`${BASE}index.json`);
    if (!res.ok) throw new Error(`the gallery index did not load (${res.status})`);
    index = (await res.json()) as GalleryIndex;
    return index;
  }

  const fmt = (v: number, digits = 4) => {
    const a = Math.abs(v);
    return a >= 100 ? v.toFixed(1) : a >= 10 ? v.toFixed(2) : v.toPrecision(digits).replace(/\.?0+$/, "");
  };

  function specsLine(s: GallerySpecs): string {
    const parts = [`f ${fmt(s.efl_mm)} mm`];
    if (s.fno !== undefined) parts.push(`f/${fmt(s.fno, 3)}`);
    if (s.object_na !== undefined) parts.push(`NA ${s.object_na}`);
    parts.push(s.field_unit === "deg" ? `±${fmt(s.field, 3)}°` : `±${fmt(s.field, 3)} mm`);
    const nm = s.wavelengths_um.map((w) => (w >= 1 ? `${fmt(w, 3)} µm` : `${Math.round(w * 1000)} nm`));
    parts.push(nm.length > 2 ? `${nm.length} λ` : nm.join(", "));
    parts.push(`${s.elements} element${s.elements === 1 ? "" : "s"}`);
    return parts.join(" · ");
  }

  function drawing(o: NonNullable<GalleryDesign["outline"]>): SVGSVGElement {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", o.view_box);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.setAttribute("aria-hidden", "true");
    for (const [cls, d] of [["glass", o.glass], ["stop", o.stop], ["image", o.image]] as const) {
      if (!d) continue;
      const p = document.createElementNS(ns, "path");
      p.setAttribute("class", cls);
      p.setAttribute("d", d);
      svg.appendChild(p);
    }
    return svg;
  }

  function card(d: GalleryDesign): HTMLButtonElement {
    const b = document.createElement("button");
    b.className = "g-card";
    b.dataset.design = d.id;
    b.title = [d.description, d.reference ? `Reference: ${d.reference}` : "", d.source].filter(Boolean).join("\n");
    if (d.outline) b.appendChild(drawing(d.outline));
    const name = document.createElement("div");
    name.className = "g-name";
    name.textContent = d.name;
    const cat = document.createElement("div");
    cat.className = "g-cat";
    cat.textContent = d.category;
    b.append(name, cat);
    if (d.specs) {
      const specs = document.createElement("div");
      specs.className = "g-specs";
      specs.textContent = specsLine(d.specs);
      b.appendChild(specs);
    }
    const desc = document.createElement("div");
    desc.className = "g-desc";
    desc.textContent = d.reference ? `${d.description} ${d.reference}` : d.description;
    b.appendChild(desc);
    if (d.notes && d.notes.length) {
      const n = document.createElement("div");
      n.className = "g-note";
      n.textContent = `⚠ ${d.notes.length} approximation${d.notes.length === 1 ? "" : "s"}: hover for detail`;
      n.title = d.notes.join("\n");
      b.appendChild(n);
    }
    b.addEventListener("click", () => void open(d.id));
    return b;
  }

  function render() {
    if (!index) return;
    const ported = index.designs.filter((d) => d.status === "ported");
    const refused = index.designs.filter((d) => d.status !== "ported");
    sub.textContent =
      `${ported.length} of Optiland's ${index.designs.length} sample designs (Optiland ${index.optiland}), ` +
      `each checked ray for ray against Optiland's own trace`;

    const categories = ["All", ...new Set(ported.map((d) => d.category))];
    cats.replaceChildren(
      ...categories.map((c) => {
        const b = document.createElement("button");
        b.textContent = c;
        b.classList.toggle("active", c === category);
        b.addEventListener("click", () => {
          category = c;
          render();
        });
        return b;
      }),
    );
    grid.replaceChildren(...ported.filter((d) => category === "All" || d.category === category).map(card));

    out.replaceChildren();
    if (refused.length && category === "All") {
      const h = document.createElement("h3");
      h.textContent = "Not in the canvas yet";
      out.appendChild(h);
      for (const d of refused) {
        const row = document.createElement("div");
        const name = document.createElement("b");
        name.textContent = d.name;
        row.append(name, `: ${(d.reasons ?? []).join("; ")}`);
        out.appendChild(row);
      }
    }
  }

  async function show() {
    overlay.hidden = false;
    err.textContent = "";
    try {
      await fetchIndex();
      render();
      (grid.querySelector(".g-card") as HTMLElement | null)?.focus();
    } catch (e) {
      err.textContent = String(e instanceof Error ? e.message : e);
    }
  }

  function hide() {
    overlay.hidden = true;
  }

  async function open(id: string): Promise<boolean> {
    err.textContent = "";
    let design: GalleryDesign | undefined;
    const busy = grid.querySelector(`[data-design="${CSS.escape(id)}"]`);
    busy?.classList.add("g-busy");
    try {
      design = (await fetchIndex()).designs.find((d) => d.id === id);
      if (!design) throw new Error(`the gallery has no design "${id}"`);
      if (design.status !== "ported" || !design.file) {
        throw new Error(`${design.name} is not in the canvas yet: ${(design.reasons ?? []).join("; ")}`);
      }
      const res = await fetch(`${BASE}${design.file}`);
      if (!res.ok) throw new Error(`${design.file} did not load (${res.status})`);
      const text = await res.text();
      if (!opts.load(design.name, text)) throw new Error(`${design.name} did not open`);
    } catch (e) {
      busy?.classList.remove("g-busy");
      // A design asked for by a link fails in front of the gallery, so the
      // reason has somewhere to be read and the other designs are a click away.
      if (overlay.hidden) await show();
      err.textContent = String(e instanceof Error ? e.message : e);
      return false;
    }
    busy?.classList.remove("g-busy");
    hide();
    opts.frame();
    opts.opened(design);
    return true;
  }

  document.getElementById("galleryOpen")!.addEventListener("click", () => void show());
  document.getElementById("galleryClose")!.addEventListener("click", hide);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) hide();
  });
  // While the gallery is up, keys are its own: Escape closes it, Tab and Enter
  // move among and open the cards, and nothing reaches the canvas's shortcuts
  // (a Delete here must not delete the selection behind it).
  window.addEventListener(
    "keydown",
    (e) => {
      if (overlay.hidden) return;
      e.stopPropagation();
      if (e.key === "Escape") hide();
    },
    true,
  );

  return { show, hide, open };
}
