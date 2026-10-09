// Card sections shared by the 2D section's cards (slice3d.ts) and the 3D
// view's (render3d/cards3d.ts): an aperture's opening and mask, and the link
// that locks an element to another.
//
// Every value is read back from Rust on render and written through one call
// on commit; a refused edit leaves the scene as it was, says why in the card,
// and the next render puts the real values back in the fields.

import type { Canvas } from "../pkg/oc_wasm.js";

export type Outline =
  | { kind: "circle"; d: number }
  | { kind: "ring"; dIn: number; d: number }
  | { kind: "rect"; w: number; h: number }
  | { kind: "none" }
  | { kind: "polygon" };
export type ApertureShape = { opening: Outline; mask: Outline };

/** Half width and half height an outline covers; `null` when unbounded. */
export function halfExtent(o: Outline): [number, number] | null {
  switch (o.kind) {
    case "circle": case "ring": return [o.d / 2, o.d / 2];
    case "rect": return [o.w / 2, o.h / 2];
    default: return null;
  }
}

/** The outline `o` becomes as kind `kind`, keeping its size as far as the
 *  kinds share one. */
export function reshape(o: Outline, kind: Outline["kind"], around?: Outline): Outline {
  const e = halfExtent(o) ?? (around ? halfExtent(around)?.map((v) => v * 2.5) as [number, number] : null) ?? [10, 10];
  const d = 2 * Math.max(e[0], e[1]);
  switch (kind) {
    case "circle": return { kind, d };
    case "ring": return { kind, d, dIn: o.kind === "ring" ? o.dIn : d / 3 };
    case "rect": return { kind, w: 2 * e[0], h: 2 * e[1] };
    default: return { kind: "none" };
  }
}

/** The same outline, by value (the key order Rust writes is not the page's). */
function sameOutline(a: Outline, b: Outline): boolean {
  const k = (o: Outline) => o as unknown as Record<string, unknown>;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  return [...keys].every((key) => k(a)[key] === k(b)[key]);
}

/** Does the opening lie strictly inside the mask? The rule Rust applies. */
function fits(opening: Outline, mask: Outline): boolean {
  const o = halfExtent(opening), m = halfExtent(mask);
  if (!o || !m) return true;
  const r = opening.kind === "rect" ? Math.hypot(o[0], o[1]) : o[0];
  return mask.kind === "rect" ? o[0] < m[0] && o[1] < m[1] : r < m[0];
}

/** The mask grown, if it has to, to hold the opening with a rim: what a hole
 *  made larger than its plate means. A mask made smaller than its hole is a
 *  different edit, and is refused rather than grown. */
export function roomForOpening(next: ApertureShape): ApertureShape {
  if (fits(next.opening, next.mask)) return next;
  const o = halfExtent(next.opening)!;
  const rim = 1.25;
  // Whole millimetres: a plate grown to fit reads as a size someone would cut.
  const up = (v: number) => Math.ceil(v);
  const mask: Outline =
    next.mask.kind === "rect"
      ? { kind: "rect", w: Math.max(next.mask.w, up(2 * rim * o[0])), h: Math.max(next.mask.h, up(2 * rim * o[1])) }
      : next.mask.kind === "circle"
        ? { kind: "circle", d: Math.max(next.mask.d, up(2 * rim * (next.opening.kind === "rect" ? Math.hypot(o[0], o[1]) : o[0]))) }
        : next.mask;
  return { opening: next.opening, mask };
}

const q = (root: HTMLElement, sel: string) => root.querySelector(sel) as HTMLElement | null;
const input = (root: HTMLElement, name: string) =>
  root.querySelector(`input[data-name="${name}"]`) as HTMLInputElement | null;
const select = (root: HTMLElement, name: string) =>
  root.querySelector(`select[data-name="${name}"]`) as HTMLSelectElement | null;
const setV = (root: HTMLElement, name: string, v: number) => {
  const el = input(root, name);
  if (el && document.activeElement !== el) el.value = String(Number(v.toFixed(4)));
};
const setS = (root: HTMLElement, name: string, v: string) => {
  const el = select(root, name);
  if (el && document.activeElement !== el) el.value = v;
};
const show = (root: HTMLElement, cls: string, on: boolean) => {
  const el = q(root, cls);
  if (el) el.style.display = on ? "" : "none";
};

// ---------------------------------------------------------------------------
// Apertures
// ---------------------------------------------------------------------------

export function readAperture(scene: Canvas, id: number): ApertureShape | null {
  return JSON.parse(scene.apertureShapeJSON(id));
}

/** Show an aperture's opening and mask in the card; hide the section for
 *  anything else. */
export function renderAperture(scene: Canvas, id: number, root: HTMLElement) {
  const ap = readAperture(scene, id);
  show(root, ".c3-aperture", !!ap);
  if (!ap) return;
  const { opening: o, mask: m } = ap;
  setS(root, "apOpenKind", o.kind);
  show(root, ".c3-ap-od", o.kind === "circle" || o.kind === "ring");
  show(root, ".c3-ap-oin", o.kind === "ring");
  show(root, ".c3-ap-ow", o.kind === "rect");
  show(root, ".c3-ap-oh", o.kind === "rect");
  if (o.kind === "circle" || o.kind === "ring") setV(root, "apOpenD", o.d);
  if (o.kind === "ring") setV(root, "apOpenDIn", o.dIn);
  if (o.kind === "rect") { setV(root, "apOpenW", o.w); setV(root, "apOpenH", o.h); }
  setS(root, "apMaskKind", m.kind);
  show(root, ".c3-ap-md", m.kind === "circle");
  show(root, ".c3-ap-mw", m.kind === "rect");
  show(root, ".c3-ap-mh", m.kind === "rect");
  if (m.kind === "circle") setV(root, "apMaskD", m.d);
  if (m.kind === "rect") { setV(root, "apMaskW", m.w); setV(root, "apMaskH", m.h); }
}

/** The aperture the card's fields describe. A kind just switched starts from
 *  the old outline's size. */
function apertureFromCard(root: HTMLElement, was: ApertureShape): ApertureShape {
  const num = (n: string) => Number(input(root, n)?.value);
  const openKind = (select(root, "apOpenKind")?.value ?? was.opening.kind) as Outline["kind"];
  const maskKind = (select(root, "apMaskKind")?.value ?? was.mask.kind) as Outline["kind"];
  let opening: Outline;
  if (openKind !== was.opening.kind) opening = reshape(was.opening, openKind);
  else if (openKind === "circle") opening = { kind: "circle", d: num("apOpenD") };
  else if (openKind === "ring") opening = { kind: "ring", d: num("apOpenD"), dIn: num("apOpenDIn") };
  else if (openKind === "rect") opening = { kind: "rect", w: num("apOpenW"), h: num("apOpenH") };
  else opening = was.opening;
  let mask: Outline;
  if (maskKind !== was.mask.kind) mask = reshape(was.mask, maskKind, opening);
  else if (maskKind === "circle") mask = { kind: "circle", d: num("apMaskD") };
  else if (maskKind === "rect") mask = { kind: "rect", w: num("apMaskW"), h: num("apMaskH") };
  else mask = { kind: "none" };
  return { opening, mask };
}

/** Write the card's aperture fields back. Returns whether it was an aperture. */
export function commitAperture(scene: Canvas, id: number, root: HTMLElement): boolean {
  const was = readAperture(scene, id);
  if (!was) return false;
  let next = apertureFromCard(root, was);
  // A hole made larger takes its plate with it.
  if (sameOutline(next.mask, was.mask) || next.mask.kind !== was.mask.kind) next = roomForOpening(next);
  const unchanged = sameOutline(next.opening, was.opening) && sameOutline(next.mask, was.mask);
  const why = unchanged ? "" : scene.setApertureShape(id, JSON.stringify(next));
  const note = q(root, ".c3-apNote");
  if (note) note.textContent = why ? `refused: ${why}` : "";
  return true;
}

// ---------------------------------------------------------------------------
// Links
// ---------------------------------------------------------------------------

type Obj = { id: number; kind: string; construction?: string };
type Link = { parent: number; mode: "follow" | "front" | "face"; gap: number | null };

/** Ids the sun turns: a link may not. */
function heliostats(scene: Canvas): Set<number> {
  const s = JSON.parse(scene.solarJSON()) as { heliostats: { mirror: number }[] } | null;
  return new Set(s?.heliostats.map((h) => h.mirror) ?? []);
}

const describe = (o: Obj, helio: Set<number>) =>
  `#${o.id} · ${helio.has(o.id) ? "heliostat" : o.kind === "profile" ? (o.construction ?? "lens") : o.kind}`;

/** Show what an element is locked to. Hidden for a source and for a
 *  heliostat mirror, which the sun turns. */
export function renderLink(scene: Canvas, id: number, root: HTMLElement, objects: Obj[]) {
  const helio = heliostats(scene);
  const self = objects.find((o) => o.id === id);
  const lockable = !!self && self.kind !== "source" && !helio.has(id);
  show(root, ".c3-link", lockable);
  if (!lockable) return;
  const link = JSON.parse(scene.linkJSON(id)) as Link | null;
  const sel = select(root, "linkParent");
  if (sel && document.activeElement !== sel) {
    sel.replaceChildren(
      new Option("nothing (free)", ""),
      ...objects
        .filter((o) => o.id !== id && o.kind !== "source")
        .map((o) => new Option(describe(o, helio), String(o.id))),
    );
    sel.value = link ? String(link.parent) : "";
  }
  setS(root, "linkMode", link?.mode ?? "front");
  show(root, ".c3-linkMode", !!link);
  show(root, ".c3-linkGap", link?.mode === "front");
  if (link?.gap != null) setV(root, "linkGap", link.gap);
  const note = q(root, ".c3-linkNote");
  if (note && !note.dataset.error) {
    note.textContent = !link
      ? ""
      : link.mode === "front"
        ? `parallel to #${link.parent}, on its axis, and follows it; negative gap: behind`
        : link.mode === "follow"
          ? `moves and turns with #${link.parent}; drag it or edit x, y to set where it sits`
          : `turns to face #${link.parent}; its tilt is set by the link`;
  }
}

/** Write the card's link fields back. */
export function commitLink(scene: Canvas, id: number, root: HTMLElement) {
  const sel = select(root, "linkParent");
  if (!sel || q(root, ".c3-link")?.style.display === "none") return;
  const link = JSON.parse(scene.linkJSON(id)) as Link | null;
  const parent = sel.value === "" ? null : Number(sel.value);
  // A new link starts in front of its parent: the placement a baffle, a
  // filter or a detector on an element usually wants.
  const mode = (link ? select(root, "linkMode")?.value ?? "front" : "front") as Link["mode"];
  const gapField = Number(input(root, "linkGap")?.value);
  const gap = link?.mode === "front" && Number.isFinite(gapField) ? gapField : link?.gap ?? 20;
  const note = q(root, ".c3-linkNote");
  if (note) delete note.dataset.error;
  if (parent == null) {
    if (link) scene.unlinkObject(id);
    return;
  }
  if (link && link.parent === parent && link.mode === mode && (mode !== "front" || link.gap === gap)) return;
  const why = scene.linkObject(id, parent, mode, gap);
  if (why && note) {
    note.textContent = `refused: ${why}`;
    note.dataset.error = "1";
  }
}

// ---------------------------------------------------------------------------
// Position and tilt
// ---------------------------------------------------------------------------

/** Has the field moved off the value the card shows for `actual`? A card
 *  commits every field on any edit, and writing an untouched x, y or tilt
 *  back would flatten a 3D orientation into the view plane, or move an
 *  element a link just placed. */
export function fieldChanged(root: HTMLElement, name: string, actual: number): boolean {
  const el = input(root, name);
  if (!el) return false;
  const v = Number(el.value);
  return Number.isFinite(v) && Math.abs(v - Number(actual.toFixed(4))) > 1e-9;
}
