// Inspector cards — the view-agnostic floating pop-up engine (§5.2).
//
// Double-click an element and a compact card pops up next to it; many can be
// open at once, each follows its element as you drag/pan/zoom (unless its
// location is pinned), and a leader line ties each card back to its element.
//
// This module owns only the *generic* window behaviour: create/destroy, the
// header drag, pin-location, z-order stacking, screen positioning, and the
// leader lines. What a card *shows* and *writes* is element-specific and lives
// in a `CardSource` adapter — so the same engine drives the legacy 2D scene
// today and the canonical `Scene3` slice/3D views next, with no duplication.

export type V2 = { x: number; y: number };

/** Handle the manager hands to a source's field listeners. */
export interface CardApi {
  /** Re-read this card's element-specific fields from the scene. */
  rerender: (id: number) => void;
}

/** The element-specific half of a card: where its element is, and what it shows. */
export interface CardSource {
  /** The `<template>` cloned for each card (its `.card` root and fields). */
  template: HTMLTemplateElement;
  /** World position the card anchors to, or null if the element vanished. */
  anchorWorld: (id: number) => V2 | null;
  /** World → screen (current pan/zoom). */
  toScreen: (p: V2) => V2;
  /** Redraw the main canvas (so leader lines track a card being dragged). */
  render2d: () => void;
  /**
   * Fill a card's element-specific fields from the scene. Return false if the
   * element no longer exists, so the manager closes the orphaned card.
   * (The manager renders the generic chrome — title-independent pin glyph — itself.)
   */
  render: (id: number, root: HTMLElement) => boolean;
  /** Wire the element-specific field listeners once, on card creation. */
  wire: (id: number, root: HTMLElement, api: CardApi) => void;
}

// A card tracks its element by an `offset` from the element's screen position
// (so it follows as you drag the element), unless its location is `pinned`, in
// which case it stays at an absolute screen position `abs`.
type Card = {
  id: number;
  root: HTMLElement;
  offset: { dx: number; dy: number };
  pinned: boolean;
  abs: { left: number; top: number } | null;
};

export interface CardManager {
  open: (id: number) => void;
  close: (id: number) => void;
  closeAll: () => void;
  /** Render one open card (no-op if it isn't open). */
  render: (id: number) => void;
  renderAll: () => void;
  /** Position all open cards (call on pan/zoom/resize). */
  positionAll: () => void;
  /** Render + reposition one card (used while its element is dragged). */
  refresh: (id: number) => void;
  has: (id: number) => boolean;
  ids: () => number[];
  /** After deleting element `deleted`, shift higher card ids down by one. */
  shiftIdsAfter: (deleted: number) => void;
  /** Draw a leader line from each open card to its element (call in the overlay). */
  drawLeaders: (ctx: CanvasRenderingContext2D, selected: number) => void;
}

/** The top of the floating stack, shared by every window that can cover
 *  another: the inspector cards and the component inset.
 *
 *  It has to be shared. Cards used to climb from 40 without bound while the
 *  inset sat at a fixed 30, so a card that opened over the inset buried it
 *  permanently and a control under the card could not be clicked at all. One
 *  counter makes the rule the obvious one: whatever you touched last is on
 *  top. */
let zTop = 40;

/** Bring a floating element to the front of that stack. */
export function bringToFront(el: HTMLElement) {
  el.style.zIndex = String(++zTop);
}

export function createCardManager(source: CardSource): CardManager {
  const cards: Card[] = [];
  const cardOf = (id: number): Card | undefined => cards.find((c) => c.id === id);

  function open(id: number) {
    const existing = cardOf(id);
    if (existing) {
      renderOne(existing);
      positionOne(existing);
      bringToFront(existing.root);
      return;
    }
    const frag = source.template.content.cloneNode(true) as DocumentFragment;
    const root = frag.querySelector(".card") as HTMLElement;
    document.body.appendChild(root);
    // Stagger new cards diagonally so those anchored to nearby elements don't
    // land exactly on top of each other (drag / pin them from there).
    const n = cards.length % 6;
    const card: Card = { id, root, offset: { dx: 22 + n * 16, dy: -24 + n * 16 }, pinned: false, abs: null };
    cards.push(card);
    wireGeneric(card);
    source.wire(id, root, { rerender: (i) => render(i) });
    renderOne(card);
    positionOne(card);
    bringToFront(root);
  }

  function close(id: number) {
    const i = cards.findIndex((c) => c.id === id);
    if (i >= 0) { cards[i].root.remove(); cards.splice(i, 1); }
  }
  function closeAll() { for (const c of cards) c.root.remove(); cards.length = 0; }

  const render = (id: number) => { const c = cardOf(id); if (c) renderOne(c); };
  function renderAll() { for (const c of [...cards]) renderOne(c); }
  function positionAll() { for (const c of cards) positionOne(c); }
  function refresh(id: number) { const c = cardOf(id); if (c) { renderOne(c); positionOne(c); } }

  // Record where a card sits: its absolute position (if pinned), or its offset
  // from the element (so it keeps following after being dragged).
  function anchorCard(card: Card, left: number, top: number) {
    if (card.pinned) { card.abs = { left, top }; return; }
    const w = source.anchorWorld(card.id);
    if (w) { const s = source.toScreen(w); card.offset = { dx: left - s.x, dy: top - s.y }; }
  }

  function wireGeneric(card: Card) {
    const q = <T extends HTMLElement>(sel: string) => card.root.querySelector(sel) as T;
    card.root.addEventListener("pointerdown", () => bringToFront(card.root));
    q<HTMLButtonElement>(".c-close").addEventListener("click", () => close(card.id));

    // pin the card's *location* (park it; it stops following the element).
    q<HTMLButtonElement>(".c-pinpos").addEventListener("click", () => {
      const r = card.root.getBoundingClientRect();
      card.pinned = !card.pinned;
      anchorCard(card, r.left, r.top);
      renderOne(card);
    });

    // drag the card by its header (so overlapping cards can be pulled apart).
    const header = q<HTMLElement>("h3");
    let grab: { x: number; y: number } | null = null;
    header.addEventListener("pointerdown", (e) => {
      if ((e.target as HTMLElement).closest("button")) return; // buttons aren't drag
      const r = card.root.getBoundingClientRect();
      grab = { x: e.clientX - r.left, y: e.clientY - r.top };
      bringToFront(card.root);
      header.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    header.addEventListener("pointermove", (e) => {
      if (!grab) return;
      const pw = card.root.offsetWidth, ph = card.root.offsetHeight;
      const left = Math.min(Math.max(0, e.clientX - grab.x), window.innerWidth - pw);
      const top = Math.min(Math.max(0, e.clientY - grab.y), window.innerHeight - ph);
      card.root.style.left = `${left}px`;
      card.root.style.top = `${top}px`;
      anchorCard(card, left, top);
      source.render2d(); // redraw the leader line to track the card
    });
    const endCardDrag = (e: PointerEvent) => {
      if (!grab) return;
      grab = null;
      try { header.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    };
    header.addEventListener("pointerup", endCardDrag);
    header.addEventListener("pointercancel", endCardDrag);
  }

  // Element-specific fields come from the source; the pin-location glyph is
  // generic (its state lives on the Card record), so the manager renders it.
  function renderOne(card: Card) {
    if (!source.render(card.id, card.root)) { close(card.id); return; }
    const pinBtn = card.root.querySelector(".c-pinpos") as HTMLElement;
    pinBtn.classList.toggle("pinned", card.pinned);
    pinBtn.textContent = card.pinned ? "📌" : "📍";
  }

  // Place a card: at its pinned absolute position, or at its element + offset.
  function positionOne(card: Card) {
    const pw = card.root.offsetWidth || 176, ph = card.root.offsetHeight || 220;
    let left: number, top: number;
    if (card.pinned && card.abs) {
      ({ left, top } = card.abs);
    } else {
      const w = source.anchorWorld(card.id);
      if (!w) return;
      const s = source.toScreen(w);
      left = s.x + card.offset.dx;
      top = s.y + card.offset.dy;
    }
    left = Math.min(Math.max(0, left), window.innerWidth - pw);
    top = Math.min(Math.max(0, top), window.innerHeight - ph);
    card.root.style.left = `${left}px`;
    card.root.style.top = `${top}px`;
  }

  // A thin dashed connector from each open card to its element, so you can tell
  // which card belongs to which when they're pulled apart.
  function drawLeaders(ctx: CanvasRenderingContext2D, selected: number) {
    for (const card of cards) {
      const w = source.anchorWorld(card.id);
      if (!w) continue;
      const s = source.toScreen(w);
      // Anchor on the card's *border* (never the element itself) so the line is
      // always drawn — even when the card overlaps the element near the axis.
      const a = anchorOnCard(s.x, s.y, card.root.getBoundingClientRect());
      ctx.strokeStyle = card.id === selected ? "rgba(127,214,255,0.6)" : "rgba(127,214,255,0.32)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      line(ctx, s.x, s.y, a.x, a.y);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(127,214,255,0.8)";
      ctx.beginPath(); ctx.arc(s.x, s.y, 2.5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(a.x, a.y, 2, 0, Math.PI * 2); ctx.fill();
    }
  }

  return {
    open, close, closeAll, render, renderAll, positionAll, refresh,
    has: (id) => cards.some((c) => c.id === id),
    ids: () => cards.map((c) => c.id),
    shiftIdsAfter: (deleted) => { for (const c of cards) if (c.id > deleted) c.id -= 1; },
    drawLeaders,
  };
}

function line(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
}

// The point on rectangle `r`'s border nearest to `(px, py)`. For a point outside
// the box this is the clamp; for a point inside it projects to the closest edge
// (so a card overlapping its element still shows a connector).
function anchorOnCard(px: number, py: number, r: DOMRect): { x: number; y: number } {
  const cx = Math.min(Math.max(px, r.left), r.right);
  const cy = Math.min(Math.max(py, r.top), r.bottom);
  if (cx !== px || cy !== py) return { x: cx, y: cy };
  const dl = px - r.left, dr = r.right - px, dt = py - r.top, db = r.bottom - py;
  const m = Math.min(dl, dr, dt, db);
  if (m === dl) return { x: r.left, y: py };
  if (m === dr) return { x: r.right, y: py };
  if (m === dt) return { x: px, y: r.top };
  return { x: px, y: r.bottom };
}
