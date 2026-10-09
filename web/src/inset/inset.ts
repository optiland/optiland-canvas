// The inset window: a tab in the top bar that opens a small window over the
// canvas, in place with a click or wherever the tab is dragged and dropped.
//
// The window moves by its header and resizes from its corner. Its ✕ closes it,
// and so does dragging the header back onto the top bar. Where it sits and how
// big it is are remembered in this browser. What the window shows belongs to
// its content (component-view.ts); this file only handles the frame.

import { bringToFront } from "../cards";

export type InsetContent = {
  /** The window opened or closed. */
  setOpen: (open: boolean) => void;
  /** The window's size changed. */
  resized: () => void;
};

export type Inset = {
  /** The element the content fills. */
  body: HTMLElement;
  /** Text after the title (what the window is showing). */
  setSubtitle: (text: string) => void;
  attach: (content: InsetContent) => void;
  isOpen: () => boolean;
  /** Open with the top-left corner at (x, y), or where it was last. */
  open: (at?: { x: number; y: number }) => void;
  close: () => void;
  /** The window's width, and a new one: kept on screen, moved left if it
   *  would run off the right edge, and remembered like a drag would be. */
  width: () => number;
  setWidth: (w: number) => void;
};

type Placement = { open: boolean; x: number; y: number; w: number; h: number };

const KEY = "optiland-canvas:inset";
const MIN_W = 300;
const MIN_H = 260;
/** Where the pointer holds the window while it is carried from the tab. */
const GRAB = { x: 28, y: 12 };
/** A drag shorter than this is a click. */
const CLICK_SLOP = 6;

function loadPlacement(): Placement {
  // Left of the right-hand panels, under the top bar.
  const out: Placement = {
    open: false,
    x: Math.max(0, window.innerWidth - 700),
    y: 100,
    w: 420,
    h: 540,
  };
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? "{}") ?? {};
    const num = (v: unknown) => typeof v === "number" && Number.isFinite(v);
    if (typeof saved.open === "boolean") out.open = saved.open;
    for (const k of ["x", "y", "w", "h"] as const) if (num(saved[k])) out[k] = saved[k];
  } catch {
    // Unreadable or refused storage: the defaults stand.
  }
  return out;
}


export function initInset(opts: {
  tab: HTMLElement;
  /** The top bar; dropping the window's header on it closes the window. */
  bar: HTMLElement;
  frame: HTMLElement;
  ghost: HTMLElement;
}): Inset {
  const { tab, bar, frame, ghost } = opts;
  const head = frame.querySelector(".inset-head") as HTMLElement;
  const body = frame.querySelector(".inset-body") as HTMLElement;
  const sub = frame.querySelector(".inset-sub") as HTMLElement;
  const closeBtn = frame.querySelector(".inset-close") as HTMLButtonElement;
  const grip = frame.querySelector(".inset-resize") as HTMLElement;
  // The window joins the same floating stack the inspector cards use, so
  // touching it brings it forward. Without this it sat under every card and a
  // control the card happened to cover could not be reached at all.
  frame.addEventListener("pointerdown", () => bringToFront(frame), true);
  bringToFront(frame);
  let content: InsetContent | null = null;
  const place = loadPlacement();

  function save() {
    try {
      localStorage.setItem(KEY, JSON.stringify(place));
    } catch {
      // Not saved; the window still works for this page.
    }
  }

  /** Keep the window on screen and at least its minimum size. */
  function clamp() {
    const W = window.innerWidth, H = window.innerHeight;
    place.w = Math.round(Math.min(Math.max(MIN_W, place.w), W));
    place.h = Math.round(Math.min(Math.max(MIN_H, place.h), H));
    place.x = Math.round(Math.min(Math.max(0, place.x), W - place.w));
    place.y = Math.round(Math.min(Math.max(0, place.y), H - place.h));
  }

  function apply() {
    clamp();
    frame.style.left = `${place.x}px`;
    frame.style.top = `${place.y}px`;
    frame.style.width = `${place.w}px`;
    frame.style.height = `${place.h}px`;
    frame.hidden = !place.open;
    tab.classList.toggle("active", place.open);
  }

  function open(at?: { x: number; y: number }) {
    if (at) {
      place.x = at.x;
      place.y = at.y;
    }
    const was = place.open;
    place.open = true;
    apply();
    save();
    if (!was) content?.setOpen(true);
    content?.resized();
  }

  function close() {
    if (!place.open) return;
    place.open = false;
    apply();
    save();
    content?.setOpen(false);
  }

  const inside = (el: HTMLElement, x: number, y: number, pad = 0) => {
    const r = el.getBoundingClientRect();
    return x >= r.left - pad && x <= r.right + pad && y >= r.top - pad && y <= r.bottom + pad;
  };

  // --- the tab: click to toggle, drag to carry the window out ---------------
  let press: { x: number; y: number; dragging: boolean } | null = null;
  function ghostAt(x: number, y: number) {
    const W = window.innerWidth, H = window.innerHeight;
    const w = Math.min(place.w, W), h = Math.min(place.h, H);
    const left = Math.min(Math.max(0, x - GRAB.x), W - w);
    const top = Math.min(Math.max(0, y - GRAB.y), H - h);
    Object.assign(ghost.style, { left: `${left}px`, top: `${top}px`, width: `${w}px`, height: `${h}px` });
    return { x: left, y: top };
  }
  tab.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    press = { x: e.clientX, y: e.clientY, dragging: false };
    tab.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  tab.addEventListener("pointermove", (e) => {
    if (!press) return;
    if (!press.dragging && Math.hypot(e.clientX - press.x, e.clientY - press.y) < CLICK_SLOP) return;
    press.dragging = true;
    tab.style.cursor = "grabbing";
    ghost.hidden = false;
    ghostAt(e.clientX, e.clientY);
    // Over the bar, the drop does nothing; say so by dimming the ghost.
    ghost.style.opacity = inside(bar, e.clientX, e.clientY) ? "0.35" : "1";
  });
  const endPress = (e: PointerEvent, cancelled: boolean) => {
    if (!press) return;
    const p = press;
    press = null;
    try { tab.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    tab.style.cursor = "";
    ghost.hidden = true;
    if (cancelled) return;
    if (!p.dragging) {
      if (place.open) close();
      else open();
      return;
    }
    if (inside(bar, e.clientX, e.clientY)) return;
    open(ghostAt(e.clientX, e.clientY));
  };
  tab.addEventListener("pointerup", (e) => endPress(e, false));
  tab.addEventListener("pointercancel", (e) => endPress(e, true));
  // A keyboard press (no pointer) toggles too.
  tab.addEventListener("click", (e) => {
    if (e.detail !== 0) return;
    if (place.open) close();
    else open();
  });

  // --- the header: move, or drop on the bar to close ------------------------
  let move: { dx: number; dy: number } | null = null;
  head.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest("button")) return;
    const r = frame.getBoundingClientRect();
    move = { dx: e.clientX - r.left, dy: e.clientY - r.top };
    head.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  head.addEventListener("pointermove", (e) => {
    if (!move) return;
    place.x = e.clientX - move.dx;
    place.y = e.clientY - move.dy;
    apply();
    tab.classList.toggle("dock-target", inside(bar, e.clientX, e.clientY));
  });
  const endMove = (e: PointerEvent) => {
    if (!move) return;
    move = null;
    try { head.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    tab.classList.remove("dock-target");
    if (inside(bar, e.clientX, e.clientY)) close();
    else save();
  };
  head.addEventListener("pointerup", endMove);
  head.addEventListener("pointercancel", endMove);
  closeBtn.addEventListener("click", () => close());

  // --- the corner: resize ---------------------------------------------------
  let size: { x: number; y: number; w: number; h: number } | null = null;
  grip.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    size = { x: e.clientX, y: e.clientY, w: place.w, h: place.h };
    grip.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  grip.addEventListener("pointermove", (e) => {
    if (!size) return;
    // Grow toward the screen edge, never past it.
    place.w = Math.min(size.w + e.clientX - size.x, window.innerWidth - place.x);
    place.h = Math.min(size.h + e.clientY - size.y, window.innerHeight - place.y);
    apply();
    content?.resized();
  });
  const endSize = (e: PointerEvent) => {
    if (!size) return;
    size = null;
    try { grip.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    save();
  };
  grip.addEventListener("pointerup", endSize);
  grip.addEventListener("pointercancel", endSize);

  // Keys typed into the window stay out of the canvas shortcuts.
  frame.addEventListener("keydown", (e) => e.stopPropagation());

  window.addEventListener("resize", () => {
    apply();
    if (place.open) content?.resized();
  });
  apply();

  return {
    body,
    setSubtitle: (text) => { sub.textContent = text; },
    attach: (c) => {
      content = c;
      if (place.open) {
        c.setOpen(true);
        c.resized();
      }
    },
    isOpen: () => place.open,
    open,
    close,
    width: () => place.w,
    setWidth: (w) => {
      place.w = w;
      apply();
      save();
      if (place.open) content?.resized();
    },
  };
}
