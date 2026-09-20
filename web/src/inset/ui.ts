// Shared parts of the component view's editors: the editor contract, drag
// handles, and the small form controls under the drawing.

import type { Canvas } from "../../pkg/oc_wasm.js";

export type HandleShape = "square" | "circle" | "ring" | "diamond" | "triangle";

/** A point on the drawing the pointer can grab. Positions are CSS px. */
export type Handle = {
  key: string;
  x: number;
  y: number;
  shape: HandleShape;
  color: string;
  /** Shown under the drawing while the pointer is over the handle. */
  title: string;
  cursor: string;
};

export type DragPhase = "start" | "move" | "end";

export type EditorContext = {
  scene: Canvas;
  id: number;
  /** The editor changed the scene: `settled` false while a handle is held (a
   *  fast picture), true when the edit is final (an exact trace). */
  edited: (settled: boolean) => void;
  /** A message under the drawing; `warn` marks a refused edit. Empty text
   *  restores the hint. */
  note: (text: string, warn?: boolean) => void;
  /** Redraw the drawing. */
  redraw: () => void;
  bar: HTMLElement;
  fields: HTMLElement;
  read: HTMLElement;
  /** A detector's f64 readout (`detectorResultJSON`), or null while the
   *  reference is being traced. */
  readout: (id: number) => string | null;
};

export interface Editor {
  /** What the note line says when nothing else is said. */
  readonly hint: string;
  /** Re-read the object from the scene and update the controls. */
  refresh(): void;
  draw(ctx: CanvasRenderingContext2D, w: number, h: number): void;
  handles(): Handle[];
  drag(h: Handle, x: number, y: number, phase: DragPhase): void;
}

export function drawHandle(ctx: CanvasRenderingContext2D, h: Handle, hot: boolean) {
  const r = hot ? 6.5 : 5;
  ctx.save();
  ctx.lineWidth = hot ? 2 : 1.5;
  ctx.strokeStyle = hot ? "#ffffff" : h.color;
  ctx.fillStyle = h.color;
  ctx.beginPath();
  switch (h.shape) {
    case "square":
      ctx.rect(h.x - r, h.y - r, 2 * r, 2 * r);
      break;
    case "diamond":
      ctx.moveTo(h.x, h.y - r - 1);
      ctx.lineTo(h.x + r + 1, h.y);
      ctx.lineTo(h.x, h.y + r + 1);
      ctx.lineTo(h.x - r - 1, h.y);
      ctx.closePath();
      break;
    case "triangle":
      ctx.moveTo(h.x, h.y - r - 1);
      ctx.lineTo(h.x + r + 1, h.y + r);
      ctx.lineTo(h.x - r - 1, h.y + r);
      ctx.closePath();
      break;
    default:
      ctx.arc(h.x, h.y, r, 0, Math.PI * 2);
  }
  if (h.shape === "ring") {
    ctx.fillStyle = "rgba(4,6,10,0.85)";
    ctx.fill();
    ctx.strokeStyle = hot ? "#ffffff" : h.color;
    ctx.stroke();
  } else {
    ctx.fill();
    if (hot) ctx.stroke();
  }
  ctx.restore();
}

/** The handle nearest to (x, y) within `reach` px, preferring later ones
 *  (drawn on top). */
export function hitHandle(handles: Handle[], x: number, y: number, reach = 9): Handle | null {
  let best: Handle | null = null;
  let bestD = reach;
  for (const h of handles) {
    const d = Math.hypot(h.x - x, h.y - y);
    if (d <= bestD) {
      bestD = d;
      best = h;
    }
  }
  return best;
}

/** A tidy step near `raw` (1, 2 or 5 times a power of ten). */
export function niceStep(raw: number): number {
  if (!(raw > 0) || !Number.isFinite(raw)) return 1;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

/** A number for display: fixed for ordinary magnitudes, exponent otherwise. */
export function fmt(v: number | null | undefined, digits = 3): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (!Number.isFinite(v)) return "∞";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e5) return String(Number(v.toFixed(digits)));
  return v.toExponential(Math.max(1, digits - 1));
}

export type NumberField = {
  input: HTMLInputElement;
  /** Show a value, unless the user is typing in the field. */
  set: (v: number | null) => void;
};

/** A labelled number input that commits as it is typed (when the text parses)
 *  and scrubs with the mouse wheel. `∞`, `inf` or an empty field read as
 *  infinity when `allowInfinite` is set. */
export function numberField(
  parent: HTMLElement,
  label: string,
  unit: string,
  opts: {
    commit: (v: number) => void;
    step?: number;
    digits?: number;
    exponent?: boolean;
    allowInfinite?: boolean;
    title?: string;
    name?: string;
  },
): NumberField {
  const row = document.createElement("label");
  row.className = "cv-field";
  if (opts.title) row.title = opts.title;
  const l = document.createElement("span");
  l.textContent = label;
  const input = document.createElement("input");
  input.type = "text";
  input.inputMode = "decimal";
  input.spellcheck = false;
  if (opts.name) input.dataset.name = opts.name;
  const u = document.createElement("span");
  u.className = "u";
  u.textContent = unit;
  row.append(l, input, u);
  parent.appendChild(row);

  const parse = (text: string): number | null => {
    const t = text.trim().toLowerCase();
    if (opts.allowInfinite && (t === "" || t === "∞" || t === "inf" || t === "infinity")) return Infinity;
    if (t === "") return null;
    const v = Number(t);
    return Number.isFinite(v) ? v : null;
  };
  const show = (v: number | null) => {
    if (v == null) return "";
    if (!Number.isFinite(v)) return "∞";
    if (opts.exponent) return v === 0 ? "0" : v.toExponential(3);
    return String(Number(v.toFixed(opts.digits ?? 4)));
  };
  input.addEventListener("input", () => {
    const v = parse(input.value);
    if (v != null) opts.commit(v);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") input.blur();
  });
  input.addEventListener("wheel", (e) => {
    e.preventDefault();
    const cur = parse(input.value);
    if (cur == null || !Number.isFinite(cur)) return;
    const step = (opts.step ?? 0.5) * (e.shiftKey ? 10 : 1) * (e.deltaY < 0 ? 1 : -1);
    const next = opts.exponent ? (cur === 0 ? step * 1e-6 : cur * (1 + Math.sign(step) * 0.1)) : cur + step;
    input.value = show(Number(next.toPrecision(12)));
    opts.commit(next);
  }, { passive: false });

  return {
    input,
    set: (v) => {
      if (document.activeElement === input) return;
      input.value = show(v);
    },
  };
}

export function button(parent: HTMLElement, text: string, title: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement("button");
  b.className = "m4btn";
  b.type = "button";
  b.textContent = text;
  b.title = title;
  b.addEventListener("click", onClick);
  parent.appendChild(b);
  return b;
}

export function label(parent: HTMLElement, text: string) {
  const s = document.createElement("span");
  s.className = "cv-label";
  s.textContent = text;
  parent.appendChild(s);
}

/** A row of buttons of which one is active. */
export function segmented<T extends string>(
  parent: HTMLElement,
  options: { value: T; text: string; title: string }[],
  onPick: (v: T) => void,
): { set: (v: T | null) => void } {
  const buttons = options.map((o) => {
    const b = button(parent, o.text, o.title, () => onPick(o.value));
    b.dataset.value = o.value;
    return b;
  });
  return {
    set: (v) => {
      for (const b of buttons) b.classList.toggle("active", b.dataset.value === v);
    },
  };
}
