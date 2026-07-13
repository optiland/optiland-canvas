// Optiland Canvas — M3 web module ("it's precise").
//
// Everything the M3 milestone adds to the shell, kept in its own module so the
// M2 rendering pipeline in main.ts stays untouched (main just calls initM3 and
// invokes the returned overlay each frame). Scene state still lives in Rust
// (§2); this file is UI only.
//
//   • Draw tool          — freehand stroke → fit ladder → commit (§5.3)
//   • Complexity slider  — slide the fit up/down the ladder, rays change live
//   • Inspector cards    — double-click an element for a compact pop-up card;
//                          many can be open at once, each follows its element,
//                          every parameter is a numeric twin of a handle (§5.2)
//   • Grid + snapping     — a spaced grid; drag snaps to it (Shift → axis /
//                          vertex snaps instead) (§5.2)
//   • Clipboard           — Del delete · Ctrl+C/V copy/paste · Ctrl+D duplicate
//   • Pin tool            — lock a parameter, or a whole element (ghosted) (§5.1)

import type { Canvas } from "../pkg/oc_wasm.js";

type V2 = { x: number; y: number };
type Tool = "select" | "draw" | "pin";
type Rung = { level: number; name: string; dof: number; residual_um: number; rms_um: number; curve: number[] };
type Fit = { chosen: number; tol_um: number; rungs: Rung[] };
type Overlay = { id: number; x: number; y: number; rot_deg: number; locked: boolean; kind: string };
type Snap = { x: number; y: number; snaps: string[]; guideX: number | null; guideY: number | null };

export interface AppCtx {
  scene: Canvas;
  view: HTMLCanvasElement;
  toScreen: (p: V2) => V2;
  toWorld: (px: number, py: number) => V2;
  zoomAt: (clientX: number, clientY: number, factor: number) => void;
  panBy: (dxScreen: number, dyScreen: number) => void;
  resetView: () => void;
  render: () => void;   // main.ts renderView
  refresh: () => void;  // main.ts refreshReference (f64 detector numbers, on idle)
  demoLensId: number;
}

const DRAW_GLASS = { nd: 1.5168, vd: 64.17 }; // freehand optics default to N-BK7 crown
const DRAW_THICKNESS = 4; // mm, extrusion depth for a drawn surface
const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
// The singlet fields, in card order (front = 1, rear = 2).
const SINGLET_FIELDS = ["radius1", "radius2", "thickness", "half_ap1", "half_ap2", "conic_k1", "conic_k2"];
// Scrub step per parameter.
const STEP: Record<string, number> = {
  x: 0.25, y: 0.25, rot: 1,
  radius1: 0.5, radius2: 0.5, thickness: 0.2,
  half_ap1: 0.25, half_ap2: 0.25, conic_k1: 0.05, conic_k2: 0.05,
};

export function initM3(app: AppCtx): { overlay: (ctx: CanvasRenderingContext2D) => void } {
  const { scene, view } = app;
  let tool: Tool = "select";
  let selected = app.demoLensId; // -1 = nothing selected
  let snapOn = false;
  let clipboard: string | null = null;
  let pasteN = 0;

  // ---- grid ------------------------------------------------------------
  const gridBox = el<HTMLInputElement>("gridBox");
  const gridSpacing = el<HTMLInputElement>("gridSpacing");
  const gridOn = () => gridBox.checked;
  const gridStep = () => Math.max(0.1, Number(gridSpacing.value) || 5);
  const snapToGrid = (v: number) => Math.round(v / gridStep()) * gridStep();
  gridBox.addEventListener("change", () => app.render());
  gridSpacing.addEventListener("input", () => app.render());

  // ---- shared geometry helpers -----------------------------------------
  const outlineOf = (id: number): number[] => Array.from(scene.elementOutline(id));
  const overlays = (): Overlay[] => JSON.parse(scene.elementsOverlayJson());
  const overlayOf = (id: number): Overlay | undefined => overlays().find((o) => o.id === id);

  function pointInPoly(wx: number, wy: number, poly: number[]): boolean {
    let inside = false;
    for (let i = 0, j = poly.length - 2; i < poly.length; j = i, i += 2) {
      const xi = poly[i], yi = poly[i + 1], xj = poly[j], yj = poly[j + 1];
      if ((yi > wy) !== (yj > wy) && wx < ((xj - xi) * (wy - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  // Nearest element whose body contains the world point, else the nearest centre.
  function pick(wx: number, wy: number): number {
    const ovs = overlays();
    for (const o of ovs) {
      const poly = outlineOf(o.id);
      if (poly.length >= 6 && pointInPoly(wx, wy, poly)) return o.id;
    }
    let best = -1, bestD = 3 * 3; // within 3 mm of a centre
    for (const o of ovs) {
      const d = (o.x - wx) ** 2 + (o.y - wy) ** 2;
      if (d < bestD) { bestD = d; best = o.id; }
    }
    return best;
  }

  const rotHandleWorld = (o: Overlay): V2 => {
    const r = 16; // mm from centre
    return { x: o.x + r * Math.cos((o.rot_deg * Math.PI) / 180), y: o.y + r * Math.sin((o.rot_deg * Math.PI) / 180) };
  };
  const nearRotHandle = (o: Overlay, p: V2) => {
    const rh = rotHandleWorld(o);
    return (rh.x - p.x) ** 2 + (rh.y - p.y) ** 2 < 2.5 * 2.5; // within 2.5 mm
  };

  // ================= tool palette =======================================
  const toolBtns: Record<Tool, HTMLButtonElement> = {
    select: el("tool-select"),
    draw: el("tool-draw"),
    pin: el("tool-pin"),
  };
  function setTool(t: Tool) {
    if (t !== "draw") cancelDraw();
    tool = t;
    for (const k of Object.keys(toolBtns) as Tool[]) toolBtns[k].classList.toggle("active", k === t);
    view.style.cursor = t === "draw" ? "crosshair" : t === "pin" ? "not-allowed" : "grab";
    el("m3hint").textContent =
      t === "draw" ? "drag to sketch a surface · release to fit"
      : t === "pin" ? "click an element to lock it (ghosted)"
      : "drag · dbl-click inspect · scroll=zoom · drag bg=pan · 0 resets";
    app.render();
  }
  for (const k of Object.keys(toolBtns) as Tool[]) toolBtns[k].onclick = () => setTool(k);

  // ================= keyboard: tools, clipboard, delete, nudge ==========
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement) return; // let fields have their keys
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === "Shift") { snapOn = true; return; }
    if (mod && (e.key === "c" || e.key === "C")) { copySelection(); e.preventDefault(); return; }
    if (mod && (e.key === "v" || e.key === "V")) { pasteClipboard(); e.preventDefault(); return; }
    if (mod && (e.key === "x" || e.key === "X")) { copySelection(); deleteSelection(); e.preventDefault(); return; }
    if (mod && (e.key === "d" || e.key === "D")) { copySelection(); pasteClipboard(); e.preventDefault(); return; }
    if (e.key === "Delete" || e.key === "Backspace") { deleteSelection(); e.preventDefault(); return; }
    if (e.key === "Escape") { closeAllCards(); deselect(); return; }
    if (e.key.startsWith("Arrow")) { nudge(e.key); e.preventDefault(); return; }
    if (mod) return; // don't hijack other shortcuts
    if (e.key === "v") setTool("select");
    else if (e.key === "d") setTool("draw");
    else if (e.key === "p") setTool("pin");
    else if (e.key === "0") { app.resetView(); positionAllCards(); app.render(); }
    else if (e.key === "+" || e.key === "=") { app.zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1.12); positionAllCards(); app.render(); }
    else if (e.key === "-" || e.key === "_") { app.zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1 / 1.12); positionAllCards(); app.render(); }
  });
  window.addEventListener("keyup", (e) => { if (e.key === "Shift") snapOn = false; });
  window.addEventListener("resize", () => positionAllCards());

  function nudge(arrow: string) {
    if (selected < 0) return;
    const o = overlayOf(selected);
    if (!o) return;
    const s = gridOn() ? gridStep() : 1;
    const dx = arrow === "ArrowLeft" ? -s : arrow === "ArrowRight" ? s : 0;
    const dy = arrow === "ArrowUp" ? s : arrow === "ArrowDown" ? -s : 0;
    scene.setElementPos(selected, o.x + dx, o.y + dy);
    afterSceneEdit();
  }

  // ================= element lifecycle (delete / copy / paste) ==========
  function copySelection() { if (selected >= 0) clipboard = scene.cloneElementJson(selected); }
  function deleteSelection() {
    if (selected < 0) return;
    const del = selected;
    if (!scene.deleteElement(del)) return;
    // ids above `del` shift down by one; keep the open cards in step.
    closeCard(del);
    for (const c of cards) if (c.id > del) c.id -= 1;
    deselect();
    afterSceneEdit();
  }
  function pasteClipboard() {
    if (!clipboard) return;
    pasteN++;
    const off = gridOn() ? gridStep() : 5;
    const id = scene.pasteElement(clipboard, off * pasteN, -off * pasteN);
    if (id >= 0) { selected = id; openCard(id); afterSceneEdit(); }
  }
  function afterSceneEdit() { renderAllCards(); positionAllCards(); app.render(); app.refresh(); }

  // ================= view: zoom + pan ===================================
  let pan: { x: number; y: number } | null = null;
  const startPan = (e: PointerEvent) => { pan = { x: e.clientX, y: e.clientY }; view.setPointerCapture(e.pointerId); };
  view.addEventListener("wheel", (e) => {
    e.preventDefault();
    app.zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.12 : 1 / 1.12); // zoom to cursor
    positionAllCards();
    app.render();
  }, { passive: false });

  // ================= select / move / rotate =============================
  let drag: { id: number; mode: "move" | "rotate"; dx: number; dy: number } | null = null;
  let lastSnap: Snap | null = null;
  let gridMark: V2 | null = null;

  view.addEventListener("pointerdown", (e) => {
    if (e.button === 1) { startPan(e); e.preventDefault(); return; } // middle-drag pans anywhere
    if (tool === "draw") return startDraw(e);
    const p = app.toWorld(e.clientX, e.clientY);
    if (tool === "pin") {
      const id = pick(p.x, p.y);
      if (id >= 0) { scene.setElementLocked(id, !scene.isElementLocked(id)); selected = id; renderCardOf(id); app.render(); }
      return;
    }
    // select tool — the rotation handle lives *outside* the body, so grab it
    // before the body pick (this is why rotation "did nothing" before).
    if (selected >= 0) {
      const o = overlayOf(selected);
      if (o && !o.locked && nearRotHandle(o, p)) {
        drag = { id: selected, mode: "rotate", dx: 0, dy: 0 };
        view.setPointerCapture(e.pointerId);
        return;
      }
    }
    const id = pick(p.x, p.y);
    if (id < 0) { deselect(); startPan(e); return; } // empty: deselect, then drag pans the view
    selected = id;
    app.render();
    if (scene.isElementLocked(id)) return; // locked: select but don't drag
    const o = overlayOf(id)!;
    drag = nearRotHandle(o, p)
      ? { id, mode: "rotate", dx: 0, dy: 0 }
      : { id, mode: "move", dx: o.x - p.x, dy: o.y - p.y };
    view.setPointerCapture(e.pointerId);
  });

  view.addEventListener("pointermove", (e) => {
    if (pan) {
      app.panBy(e.clientX - pan.x, e.clientY - pan.y);
      pan = { x: e.clientX, y: e.clientY };
      positionAllCards();
      app.render();
      return;
    }
    if (tool === "draw") return moveDraw(e);
    if (!drag) return;
    const p = app.toWorld(e.clientX, e.clientY);
    if (drag.mode === "rotate") {
      const o = overlayOf(drag.id)!;
      const deg = (Math.atan2(p.y - o.y, p.x - o.x) * 180) / Math.PI;
      scene.setElementRotDeg(drag.id, deg);
      lastSnap = null; gridMark = null;
    } else {
      let x = p.x + drag.dx, y = p.y + drag.dy;
      lastSnap = null; gridMark = null;
      // Shift → optical snaps (axis / vertex); otherwise the grid, if on.
      if (snapOn) {
        const snap: Snap = JSON.parse(scene.snapPointJson(drag.id, x, y, 1.4));
        x = snap.x; y = snap.y; lastSnap = snap;
      } else if (gridOn()) {
        x = snapToGrid(x); y = snapToGrid(y); gridMark = { x, y };
      }
      scene.setElementPos(drag.id, x, y);
    }
    const c = cardOf(drag.id);
    if (c) { renderCard(c); positionCard(c); } // the card follows its element
    app.render();
  });

  const endDrag = (e: PointerEvent) => {
    if (pan) {
      pan = null;
      try { view.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
      return;
    }
    if (tool === "draw") return endDraw(e);
    if (!drag) return;
    drag = null; lastSnap = null; gridMark = null;
    try { view.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    app.render();
    app.refresh(); // numbers settle on release (§4.1)
  };
  view.addEventListener("pointerup", endDrag);
  view.addEventListener("pointercancel", endDrag);

  // double-click → open an inspector card for that element
  view.addEventListener("dblclick", (e) => {
    if (tool !== "select") return;
    const p = app.toWorld(e.clientX, e.clientY);
    const id = pick(p.x, p.y);
    if (id >= 0) { selected = id; openCard(id); app.render(); }
  });

  // ================= draw tool (freehand → fit, §5.3) ===================
  let stroke: number[] | null = null;         // world coords, flat [x,y,...]
  let drawingNow = false;                      // pointer is down and sketching
  let pendingFit: Fit | null = null;
  let pendingRung = 0;
  const fitPanel = el("fitpanel");
  const fitSlider = el<HTMLInputElement>("fitSlider");
  const fitLabel = el("fitLabel");
  const fitMirror = el<HTMLInputElement>("fitMirror");

  function startDraw(e: PointerEvent) {
    cancelDraw();
    const p = app.toWorld(e.clientX, e.clientY);
    stroke = [p.x, p.y];
    drawingNow = true;
    view.setPointerCapture(e.pointerId);
  }
  function moveDraw(e: PointerEvent) {
    // Only extend the stroke while the button is held. After release the stroke
    // is kept for Commit, so without this guard moving the mouse would keep drawing.
    if (!drawingNow || !stroke) return;
    const p = app.toWorld(e.clientX, e.clientY);
    stroke.push(p.x, p.y);
    app.render();
  }
  function endDraw(e: PointerEvent) {
    if (!drawingNow) return; // ignore stray pointerup/leave events
    drawingNow = false;
    try { view.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    if (!stroke || stroke.length < 8) { stroke = null; app.render(); return; }
    const json = scene.fitStroke(new Float64Array(stroke), 1.0);
    if (json === "null") { stroke = null; app.render(); return; }
    pendingFit = JSON.parse(json);
    pendingRung = pendingFit!.chosen;
    fitSlider.value = String(pendingRung);
    updateFitLabel();
    fitPanel.style.display = "block";
    app.render();
  }
  function updateFitLabel() {
    if (!pendingFit) return;
    const r = pendingFit.rungs[pendingRung];
    const dof = r.dof > 1000 ? "free" : String(r.dof);
    fitLabel.textContent = `${r.name} · ${dof} dof · residual ${r.residual_um.toFixed(2)} µm`;
  }
  fitSlider.oninput = () => { pendingRung = Number(fitSlider.value); updateFitLabel(); app.render(); };
  el("fitCommit").onclick = () => {
    if (!stroke || !pendingFit) return;
    const id = scene.addFittedElement(
      new Float64Array(stroke), 1.0, pendingRung, DRAW_THICKNESS,
      DRAW_GLASS.nd, DRAW_GLASS.vd, fitMirror.checked, // ← draw a mirror instead of glass
    );
    cancelDraw();
    if (id >= 0) { selected = id; setTool("select"); }
    app.render(); app.refresh();
  };
  el("fitCancel").onclick = () => { cancelDraw(); app.render(); };
  function cancelDraw() {
    stroke = null; drawingNow = false; pendingFit = null;
    fitPanel.style.display = "none";
  }

  // ================= inspector cards (multiple, pop-up, §5.2) ===========
  const cardTemplate = el<HTMLTemplateElement>("cardTemplate");
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
  const cards: Card[] = [];
  let zTop = 40;
  const cardOf = (id: number): Card | undefined => cards.find((c) => c.id === id);

  function openCard(id: number) {
    const existing = cardOf(id);
    if (existing) { renderCard(existing); positionCard(existing); existing.root.style.zIndex = String(++zTop); return; }
    const frag = cardTemplate.content.cloneNode(true) as DocumentFragment;
    const root = frag.querySelector(".card") as HTMLElement;
    document.body.appendChild(root);
    // Stagger new cards diagonally so those anchored to nearby elements don't
    // land exactly on top of each other (drag / pin them from there).
    const n = cards.length % 6;
    const card: Card = { id, root, offset: { dx: 22 + n * 16, dy: -24 + n * 16 }, pinned: false, abs: null };
    cards.push(card);
    wireCard(card);
    renderCard(card);
    positionCard(card);
    root.style.zIndex = String(++zTop);
  }
  function closeCard(id: number) {
    const i = cards.findIndex((c) => c.id === id);
    if (i >= 0) { cards[i].root.remove(); cards.splice(i, 1); }
  }
  function closeAllCards() { for (const c of cards) c.root.remove(); cards.length = 0; }
  const renderCardOf = (id: number) => { const c = cardOf(id); if (c) renderCard(c); };
  function renderAllCards() { for (const c of [...cards]) renderCard(c); }
  function positionAllCards() { for (const c of cards) positionCard(c); }

  // Record where a card sits: either its absolute position (if pinned) or its
  // offset from the element (so it keeps following after being dragged).
  function anchorCard(card: Card, left: number, top: number) {
    if (card.pinned) { card.abs = { left, top }; return; }
    const o = overlayOf(card.id);
    if (o) { const s = app.toScreen(o); card.offset = { dx: left - s.x, dy: top - s.y }; }
  }

  function wireCard(card: Card) {
    const q = <T extends HTMLElement>(sel: string) => card.root.querySelector(sel) as T;
    card.root.addEventListener("pointerdown", () => { card.root.style.zIndex = String(++zTop); });
    q<HTMLButtonElement>(".c-close").addEventListener("click", () => closeCard(card.id));

    // pin the card's *location* (park it; it stops following the element).
    q<HTMLButtonElement>(".c-pinpos").addEventListener("click", () => {
      const r = card.root.getBoundingClientRect();
      card.pinned = !card.pinned;
      anchorCard(card, r.left, r.top);
      renderCard(card);
    });

    // drag the card by its header (so overlapping cards can be pulled apart).
    const header = q<HTMLElement>("h3");
    let grab: { x: number; y: number } | null = null;
    header.addEventListener("pointerdown", (e) => {
      if ((e.target as HTMLElement).closest("button")) return; // buttons aren't drag
      const r = card.root.getBoundingClientRect();
      grab = { x: e.clientX - r.left, y: e.clientY - r.top };
      card.root.style.zIndex = String(++zTop);
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
      app.render(); // redraw the leader line to track the card
    });
    const endCardDrag = (e: PointerEvent) => {
      if (!grab) return;
      grab = null;
      try { header.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    };
    header.addEventListener("pointerup", endCardDrag);
    header.addEventListener("pointercancel", endCardDrag);
    q<HTMLInputElement>(".c-lock").addEventListener("change", (e) => {
      scene.setElementLocked(card.id, (e.target as HTMLInputElement).checked);
      renderCard(card); app.render();
    });
    q<HTMLInputElement>(".c-fitslider").addEventListener("input", (e) => {
      const res = scene.setElementFitRung(card.id, Number((e.target as HTMLInputElement).value));
      if (res >= 0) q(".c-fitlabel").textContent = `residual ${res.toFixed(2)} µm`;
      app.render(); app.refresh(); // ← rays visibly change as you slide (§5.3)
    });
    card.root.querySelectorAll<HTMLInputElement>("input[data-name]").forEach((input) => {
      const name = input.dataset.name!;
      const step = STEP[name] ?? 0.25;
      let before = "";
      input.addEventListener("focus", () => { before = input.value; });
      input.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { input.value = before; input.blur(); commitCard(card); }
        e.stopPropagation(); // keep field keys out of the global shortcuts
      });
      input.addEventListener("input", () => commitCard(card));
      input.addEventListener("wheel", (e) => { // hover-scroll scrubs (§5.2)
        e.preventDefault();
        const dir = e.deltaY < 0 ? 1 : -1;
        input.value = String(Number((Number(input.value) + dir * step * (e.shiftKey ? 10 : 1)).toFixed(6)));
        commitCard(card);
      }, { passive: false });
    });
    card.root.querySelectorAll<HTMLButtonElement>("button.pin[data-name]").forEach((btn) => {
      const name = btn.dataset.name!;
      btn.addEventListener("click", () => {
        scene.setParamPinned(card.id, name, !scene.isParamPinned(card.id, name));
        renderCard(card);
      });
    });
  }

  function commitCard(card: Card) {
    if (scene.isElementLocked(card.id)) { renderCard(card); return; }
    const val = (n: string) => Number((card.root.querySelector(`input[data-name="${n}"]`) as HTMLInputElement).value);
    scene.setElementPos(card.id, val("x"), val("y"));
    scene.setElementRotDeg(card.id, val("rot"));
    if (scene.singletParamsJson(card.id) !== "null") {
      scene.setSingletParams(
        card.id, val("radius1"), val("radius2"), val("thickness"),
        val("half_ap1"), val("half_ap2"), val("conic_k1"), val("conic_k2"),
      );
    }
    app.render(); app.refresh();
  }

  function renderCard(card: Card) {
    const t = JSON.parse(scene.elementTransformJson(card.id));
    if (!t) { closeCard(card.id); return; } // element vanished (deleted)
    const q = (sel: string) => card.root.querySelector(sel) as HTMLElement;
    const kind = overlayOf(card.id)?.kind ?? "element";
    q(".c-title").textContent = `#${card.id} · ${kind}`;
    const locked = scene.isElementLocked(card.id);
    (q(".c-lock") as HTMLInputElement).checked = locked;
    const pinBtn = q(".c-pinpos");
    pinBtn.classList.toggle("pinned", card.pinned);
    pinBtn.textContent = card.pinned ? "📌" : "📍";

    const setV = (name: string, v: number) => {
      const inp = card.root.querySelector(`input[data-name="${name}"]`) as HTMLInputElement;
      if (document.activeElement !== inp) inp.value = String(Number(v.toFixed(4)));
      inp.disabled = locked;
    };
    setV("x", t.x); setV("y", t.y); setV("rot", t.rot_deg);

    const sp = JSON.parse(scene.singletParamsJson(card.id));
    q(".c-singlet").style.display = sp ? "block" : "none";
    if (sp) for (const name of SINGLET_FIELDS) setV(name, sp[name]);

    const fit: (Fit & { current: number }) | null = JSON.parse(scene.elementFitJson(card.id));
    q(".c-fit").style.display = fit ? "block" : "none";
    if (fit) {
      (q(".c-fitslider") as HTMLInputElement).value = String(fit.current);
      const r = fit.rungs[fit.current];
      q(".c-fitlabel").textContent = `${r.name} · ${r.residual_um.toFixed(2)} µm`;
    }

    const pinned: string[] = JSON.parse(scene.pinnedParamsJson(card.id));
    card.root.querySelectorAll<HTMLButtonElement>("button.pin[data-name]").forEach((btn) => {
      const on = pinned.includes(btn.dataset.name!);
      btn.classList.toggle("pinned", on);
      btn.textContent = on ? "📌" : "📍";
    });
  }

  // Place a card: at its pinned absolute position, or at its element + offset.
  function positionCard(card: Card) {
    const pw = card.root.offsetWidth || 176, ph = card.root.offsetHeight || 220;
    let left: number, top: number;
    if (card.pinned && card.abs) {
      ({ left, top } = card.abs);
    } else {
      const o = overlayOf(card.id);
      if (!o) return;
      const s = app.toScreen(o);
      left = s.x + card.offset.dx;
      top = s.y + card.offset.dy;
    }
    left = Math.min(Math.max(0, left), window.innerWidth - pw);
    top = Math.min(Math.max(0, top), window.innerHeight - ph);
    card.root.style.left = `${left}px`;
    card.root.style.top = `${top}px`;
  }

  function deselect() { selected = -1; app.render(); }

  // ================= canvas overlay (called from renderView) ============
  function drawGrid(ctx: CanvasRenderingContext2D) {
    if (!gridOn()) return;
    const W = view.clientWidth, H = view.clientHeight, sp = gridStep();
    const tl = app.toWorld(0, 0), br = app.toWorld(W, H);
    const x0 = Math.floor(Math.min(tl.x, br.x) / sp) * sp, x1 = Math.ceil(Math.max(tl.x, br.x) / sp) * sp;
    const y0 = Math.floor(Math.min(tl.y, br.y) / sp) * sp, y1 = Math.ceil(Math.max(tl.y, br.y) / sp) * sp;
    if ((x1 - x0) / sp > 600 || (y1 - y0) / sp > 600) return; // too dense to be useful
    ctx.lineWidth = 1;
    for (let gx = x0; gx <= x1 + 1e-9; gx += sp) {
      const s = app.toScreen({ x: gx, y: 0 });
      ctx.strokeStyle = Math.abs(gx) < 1e-9 ? "rgba(120,150,190,0.22)" : "rgba(120,150,190,0.08)";
      line(ctx, s.x, 0, s.x, H);
    }
    for (let gy = y0; gy <= y1 + 1e-9; gy += sp) {
      const s = app.toScreen({ x: 0, y: gy });
      ctx.strokeStyle = Math.abs(gy) < 1e-9 ? "rgba(120,150,190,0.22)" : "rgba(120,150,190,0.08)";
      line(ctx, 0, s.y, W, s.y);
    }
  }

  function overlay(ctx: CanvasRenderingContext2D) {
    ctx.save();
    ctx.lineJoin = "round";
    drawGrid(ctx);

    // 1. every element outline, faint — so a drawn optic is actually visible.
    for (const o of overlays()) {
      const poly = outlineOf(o.id);
      if (poly.length < 6) continue;
      const sel = o.id === selected;
      ctx.beginPath();
      const a = app.toScreen({ x: poly[0], y: poly[1] });
      ctx.moveTo(a.x, a.y);
      for (let i = 2; i < poly.length; i += 2) { const s = app.toScreen({ x: poly[i], y: poly[i + 1] }); ctx.lineTo(s.x, s.y); }
      if (o.locked) {
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = "rgba(150,160,180,0.4)"; // ghosted (§5.1)
        ctx.lineWidth = 1.2;
      } else {
        ctx.setLineDash([]);
        ctx.strokeStyle = sel ? "rgba(127,214,255,0.95)" : "rgba(160,190,225,0.35)";
        ctx.lineWidth = sel ? 2 : 1;
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 2. handles for the selected, unlocked element: centre dot + rotation ring.
    const selOv = selected >= 0 ? overlayOf(selected) : undefined;
    if (selOv && !selOv.locked && tool !== "draw") {
      const c = app.toScreen(selOv);
      ctx.fillStyle = "#7fd6ff";
      ctx.beginPath(); ctx.arc(c.x, c.y, 4, 0, Math.PI * 2); ctx.fill();
      const rh = app.toScreen(rotHandleWorld(selOv));
      ctx.strokeStyle = "rgba(127,214,255,0.6)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(rh.x, rh.y); ctx.stroke();
      ctx.fillStyle = "#ffd678";
      ctx.beginPath(); ctx.arc(rh.x, rh.y, 4, 0, Math.PI * 2); ctx.fill();
    }

    // 2b. leader lines — a thin connector from each open card to its element,
    //     so you can tell which card belongs to which when they're pulled apart.
    for (const card of cards) {
      const o = overlayOf(card.id);
      if (!o) continue;
      const s = app.toScreen(o);
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

    // 3. snap feedback — glowing guides (axis/vertex) or a grid marker (§5.2).
    if (lastSnap && lastSnap.snaps.length) {
      ctx.strokeStyle = "rgba(127,255,160,0.7)"; ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
      const W = view.clientWidth, H = view.clientHeight;
      if (lastSnap.guideY != null) { const y = app.toScreen({ x: 0, y: lastSnap.guideY }).y; line(ctx, 0, y, W, y); }
      if (lastSnap.guideX != null) { const x = app.toScreen({ x: lastSnap.guideX, y: 0 }).x; line(ctx, x, 0, x, H); }
      ctx.setLineDash([]);
    }
    if (gridMark) {
      const s = app.toScreen(gridMark);
      ctx.strokeStyle = "rgba(127,255,160,0.9)"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(s.x, s.y, 5, 0, Math.PI * 2); ctx.stroke();
    }

    // 4. draw tool: the raw stroke and the fitted-curve preview + residual.
    if (stroke && stroke.length >= 4) {
      ctx.strokeStyle = "rgba(255,255,255,0.5)"; ctx.lineWidth = 1.5;
      ctx.beginPath();
      const a = app.toScreen({ x: stroke[0], y: stroke[1] });
      ctx.moveTo(a.x, a.y);
      for (let i = 2; i < stroke.length; i += 2) { const s = app.toScreen({ x: stroke[i], y: stroke[i + 1] }); ctx.lineTo(s.x, s.y); }
      ctx.stroke();
    }
    if (pendingFit) {
      const curve = pendingFit.rungs[pendingRung].curve;
      ctx.strokeStyle = "#ffd678"; ctx.lineWidth = 2;
      ctx.beginPath();
      const a = app.toScreen({ x: curve[0], y: curve[1] });
      ctx.moveTo(a.x, a.y);
      for (let i = 2; i < curve.length; i += 2) { const s = app.toScreen({ x: curve[i], y: curve[i + 1] }); ctx.lineTo(s.x, s.y); }
      ctx.stroke();
      const r = pendingFit.rungs[pendingRung];
      const mid = app.toScreen({ x: curve[curve.length - 2], y: curve[curve.length - 1] });
      ctx.fillStyle = "#ffd678"; ctx.font = "12px ui-monospace, monospace";
      ctx.fillText(`${r.name} · ${r.residual_um.toFixed(2)} µm`, mid.x + 8, mid.y);
    }
    ctx.restore();
  }

  setTool("select");
  return { overlay };
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
