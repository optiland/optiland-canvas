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
//                          every parameter is a numeric twin of a handle (§5.2).
//                          The floating-window engine lives in ./cards; this
//                          module supplies the legacy-scene data adapter.
//   • Grid + snapping     — a spaced grid; drag snaps to it (Shift → axis /
//                          vertex snaps instead) (§5.2)
//   • Clipboard           — Del delete · Ctrl+C/V copy/paste · Ctrl+D duplicate
//   • Pin tool            — lock a parameter, or a whole element (ghosted) (§5.1)

import type { Canvas } from "../pkg/oc_wasm.js";
import { createCardManager, type CardSource } from "./cards";
import { panKeyHeld } from "./pan-key";

type V2 = { x: number; y: number };
type Tool = "select" | "draw" | "pin";
type Rung = { level: number; name: string; dof: number; residual_um: number; rms_um: number; curve: number[] };
type Fit = { chosen: number; tol_um: number; rungs: Rung[] };
type Overlay = { id: number; x: number; y: number; rot_deg: number; locked: boolean; kind: string; layer: number };
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
  /// True when a canonical 3D scene owns the canvas (§10.1): the v1 element
  /// layer is a hidden legacy scene, so its picking/selection is suppressed
  /// (pan and zoom still work). The Scene3 view owns interaction then.
  threeDActive?: () => boolean;
  /// False when another view (the orbitable 3D model) owns the canvas entirely:
  /// M3 then skips pan/zoom too, so it does not fight the 3D camera. Defaults to
  /// always-on. The 2D cross-section view keeps this true so its pan/zoom work.
  viewActive?: () => boolean;
  // --- M4 hooks (§5.2/§5.6). Optional so M3 stays usable standalone. ---
  /// Snapshot the settled scene into the undo tree (§5.2). Deduped in Rust.
  commit?: (label: string) => void;
  /// Layer show/hide + onion-skin policy for the element outlines (§5.6).
  layerVisible?: (layerId: number) => boolean;
  layerOpacity?: (layerId: number) => number;
  onionSkin?: () => boolean;
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

export function initM3(app: AppCtx): {
  overlay: (ctx: CanvasRenderingContext2D) => void;
  sync: () => void;
  getSelected: () => number;
} {
  const { scene, view } = app;
  // The 2D cross-section view lets M3 own pan/zoom; the 3D model view takes the
  // canvas over entirely, so M3 stands down (its handlers no-op) to avoid a tug
  // of war with the orbit camera.
  const viewOn = () => app.viewActive?.() ?? true;
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

  // ---- M4 layer policy + undo-commit shims (no-ops when unwired) --------
  const layerShown = (id: number) => (app.layerVisible ? app.layerVisible(id) : true);
  const layerAlpha = (id: number) => (app.layerOpacity ? app.layerOpacity(id) : 1);
  const onion = () => (app.onionSkin ? app.onionSkin() : false);
  const commit = (label: string) => app.commit?.(label);
  // Collapse a burst of continuous edits (a field scrub, a slider drag) into a
  // single undo-tree node ~450 ms after the last change (§5.2).
  let commitTimer: number | undefined;
  function commitSoon(label: string, ms = 450) {
    if (commitTimer) clearTimeout(commitTimer);
    commitTimer = window.setTimeout(() => { commitTimer = undefined; commit(label); }, ms);
  }

  function pointInPoly(wx: number, wy: number, poly: number[]): boolean {
    let inside = false;
    for (let i = 0, j = poly.length - 2; i < poly.length; j = i, i += 2) {
      const xi = poly[i], yi = poly[i + 1], xj = poly[j], yj = poly[j + 1];
      if ((yi > wy) !== (yj > wy) && wx < ((xj - xi) * (wy - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  // Nearest element whose body contains the world point, else the nearest
  // centre. Elements on hidden layers aren't pickable — make the layer visible
  // to edit it (§5.6).
  function pick(wx: number, wy: number): number {
    // A 3D scene owns the canvas: the v1 element layer is hidden legacy state, so
    // nothing here is pickable (empty-space drag still pans, wheel still zooms).
    if (app.threeDActive?.()) return -1;
    const ovs = overlays().filter((o) => layerShown(o.layer));
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

  // ================= inspector cards — legacy-scene adapter (§5.2) =======
  // The floating-window engine is in ./cards; this adapter is the element-
  // specific half: what a card shows (render) and how its fields write back to
  // the v1 scene (wire/commitCard).
  const cardTemplate = el<HTMLTemplateElement>("cardTemplate");

  // Write a card's field values back to the scene (unless the element is locked).
  function commitCard(id: number, root: HTMLElement) {
    if (scene.isElementLocked(id)) { cardMgr.render(id); return; }
    const val = (n: string) => Number((root.querySelector(`input[data-name="${n}"]`) as HTMLInputElement).value);
    scene.setElementPos(id, val("x"), val("y"));
    scene.setElementRotDeg(id, val("rot"));
    if (scene.singletParamsJson(id) !== "null") {
      scene.setSingletParams(
        id, val("radius1"), val("radius2"), val("thickness"),
        val("half_ap1"), val("half_ap2"), val("conic_k1"), val("conic_k2"),
      );
    }
    app.render(); app.refresh();
  }

  const cardSource: CardSource = {
    template: cardTemplate,
    toScreen: app.toScreen,
    render2d: app.render,
    anchorWorld: (id) => {
      const o = overlayOf(id);
      return o ? { x: o.x, y: o.y } : null;
    },
    render(id, root) {
      const t = JSON.parse(scene.elementTransformJson(id));
      if (!t) return false; // element vanished (deleted)
      const q = (sel: string) => root.querySelector(sel) as HTMLElement;
      const kind = overlayOf(id)?.kind ?? "element";
      q(".c-title").textContent = `#${id} · ${kind}`;
      const locked = scene.isElementLocked(id);
      (q(".c-lock") as HTMLInputElement).checked = locked;

      const setV = (name: string, v: number) => {
        const inp = root.querySelector(`input[data-name="${name}"]`) as HTMLInputElement;
        if (document.activeElement !== inp) inp.value = String(Number(v.toFixed(4)));
        inp.disabled = locked;
      };
      setV("x", t.x); setV("y", t.y); setV("rot", t.rot_deg);

      const sp = JSON.parse(scene.singletParamsJson(id));
      q(".c-singlet").style.display = sp ? "block" : "none";
      if (sp) for (const name of SINGLET_FIELDS) setV(name, sp[name]);

      const fit: (Fit & { current: number }) | null = JSON.parse(scene.elementFitJson(id));
      q(".c-fit").style.display = fit ? "block" : "none";
      if (fit) {
        (q(".c-fitslider") as HTMLInputElement).value = String(fit.current);
        const r = fit.rungs[fit.current];
        q(".c-fitlabel").textContent = `${r.name} · ${r.residual_um.toFixed(2)} µm`;
      }

      const pinned: string[] = JSON.parse(scene.pinnedParamsJson(id));
      root.querySelectorAll<HTMLButtonElement>("button.pin[data-name]").forEach((btn) => {
        const on = pinned.includes(btn.dataset.name!);
        btn.classList.toggle("pinned", on);
        btn.textContent = on ? "📌" : "📍";
      });
      return true;
    },
    wire(id, root, api) {
      const q = <T extends HTMLElement>(sel: string) => root.querySelector(sel) as T;
      q<HTMLInputElement>(".c-lock").addEventListener("change", (e) => {
        scene.setElementLocked(id, (e.target as HTMLInputElement).checked);
        api.rerender(id); app.render(); commit("lock");
      });
      q<HTMLInputElement>(".c-fitslider").addEventListener("input", (e) => {
        const res = scene.setElementFitRung(id, Number((e.target as HTMLInputElement).value));
        if (res >= 0) q(".c-fitlabel").textContent = `residual ${res.toFixed(2)} µm`;
        app.render(); app.refresh(); // ← rays visibly change as you slide (§5.3)
        commitSoon("fit complexity");
      });
      root.querySelectorAll<HTMLInputElement>("input[data-name]").forEach((input) => {
        const name = input.dataset.name!;
        const step = STEP[name] ?? 0.25;
        let before = "";
        input.addEventListener("focus", () => { before = input.value; });
        input.addEventListener("keydown", (e) => {
          if (e.key === "Escape") { input.value = before; input.blur(); commitCard(id, root); }
          e.stopPropagation(); // keep field keys out of the global shortcuts
        });
        input.addEventListener("input", () => { commitCard(id, root); commitSoon("edit params"); });
        input.addEventListener("wheel", (e) => { // hover-scroll scrubs (§5.2)
          e.preventDefault();
          const dir = e.deltaY < 0 ? 1 : -1;
          input.value = String(Number((Number(input.value) + dir * step * (e.shiftKey ? 10 : 1)).toFixed(6)));
          commitCard(id, root); commitSoon("edit params");
        }, { passive: false });
      });
      root.querySelectorAll<HTMLButtonElement>("button.pin[data-name]").forEach((btn) => {
        const name = btn.dataset.name!;
        btn.addEventListener("click", () => {
          scene.setParamPinned(id, name, !scene.isParamPinned(id, name));
          api.rerender(id); commit("pin");
        });
      });
    },
  };

  const cardMgr = createCardManager(cardSource);

  // ================= tool palette =======================================
  // The v1 select/draw/pin palette is retired in the always-3D app (the Scene3
  // view owns editing). When its buttons/fit panel are absent, these lookups are
  // null and the tool concept degrades to a no-op: pan/zoom and the shared grid
  // are all m3 still contributes, and both work without a palette.
  const maybe = <T extends HTMLElement>(id: string) => document.getElementById(id) as T | null;
  const toolBtns: Partial<Record<Tool, HTMLButtonElement | null>> = {
    select: maybe<HTMLButtonElement>("tool-select"),
    draw: maybe<HTMLButtonElement>("tool-draw"),
    pin: maybe<HTMLButtonElement>("tool-pin"),
  };
  const hasTools = !!toolBtns.select;
  function setTool(t: Tool) {
    if (t === "draw" && !hasFit) return; // no fit panel → drawing is disabled
    if (t !== "draw") cancelDraw();
    tool = t;
    if (!hasTools) return;
    for (const k of Object.keys(toolBtns) as Tool[]) toolBtns[k]?.classList.toggle("active", k === t);
    view.style.cursor = t === "draw" ? "crosshair" : t === "pin" ? "not-allowed" : "grab";
    const hint = maybe("m3hint");
    if (hint) hint.textContent =
      t === "draw" ? "drag to sketch a surface · release to fit"
      : t === "pin" ? "click an element to lock it (ghosted)"
      : "drag · dbl-click inspect · scroll=zoom · drag bg=pan · 0 resets";
    app.render();
  }
  if (hasTools) for (const k of Object.keys(toolBtns) as Tool[]) {
    const btn = toolBtns[k];
    if (btn) btn.onclick = () => setTool(k);
  }

  // ================= keyboard: tools, clipboard, delete, nudge ==========
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement) return; // let fields have their keys
    const mod = e.ctrlKey || e.metaKey;
    // A 3D scene owns element editing: the clipboard/delete/nudge verbs act on
    // the Scene3 selection (handled in slice3d), so defer them here.
    const in3d = app.threeDActive?.() ?? false;
    if (e.key === "Shift") { snapOn = true; return; }
    if (!in3d && mod && (e.key === "c" || e.key === "C")) { copySelection(); e.preventDefault(); return; }
    if (!in3d && mod && (e.key === "v" || e.key === "V")) { pasteClipboard(); e.preventDefault(); return; }
    if (!in3d && mod && (e.key === "x" || e.key === "X")) { copySelection(); deleteSelection(); e.preventDefault(); return; }
    if (!in3d && mod && (e.key === "d" || e.key === "D")) { copySelection(); pasteClipboard(); e.preventDefault(); return; }
    if (!in3d && (e.key === "Delete" || e.key === "Backspace")) { deleteSelection(); e.preventDefault(); return; }
    if (e.key === "Escape") { cardMgr.closeAll(); deselect(); return; }
    if (!in3d && e.key.startsWith("Arrow")) { nudge(e.key); e.preventDefault(); return; }
    if (mod) return; // don't hijack other shortcuts
    if (hasTools && e.key === "v") setTool("select");
    else if (hasTools && e.key === "d") setTool("draw");
    else if (hasTools && e.key === "p") setTool("pin");
    else if (!viewOn()) return; // the view keys below belong to the active view
    else if (e.key === "0") { app.resetView(); cardMgr.positionAll(); app.render(); }
    else if (e.key === "+" || e.key === "=") { app.zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1.12); cardMgr.positionAll(); app.render(); }
    else if (e.key === "-" || e.key === "_") { app.zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1 / 1.12); cardMgr.positionAll(); app.render(); }
  });
  window.addEventListener("keyup", (e) => { if (e.key === "Shift") snapOn = false; });
  window.addEventListener("resize", () => cardMgr.positionAll());

  function nudge(arrow: string) {
    if (selected < 0) return;
    const o = overlayOf(selected);
    if (!o) return;
    const s = gridOn() ? gridStep() : 1;
    const dx = arrow === "ArrowLeft" ? -s : arrow === "ArrowRight" ? s : 0;
    const dy = arrow === "ArrowUp" ? s : arrow === "ArrowDown" ? -s : 0;
    scene.setElementPos(selected, o.x + dx, o.y + dy);
    afterSceneEdit("nudge");
  }

  // ================= element lifecycle (delete / copy / paste) ==========
  function copySelection() { if (selected >= 0) clipboard = scene.cloneElementJson(selected); }
  function deleteSelection() {
    if (selected < 0) return;
    const del = selected;
    if (!scene.deleteElement(del)) return;
    // ids above `del` shift down by one; keep the open cards in step.
    cardMgr.close(del);
    cardMgr.shiftIdsAfter(del);
    deselect();
    afterSceneEdit("delete");
  }
  function pasteClipboard() {
    if (!clipboard) return;
    pasteN++;
    const off = gridOn() ? gridStep() : 5;
    const id = scene.pasteElement(clipboard, off * pasteN, -off * pasteN);
    if (id >= 0) { selected = id; cardMgr.open(id); afterSceneEdit("paste"); }
  }
  function afterSceneEdit(label = "edit") { cardMgr.renderAll(); cardMgr.positionAll(); app.render(); app.refresh(); commit(label); }

  // ================= view: zoom + pan ===================================
  let pan: { x: number; y: number } | null = null;
  const startPan = (e: PointerEvent) => { pan = { x: e.clientX, y: e.clientY }; view.setPointerCapture(e.pointerId); };
  view.addEventListener("wheel", (e) => {
    if (!viewOn()) return; // the 3D model view owns the wheel
    e.preventDefault();
    app.zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.12 : 1 / 1.12); // zoom to cursor
    cardMgr.positionAll();
    app.render();
  }, { passive: false });

  // ================= select / move / rotate =============================
  let drag: { id: number; mode: "move" | "rotate"; dx: number; dy: number } | null = null;
  let lastSnap: Snap | null = null;
  let gridMark: V2 | null = null;

  view.addEventListener("pointerdown", (e) => {
    if (!viewOn()) return; // the 3D model view owns the canvas
    // Middle-drag, or any drag with Space held, pans anywhere, over parts too.
    if (e.button === 1 || panKeyHeld()) { startPan(e); e.preventDefault(); return; }
    if (tool === "draw") return startDraw(e);
    const p = app.toWorld(e.clientX, e.clientY);
    if (tool === "pin") {
      const id = pick(p.x, p.y);
      if (id >= 0) { scene.setElementLocked(id, !scene.isElementLocked(id)); selected = id; cardMgr.render(id); app.render(); commit("lock"); }
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
      cardMgr.positionAll();
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
    cardMgr.refresh(drag.id); // the card follows its element
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
    const mode = drag.mode;
    drag = null; lastSnap = null; gridMark = null;
    try { view.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    app.render();
    app.refresh(); // numbers settle on release (§4.1)
    commit(mode === "rotate" ? "rotate" : "move"); // undo-tree snapshot (§5.2)
  };
  view.addEventListener("pointerup", endDrag);
  view.addEventListener("pointercancel", endDrag);

  // double-click → open an inspector card for that element
  view.addEventListener("dblclick", (e) => {
    if (tool !== "select") return;
    const p = app.toWorld(e.clientX, e.clientY);
    const id = pick(p.x, p.y);
    if (id >= 0) { selected = id; cardMgr.open(id); app.render(); }
  });

  // ================= draw tool (freehand → fit, §5.3) ===================
  let stroke: number[] | null = null;         // world coords, flat [x,y,...]
  let drawingNow = false;                      // pointer is down and sketching
  let pendingFit: Fit | null = null;
  let pendingRung = 0;
  const fitPanel = maybe("fitpanel");
  const fitSlider = maybe<HTMLInputElement>("fitSlider");
  const fitLabel = maybe("fitLabel");
  const fitMirror = maybe<HTMLInputElement>("fitMirror");
  const hasFit = !!fitPanel;

  function startDraw(e: PointerEvent) {
    if (!hasFit) return;
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
    if (fitSlider) fitSlider.value = String(pendingRung);
    updateFitLabel();
    if (fitPanel) fitPanel.style.display = "block";
    app.render();
  }
  function updateFitLabel() {
    if (!pendingFit || !fitLabel) return;
    const r = pendingFit.rungs[pendingRung];
    const dof = r.dof > 1000 ? "free" : String(r.dof);
    fitLabel.textContent = `${r.name} · ${dof} dof · residual ${r.residual_um.toFixed(2)} µm`;
  }
  if (fitSlider) fitSlider.oninput = () => { pendingRung = Number(fitSlider.value); updateFitLabel(); app.render(); };
  if (hasFit) maybe<HTMLButtonElement>("fitCommit")!.onclick = () => {
    if (!stroke || !pendingFit) return;
    const id = scene.addFittedElement(
      new Float64Array(stroke), 1.0, pendingRung, DRAW_THICKNESS,
      DRAW_GLASS.nd, DRAW_GLASS.vd, fitMirror!.checked, // ← draw a mirror instead of glass
    );
    cancelDraw();
    if (id >= 0) { selected = id; setTool("select"); }
    app.render(); app.refresh();
    if (id >= 0) commit("draw"); // undo-tree snapshot (§5.2)
  };
  if (hasFit) maybe<HTMLButtonElement>("fitCancel")!.onclick = () => { cancelDraw(); app.render(); };
  function cancelDraw() {
    stroke = null; drawingNow = false; pendingFit = null;
    if (fitPanel) fitPanel.style.display = "none";
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
    //    Layer policy (§5.6): a hidden layer's optics don't trace; draw them
    //    only as onion-skin ghosts (and only when that toggle is on). A visible
    //    layer's outlines are dimmed by the layer's opacity.
    for (const o of overlays()) {
      const poly = outlineOf(o.id);
      if (poly.length < 6) continue;
      const shown = layerShown(o.layer);
      if (!shown && !onion()) continue;
      const sel = o.id === selected;
      ctx.beginPath();
      const a = app.toScreen({ x: poly[0], y: poly[1] });
      ctx.moveTo(a.x, a.y);
      for (let i = 2; i < poly.length; i += 2) { const s = app.toScreen({ x: poly[i], y: poly[i + 1] }); ctx.lineTo(s.x, s.y); }
      if (!shown) {
        // Onion-skin ghost of an alternate architecture (§5.6).
        ctx.setLineDash([2, 3]);
        ctx.strokeStyle = "rgba(150,175,210,0.28)";
        ctx.lineWidth = 1;
      } else if (o.locked) {
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = `rgba(150,160,180,${0.4 * layerAlpha(o.layer)})`; // ghosted (§5.1)
        ctx.lineWidth = 1.2;
      } else {
        ctx.setLineDash([]);
        const al = layerAlpha(o.layer);
        ctx.strokeStyle = sel ? `rgba(127,214,255,${0.95 * al})` : `rgba(160,190,225,${0.35 * al})`;
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
    cardMgr.drawLeaders(ctx, selected);

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

  // Re-sync UI to the scene after an external mutation the module didn't drive
  // (undo/redo/jump, a layer edit): clamp the selection and refresh open cards
  // (the card source's render self-closes any whose element vanished).
  function sync() {
    const n = scene.elementCount();
    if (selected >= n) selected = n > 0 ? n - 1 : -1;
    cardMgr.renderAll();
    cardMgr.positionAll();
  }

  setTool("select");
  return { overlay, sync, getSelected: () => selected };
}

function line(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
}
