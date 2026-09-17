// Interaction quality for the 3D view (no three.js here).
//
// While the user drags, orbits, pans, zooms, or scrubs a visual control, the
// view renders at a lower pixel ratio. Shortly after the last such event it
// returns to the settled ratio and asks for one more frame. Nothing here touches
// the optics: the ray picture's own fast/exact split is decided by the tracer
// code, and this only trades pixels for frame time.

export type RenderQualityState = "interactive" | "settled";

/** Pixel-ratio caps. On a 2x or 3x display the view never renders at native
 *  resolution by default; these are the measured-good starting points from the
 *  spec, adjustable after profiling. */
export const SETTLED_DPR_CAP = 1.5;
export const INTERACTIVE_DPR_CAP = 1.0;

/** How long after the last interaction event the view returns to settled. */
export const SETTLE_DELAY_MS = 150;

export class QualityController {
  state: RenderQualityState = "settled";
  private timer: number | undefined;
  /** Held while a pointer gesture is in progress, so a slow drag with pauses
   *  longer than the settle delay does not flip back and forth. */
  private held = false;

  constructor(private readonly onChange: (state: RenderQualityState) => void) {}

  /** The pixel ratio for the current state. */
  pixelRatio(deviceRatio = window.devicePixelRatio || 1): number {
    const cap = this.state === "interactive" ? INTERACTIVE_DPR_CAP : SETTLED_DPR_CAP;
    return Math.min(deviceRatio, cap);
  }

  /** A pointer gesture started (drag, orbit, pan). */
  begin() {
    this.held = true;
    this.clearTimer();
    this.set("interactive");
  }

  /** The gesture ended; settle after the delay. */
  end() {
    this.held = false;
    this.settleSoon();
  }

  /** A one-shot interaction (a wheel tick, a slider step). */
  pulse() {
    if (!this.held) {
      this.set("interactive");
      this.settleSoon();
    }
  }

  /** Force the settled state now (view switch, tests). */
  settleNow() {
    this.held = false;
    this.clearTimer();
    this.set("settled");
  }

  dispose() {
    this.clearTimer();
  }

  private settleSoon() {
    this.clearTimer();
    this.timer = window.setTimeout(() => {
      this.timer = undefined;
      if (!this.held) this.set("settled");
    }, SETTLE_DELAY_MS);
  }

  private clearTimer() {
    if (this.timer !== undefined) {
      clearTimeout(this.timer);
      this.timer = undefined;
    }
  }

  private set(state: RenderQualityState) {
    if (state === this.state) return;
    this.state = state;
    this.onChange(state);
  }
}
