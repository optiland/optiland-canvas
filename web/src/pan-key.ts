// Hold Space to pan: while it is down, a drag moves the view in both the 2D
// section and the 3D model, whatever is under the pointer. In a large scene
// (a heliostat field, a 37 m light pipe) the parts cover most of the window,
// so "drag empty space to pan" leaves nowhere to grab.

let held = false;

const typing = (t: EventTarget | null) =>
  t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement;

window.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || typing(e.target)) return;
  // Space would otherwise scroll the page or press the focused button.
  e.preventDefault();
  if (!held) {
    held = true;
    document.body.classList.add("pan-key");
  }
});

const release = () => {
  held = false;
  document.body.classList.remove("pan-key");
};
window.addEventListener("keyup", (e) => { if (e.code === "Space") release(); });
window.addEventListener("blur", release);

/** Is Space held down (outside a text field)? */
export const panKeyHeld = () => held;
