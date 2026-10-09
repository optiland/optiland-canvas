// The material picker for a coating layer (coating-fidelity spec CF-11): the
// built-in film catalogue with each entry's form, source and wavelength range,
// and a warning on any entry whose data do not reach a wavelength the scene
// launches. An entry with no data there is refused by a Strict trace and held
// at its table's end by a Permissive one, so the picker says so before the
// layer is made of it.

import type { Canvas } from "../../pkg/oc_wasm.js";

export type CatalogEntry = {
  key: string;
  name: string;
  form: string;
  reference: string;
  rangeNm: [number, number] | null;
  thermalNote: string | null;
};

/** The built-in catalogue, as the picker lists it. */
export function catalogEntries(scene: Canvas): CatalogEntry[] {
  try {
    const cat = JSON.parse(scene.filmCatalogJSON()) as {
      builtin: { key: string; material: any; thermalNote: string | null }[];
    };
    return cat.builtin.map((e) => {
      const p = e.material.provenance ?? {};
      const r = p.range_um as [number, number] | undefined;
      return {
        key: e.key,
        name: e.material.name as string,
        form: (e.material.form as string | undefined) ?? "Bulk",
        reference: (p.reference as string | undefined) ?? "",
        rangeNm: r ? [r[0] * 1000, r[1] * 1000] : null,
        thermalNote: e.thermalNote,
      };
    });
  } catch {
    return [];
  }
}

/** The scene's launch wavelengths an entry has no data at, nm. */
export function missingLines(e: CatalogEntry, lines: number[]): number[] {
  if (!e.rangeNm) return [];
  return lines.filter((nm) => nm < e.rangeNm![0] - 1e-6 || nm > e.rangeNm![1] + 1e-6);
}

export type FilmPicker = { el: HTMLSelectElement; set: (key: string | null) => void };

/** A select of catalogue entries; `onPick` gets the chosen key. */
export function filmPicker(
  parent: HTMLElement,
  scene: Canvas,
  opts: { title: string; onPick: (key: string) => void; placeholder?: string },
): FilmPicker {
  const sel = document.createElement("select");
  sel.className = "cv-film-pick";
  sel.title = opts.title;
  const lines = JSON.parse(scene.sceneWavelengthsJSON()) as number[];
  if (opts.placeholder) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = opts.placeholder;
    sel.appendChild(o);
  }
  for (const e of catalogEntries(scene)) {
    const o = document.createElement("option");
    o.value = e.key;
    const range = e.rangeNm ? ` · ${Math.round(e.rangeNm[0])} to ${Math.round(e.rangeNm[1])} nm` : "";
    const missing = missingLines(e, lines);
    o.textContent = `${e.name}${range}${missing.length ? ` · no data at ${Math.round(missing[0])} nm` : ""}`;
    o.title = [`${e.form}`, e.reference, e.thermalNote ?? ""].filter(Boolean).join("\n");
    if (missing.length) o.dataset.out = "1";
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => {
    if (sel.value) opts.onPick(sel.value);
  });
  parent.appendChild(sel);
  return {
    el: sel,
    set: (key) => {
      sel.value = key ?? "";
    },
  };
}
