// The sun panel: where and when, and what the sky delivers.
//
// A scene has at most one sun. The kernel places it from the site and the
// clock (NREL SPA) and turns every heliostat to send it onto its target, so
// this panel only edits the rig and reads back what the rig decided: where
// the sun stands, how many watts its beam carries, and each heliostat's
// cosine efficiency. Dragging the time slider sweeps the day, and the mirrors
// follow it.

type SunState = {
  site: { lat: number; lon: number; elevation: number };
  time: { year: number; month: number; day: number; minutes: number; utcOffset: number };
  bearing: number;
  upsideDown: boolean;
  dni: number;
  band: [number, number];
  bands: number;
  beamRadius: number;
  standoff: number;
  continuous: boolean;
  disc: "uniform" | "limb" | "buie";
  csr: number;
  spectrum: { name: string; rows: number; from: number; to: number } | null;
  wavelengths: number;
  azimuth: number;
  elevationAngle: number;
  up: boolean;
  halfAngleMrad: number;
  fluxW: number;
  heliostats: { mirror: number; incidence: number; cosine: number; tracking: boolean }[];
};

/** What the panel needs from the page. */
export type SunPanelDeps = {
  scene: {
    solarJSON(): string;
    addSun(
      lat: number, lon: number, elev: number, year: number, month: number, day: number,
      minutes: number, utcOffset: number, bearing: number,
    ): number;
    setSunTime(year: number, month: number, day: number, minutes: number, utcOffset: number): boolean;
    setSunSite(lat: number, lon: number, elev: number, bearing: number): boolean;
    setSunUpsideDown(on: boolean): boolean;
    setSunSky(dni: number, lo: number, hi: number, bands: number): boolean;
    setSunBeam(radius: number, standoff: number): boolean;
    setSunShape(disc: string, csr: number): boolean;
    setSunSpectral(continuous: boolean, bands: number): boolean;
    importSunSpectrum(name: string, text: string): string;
    clearSunSpectrum(): boolean;
  };
  /** Retrace and redraw after an edit. */
  edited: (live: boolean) => void;
  /** Which way up the model is drawn, read on every sync (false without a sun). */
  upsideDown?: (on: boolean) => void;
};

export type SunPanel = {
  /** Re-read the rig (after a load, an undo, an edit elsewhere). */
  sync: () => void;
  /** Add a sun if there is none, with today's date on the local clock. */
  ensureSun: () => void;
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const hhmm = (minutes: number) => {
  const m = Math.round(minutes);
  return `${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
};

export function initSunPanel(deps: SunPanelDeps): SunPanel {
  const { scene } = deps;
  const sec = $<HTMLElement>("sunSec");
  const lat = $<HTMLInputElement>("sunLat");
  const lon = $<HTMLInputElement>("sunLon");
  const elev = $<HTMLInputElement>("sunElev");
  const date = $<HTMLInputElement>("sunDate");
  const utc = $<HTMLInputElement>("sunUtc");
  const time = $<HTMLInputElement>("sunTime");
  const timeLabel = $<HTMLElement>("sunTimeLabel");
  const bearing = $<HTMLInputElement>("sunBearing");
  const upsideDown = $<HTMLInputElement>("sunUpsideDown");
  const dni = $<HTMLInputElement>("sunDni");
  const beam = $<HTMLInputElement>("sunBeam");
  const standoff = $<HTMLInputElement>("sunStandoff");
  const readout = $<HTMLElement>("sunReadout");
  const disc = $<HTMLSelectElement>("sunDisc");
  const csr = $<HTMLInputElement>("sunCsr");
  const spectral = $<HTMLSelectElement>("sunSpectral");
  const bands = $<HTMLInputElement>("sunBands");
  const table = $<HTMLElement>("sunTable");
  const importBtn = $<HTMLButtonElement>("sunImport");
  const g173 = $<HTMLButtonElement>("sunG173");
  const file = $<HTMLInputElement>("sunFile");
  // What the last import read, or why it was refused; shown until the next.
  let imported = "";

  const read = (): SunState | null => JSON.parse(scene.solarJSON());

  const setIfIdle = (el: HTMLInputElement, v: string) => {
    if (document.activeElement !== el) el.value = v;
  };

  function sync() {
    const s = read();
    sec.hidden = !s;
    deps.upsideDown?.(!!s?.upsideDown);
    if (!s) return;
    upsideDown.checked = s.upsideDown;
    setIfIdle(lat, String(s.site.lat));
    setIfIdle(lon, String(s.site.lon));
    setIfIdle(elev, String(s.site.elevation));
    const t = s.time;
    setIfIdle(date, `${t.year}-${String(t.month).padStart(2, "0")}-${String(t.day).padStart(2, "0")}`);
    setIfIdle(utc, String(t.utcOffset));
    setIfIdle(time, String(Math.round(t.minutes)));
    timeLabel.textContent = hhmm(t.minutes);
    setIfIdle(bearing, String(s.bearing));
    // Two decimals: an imported table's total is its DNI, to more digits
    // than the field needs.
    setIfIdle(dni, String(+s.dni.toFixed(2)));
    setIfIdle(beam, String(Number((2 * s.beamRadius).toFixed(1))));
    setIfIdle(standoff, String(Math.round(s.standoff)));
    if (document.activeElement !== disc) disc.value = s.disc;
    setIfIdle(csr, String(Number((100 * s.csr).toFixed(2))));
    if (document.activeElement !== spectral) spectral.value = s.continuous ? "continuous" : "lines";
    setIfIdle(bands, String(s.bands));
    bands.hidden = s.continuous;
    table.textContent = s.spectrum ? s.spectrum.name : "ASTM G173-03";
    table.title = s.spectrum
      ? `${s.spectrum.rows} rows, ${s.spectrum.from}–${s.spectrum.to} nm, imported`
      : "ASTM G173-03 direct normal + circumsolar, air mass 1.5, 280–4000 nm";
    g173.disabled = !s.spectrum;
    // With heliostats the beam sizes itself to cover all of them.
    beam.disabled = s.heliostats.length > 0;
    beam.title = beam.disabled ? "sized to cover every heliostat" : "";
    const where = s.up
      ? `az ${s.azimuth.toFixed(1)}° · el ${s.elevationAngle.toFixed(1)}° · disc ±${s.halfAngleMrad.toFixed(2)} mrad`
      : `below the horizon (el ${s.elevationAngle.toFixed(1)}°): no direct sun`;
    const sampled = s.continuous ? `continuous, ${s.wavelengths} wavelengths` : `${s.bands} lines`;
    const shape =
      (s.disc === "limb" ? "limb-darkened disc" : s.disc === "buie" ? "Buie disc" : "uniform disc") +
      (s.csr > 0 ? `, ${Number((100 * s.csr).toFixed(2))}% aureole` : "");
    const lines = [where, `beam ${s.fluxW.toFixed(1)} W (${s.band[0]}–${s.band[1]} µm, ${sampled})`, shape];
    if (imported) lines.push(imported);
    for (const h of s.heliostats) {
      lines.push(
        h.tracking
          ? `heliostat #${h.mirror}: incidence ${h.incidence.toFixed(1)}° · cosine ${h.cosine.toFixed(3)}`
          : `heliostat #${h.mirror}: cannot reach its target from here`,
      );
    }
    readout.textContent = lines.join("\n");
  }

  const num = (el: HTMLInputElement) => Number(el.value);
  const clock = () => {
    const [y, m, d] = date.value.split("-").map(Number);
    return { y, m, d };
  };

  function commitTime(live: boolean) {
    const { y, m, d } = clock();
    if (!y) return;
    timeLabel.textContent = hhmm(num(time));
    if (scene.setSunTime(y, m, d, num(time), num(utc))) deps.edited(live);
    sync();
  }
  function commitSite() {
    if (scene.setSunSite(num(lat), num(lon), num(elev), num(bearing))) deps.edited(false);
    sync();
  }
  function commitSky() {
    const s = read();
    if (!s) return;
    if (scene.setSunSky(num(dni), s.band[0], s.band[1], s.bands)) deps.edited(false);
    sync();
  }
  function commitShape() {
    const c = num(csr) / 100;
    if (scene.setSunShape(disc.value, c)) deps.edited(false);
    sync();
  }
  function commitSpectral() {
    const n = Math.max(1, Math.round(num(bands)) || 6);
    if (scene.setSunSpectral(spectral.value === "continuous", n)) deps.edited(false);
    sync();
  }
  async function importTable() {
    const f = file.files?.[0];
    file.value = "";
    if (!f) return;
    const said = scene.importSunSpectrum(f.name, await f.text());
    imported = said.startsWith("error: ") ? `${f.name} refused: ${said.slice(7)}` : `imported ${said}`;
    if (!said.startsWith("error: ")) deps.edited(false);
    sync();
  }

  function commitBeam() {
    const s = read();
    if (!s) return;
    // Over a field the kernel sizes the beam, and keeps the disc at least a
    // radius out; on one target the distance is the user's, 0 or more.
    const r = beam.disabled ? s.beamRadius : num(beam) / 2;
    const d = Math.max(0, num(standoff));
    if (r > 0 && Number.isFinite(d) && scene.setSunBeam(r, d)) deps.edited(false);
    sync();
  }

  // The slider previews while it moves and settles on release.
  time.addEventListener("input", () => commitTime(true));
  time.addEventListener("change", () => commitTime(false));
  date.addEventListener("change", () => commitTime(false));
  utc.addEventListener("change", () => commitTime(false));
  for (const el of [lat, lon, elev, bearing]) el.addEventListener("change", commitSite);
  upsideDown.addEventListener("change", () => {
    if (scene.setSunUpsideDown(upsideDown.checked)) deps.edited(false);
    sync();
  });
  dni.addEventListener("change", commitSky);
  beam.addEventListener("change", commitBeam);
  standoff.addEventListener("change", commitBeam);
  disc.addEventListener("change", commitShape);
  csr.addEventListener("change", commitShape);
  spectral.addEventListener("change", commitSpectral);
  bands.addEventListener("change", commitSpectral);
  importBtn.addEventListener("click", () => file.click());
  file.addEventListener("change", () => void importTable());
  g173.addEventListener("click", () => {
    imported = "";
    if (scene.clearSunSpectrum()) deps.edited(false);
    sync();
  });
  for (const el of [lat, lon, elev, date, utc, time, bearing, dni, beam, standoff, csr, bands]) {
    el.addEventListener("keydown", (e) => e.stopPropagation());
  }

  function ensureSun() {
    if (read()) return;
    const now = new Date();
    // Jena, the Deutsches Optisches Museum at Carl-Zeiss-Platz 12, until
    // someone says where they are; the clock at about solar noon on this
    // browser's own zone (12:00 plus the zone, less 4 minutes a degree east,
    // leaving out the equation of time's quarter hour).
    const lon0 = 11.57928;
    const zone = -now.getTimezoneOffset() / 60;
    scene.addSun(
      50.92781, lon0, 160,
      now.getFullYear(), now.getMonth() + 1, now.getDate(),
      Math.round(12 * 60 + 60 * zone - 4 * lon0), zone, 180,
    );
    deps.edited(false);
    sync();
  }

  sync();
  return { sync, ensureSun };
}
