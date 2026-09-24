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
  dni: number;
  band: [number, number];
  bands: number;
  beamRadius: number;
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
    setSunSky(dni: number, lo: number, hi: number, bands: number): boolean;
    setSunBeam(radius: number, standoff: number): boolean;
  };
  /** Retrace and redraw after an edit. */
  edited: (live: boolean) => void;
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
  const dni = $<HTMLInputElement>("sunDni");
  const beam = $<HTMLInputElement>("sunBeam");
  const readout = $<HTMLElement>("sunReadout");

  const read = (): SunState | null => JSON.parse(scene.solarJSON());

  const setIfIdle = (el: HTMLInputElement, v: string) => {
    if (document.activeElement !== el) el.value = v;
  };

  function sync() {
    const s = read();
    sec.hidden = !s;
    if (!s) return;
    setIfIdle(lat, String(s.site.lat));
    setIfIdle(lon, String(s.site.lon));
    setIfIdle(elev, String(s.site.elevation));
    const t = s.time;
    setIfIdle(date, `${t.year}-${String(t.month).padStart(2, "0")}-${String(t.day).padStart(2, "0")}`);
    setIfIdle(utc, String(t.utcOffset));
    setIfIdle(time, String(Math.round(t.minutes)));
    timeLabel.textContent = hhmm(t.minutes);
    setIfIdle(bearing, String(s.bearing));
    setIfIdle(dni, String(s.dni));
    setIfIdle(beam, String(Number((2 * s.beamRadius).toFixed(1))));
    // With heliostats the beam sizes itself to cover all of them.
    beam.disabled = s.heliostats.length > 0;
    beam.title = beam.disabled ? "sized to cover every heliostat" : "";
    const where = s.up
      ? `az ${s.azimuth.toFixed(1)}° · el ${s.elevationAngle.toFixed(1)}° · disc ±${s.halfAngleMrad.toFixed(2)} mrad`
      : `below the horizon (el ${s.elevationAngle.toFixed(1)}°): no direct sun`;
    const lines = [where, `beam ${s.fluxW.toFixed(1)} W (${s.band[0]}–${s.band[1]} µm, ${s.bands} lines)`];
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
  function commitBeam() {
    const s = read();
    if (!s) return;
    const r = num(beam) / 2;
    if (r > 0 && scene.setSunBeam(r, Math.max(r, 200))) deps.edited(false);
    sync();
  }

  // The slider previews while it moves and settles on release.
  time.addEventListener("input", () => commitTime(true));
  time.addEventListener("change", () => commitTime(false));
  date.addEventListener("change", () => commitTime(false));
  utc.addEventListener("change", () => commitTime(false));
  for (const el of [lat, lon, elev, bearing]) el.addEventListener("change", commitSite);
  dni.addEventListener("change", commitSky);
  beam.addEventListener("change", commitBeam);
  for (const el of [lat, lon, elev, date, utc, time, bearing, dni, beam]) {
    el.addEventListener("keydown", (e) => e.stopPropagation());
  }

  function ensureSun() {
    if (read()) return;
    const now = new Date();
    // Almería's solar research platform until someone says where they are.
    scene.addSun(
      37.094, -2.358, 500,
      now.getFullYear(), now.getMonth() + 1, now.getDate(),
      13 * 60, -now.getTimezoneOffset() / 60, 180,
    );
    deps.edited(false);
    sync();
  }

  sync();
  return { sync, ensureSun };
}
