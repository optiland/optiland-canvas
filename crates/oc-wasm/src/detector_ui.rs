//! What the detector's component view reads and sets beyond the face: the
//! layout of an intensity histogram, and the per-wavelength numbers its colour
//! diagnostic plots.

use oc_core::scene3::{DetectorMeasure, DirectionLayout};
use wasm_bindgen::prelude::*;

use crate::{json_num, Canvas};

/// The bins an intensity histogram starts with in each layout: the `I(θ)`
/// curve polar, and an odd grid, so one cell sits on the normal and a
/// cross-section through the middle runs along the axis.
pub(crate) const POLAR_BINS: [usize; 2] = [24, 1];
pub(crate) const GRID_BINS: [usize; 2] = [31, 31];

#[wasm_bindgen]
impl Canvas {
    /// Lay a detector's intensity histogram out as `"grid"` (two angles, the
    /// far field with straight axes) or `"polar"` (rings and sectors). A
    /// layout switched to starts with its own default bins. Bumps the
    /// revision. False, with the scene unchanged, for a detector that
    /// measures anything but intensity, or an unknown layout.
    #[wasm_bindgen(js_name = setDetectorDirectionLayout)]
    pub fn set_detector_direction_layout(&mut self, id: f64, layout: &str) -> bool {
        let layout = match layout {
            "grid" => DirectionLayout::Grid,
            "polar" => DirectionLayout::Polar,
            _ => return false,
        };
        let Some(scene) = self.scene3.as_mut() else {
            return false;
        };
        let was = scene
            .detectors
            .iter()
            .find(|d| d.common.id.0 == id as u64)
            .and_then(|d| d.measure.direction_layout());
        if was == Some(layout) {
            return true;
        }
        let bins = match layout {
            DirectionLayout::Grid => GRID_BINS,
            DirectionLayout::Polar => POLAR_BINS,
        };
        if !scene.set_detector_direction_layout(id as u64, layout, bins) {
            return false;
        }
        self.bump();
        true
    }
}

/// The layout of `measure`'s direction histogram, for the readout.
pub(crate) fn layout_json(measure: &DetectorMeasure) -> String {
    measure
        .direction_layout()
        .map_or("null".into(), |l| format!("\"{}\"", l.name()))
}

/// A detector's hits by wavelength, as the colour diagnostic reads them: one
/// object per wavelength bin, in order of wavelength, with
/// `{nm, hits, flux, centroid: [u, v], rms, rmsUv: [u, v], angle: [x, y],
/// spread, focus, focusUv: [u, v], sums}`.
/// `nm` is the flux-weighted wavelength of the bin's hits. `sums` are the
/// bin's raw sums (`SpectralSums3::to_array` after its first two fields), so
/// a reader can merge bins exactly before taking any statistic.
/// Lengths are mm. `angle` is the mean arrival direction's angle to the
/// normal in the `u` and `v` planes and `spread` the RMS angular spread about
/// it, both radians. `focus` is how far along the normal, in the direction
/// of travel, this line's spot is smallest, or `null` for rays that are all
/// parallel. `rmsUv` and `focusUv` are `rms` and `focus` along the face's
/// `u` and `v` axes each on its own, for light that focuses differently in
/// the two (a cylindrical lens); `rms` is `hypot` of `rmsUv`, and either
/// entry of `focusUv` is `null` when the rays are parallel in that plane.
pub(crate) fn spectral_json(d: &oc_trace::DetectorResult3) -> String {
    let lines: Vec<String> = d
        .spectral
        .iter()
        .filter(|l| l.flux > 0.0)
        .map(|l| {
            let c = l.centroid().unwrap_or_default();
            let a = l.mean_slope().unwrap_or_default();
            format!(
                "{{\"nm\":{},\"hits\":{},\"flux\":{},\"centroid\":[{},{}],\"rms\":{},\"rmsUv\":{},\"angle\":[{},{}],\"spread\":{},\"focus\":{},\"focusUv\":{},\"sums\":[{}]}}",
                json_num(l.mean_lambda_um() * 1000.0),
                l.hits,
                json_num(l.flux),
                json_num(c.x),
                json_num(c.y),
                l.rms_radius().map_or("null".into(), json_num),
                l.rms_uv()
                    .map_or("null".into(), |r| format!("[{},{}]", json_num(r[0]), json_num(r[1]))),
                json_num(a.x.atan()),
                json_num(a.y.atan()),
                l.rms_slope().map_or("null".into(), json_num),
                l.best_focus_mm().map_or("null".into(), json_num),
                l.best_focus_uv_mm().map_or("null".into(), |f| {
                    let axis = |s: Option<f64>| s.map_or("null".into(), json_num);
                    format!("[{},{}]", axis(f[0]), axis(f[1]))
                }),
                l.to_array()[2..].iter().map(|v| json_num(*v)).collect::<Vec<_>>().join(","),
            )
        })
        .collect();
    format!("[{}]", lines.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default scene's detector, measuring intensity.
    fn intensity_canvas() -> (Canvas, f64) {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let id = c.scene3.as_ref().unwrap().detectors[0].common.id.0 as f64;
        assert!(c.set_detector_measure(id, "intensity", 20.0));
        (c, id)
    }

    #[test]
    fn intensity_starts_on_a_grid_and_switches_to_polar() {
        let (mut c, id) = intensity_canvas();
        let m = c.scene3.as_ref().unwrap().detectors[0].measure;
        assert_eq!(m.direction_layout(), Some(DirectionLayout::Grid));
        assert_eq!(m.direction_grid().map(|g| (g.1, g.2)), Some((31, 31)));
        assert!(c.set_detector_direction_layout(id, "polar"));
        let m = c.scene3.as_ref().unwrap().detectors[0].measure;
        assert_eq!(m.direction_layout(), Some(DirectionLayout::Polar));
        assert_eq!(m.direction_grid().map(|g| (g.1, g.2)), Some((24, 1)));
        assert!(!c.set_detector_direction_layout(id, "hexagonal"));
        let json: serde_json::Value = serde_json::from_str(&c.detector_result_json(id)).unwrap();
        assert_eq!(json["directionLayout"], "polar");
    }

    #[test]
    fn the_readout_breaks_the_flux_out_by_wavelength() {
        let (c, id) = intensity_canvas();
        let json: serde_json::Value = serde_json::from_str(&c.detector_result_json(id)).unwrap();
        let lines = json["spectral"].as_array().unwrap();
        assert!(!lines.is_empty());
        let sum: f64 = lines.iter().map(|l| l["flux"].as_f64().unwrap()).sum();
        let total = json["totalIncident"].as_f64().unwrap();
        assert!((sum - total).abs() <= 1e-9 * total, "{sum} vs {total}");
    }

    /// Each line's spot and focus per axis: the spot sizes add in quadrature
    /// to `rms`, and `focus` is the two axes' foci weighted by each one's
    /// slope variance, both from the line's own `sums`.
    #[test]
    fn the_readout_gives_spot_and_focus_per_axis() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let id = c.scene3.as_ref().unwrap().detectors[0].common.id.0 as f64;
        let json: serde_json::Value = serde_json::from_str(&c.detector_result_json(id)).unwrap();
        let lines = json["spectral"].as_array().unwrap();
        assert!(!lines.is_empty());
        for l in lines {
            let f = |v: &serde_json::Value| v.as_f64().unwrap();
            let [ru, rv] = [f(&l["rmsUv"][0]), f(&l["rmsUv"][1])];
            let rms = f(&l["rms"]);
            assert!((ru.hypot(rv) - rms).abs() <= 1e-12 * rms, "{ru}, {rv} against {rms}");
            let [fu, fv] = [f(&l["focusUv"][0]), f(&l["focusUv"][1])];
            let s: Vec<f64> = l["sums"].as_array().unwrap().iter().map(f).collect();
            let w = s[1];
            let var_a = s[8] / w - (s[6] / w).powi(2);
            let var_b = s[9] / w - (s[7] / w).powi(2);
            let mean = (var_a * fu + var_b * fv) / (var_a + var_b);
            let focus = f(&l["focus"]);
            assert!((mean - focus).abs() <= 1e-9 * focus.abs().max(1.0), "{mean} against {focus}");
        }
    }

    #[test]
    fn more_analysis_rays_keep_the_totals_and_the_picture() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let id = c.scene3.as_ref().unwrap().detectors[0].common.id.0 as f64;
        let read = |c: &Canvas| -> (f64, u64) {
            let v: serde_json::Value = serde_json::from_str(&c.detector_result_json(id)).unwrap();
            (
                v["totalIncident"].as_f64().unwrap(),
                v["hits"].as_u64().unwrap(),
            )
        };
        let (flux1, hits1) = read(&c);
        let picture1 = c.trace_projected_3d().len();
        let own = c.launch_ray_counts()[0];
        assert_eq!(c.set_analysis_rays(16.0 * own), 16.0 * own);
        let (flux16, hits16) = read(&c);
        assert!(hits16 > 8 * hits1, "{hits16} hits at 16x, {hits1} at 1x");
        assert!(
            (flux16 - flux1).abs() <= 0.05 * flux1,
            "{flux16} vs {flux1}"
        );
        assert_eq!(
            c.trace_projected_3d().len(),
            picture1,
            "the picture keeps its rays"
        );
        let counts = c.launch_ray_counts();
        assert!((counts[1] - 16.0 * counts[0]).abs() <= 0.05 * counts[1], "{counts:?}");
        assert_eq!(counts[2], 1.0);
        // Fewer than the scene's own traces the scene's own.
        c.set_analysis_rays(own / 2.0);
        assert_eq!(c.launch_ray_counts()[1], own);
        assert_eq!(c.set_analysis_rays(0.0), 0.0);
        assert_eq!(c.set_analysis_rays(-5.0), 0.0);
        assert_eq!(c.set_analysis_rays(f64::NAN), 0.0);
        assert_eq!(c.set_analysis_rays(1e12), 1e9);
    }

    /// Past one pass the numbers come in several, and every total stays what
    /// the scene's own rays give while the hits grow with the count.
    #[test]
    fn a_count_past_one_pass_is_traced_in_passes() {
        let mut c = Canvas::new();
        c.new_default_scene3();
        let id = c.scene3.as_ref().unwrap().detectors[0].common.id.0 as f64;
        let read = |c: &Canvas| -> (f64, u64) {
            let v: serde_json::Value = serde_json::from_str(&c.detector_result_json(id)).unwrap();
            (
                v["totalIncident"].as_f64().unwrap(),
                v["hits"].as_u64().unwrap(),
            )
        };
        let (flux1, hits1) = read(&c);
        c.set_analysis_rays(1.5e6);
        let counts = c.launch_ray_counts();
        assert_eq!(counts[2], 3.0, "{counts:?}");
        assert!((counts[1] - 1.5e6).abs() <= 0.02 * 1.5e6, "{counts:?}");
        let (flux, hits) = read(&c);
        assert!(hits as f64 > 0.5 * hits1 as f64 * counts[1] / counts[0], "{hits} vs {hits1}");
        assert!((flux - flux1).abs() <= 0.05 * flux1, "{flux} vs {flux1}");
    }
}
