//! The datasets behind the oc-wave validation figures (tools/wave_gallery.py).
//!
//! Every field here is computed through the crate's public API and compared
//! with a closed-form result from the literature. The closed forms are
//! evaluated with the crate's own special functions (`oc_wave::special`); the
//! plotting script cross-checks those against SciPy.
//!
//! ```text
//! cargo run -p oc-wave --release --example gallery -- <outdir>
//! ```
//!
//! For each dataset `<name>` the output directory receives `<name>.json`
//! (parameters, the method each step ran, the reports' warnings and ledgers,
//! the measured errors, and a description of every array) and one raw
//! little-endian `f64` file per array, `<name>__<array>.f64`, row-major with
//! the last index fastest.
//!
//! Sources (as cited in docs/wave-optics.md): Goodman, *Introduction to
//! Fourier Optics*, 4th ed. (2017); Matsushima & Shimobaba, Opt. Express 17,
//! 19662 (2009); Voelz & Roggemann, Appl. Opt. 48, 6132 (2009); Shen & Wang,
//! Appl. Opt. 45, 1102 (2006); Heurtley, JOSA 63, 1003 (1973); Kogelnik & Li,
//! Appl. Opt. 5, 1550 (1966); Self, Appl. Opt. 22, 658 (1983); Wen, Zhang &
//! Xiao, Adv. Opt. Photon. 5, 83 (2013); NIST DLMF ch. 7 and 10; Born & Wolf,
//! *Principles of Optics*, 7th ed. (1999). The modulator figures (F8, F9):
//! Popoff et al., J. Phys. Photonics 6, 043001 (2024); Texas Instruments
//! DLPA083B (2019, rev. 2023); Harvey & Vernold, Appl. Opt. 37, 8158 (1998);
//! Harvey et al., Proc. SPIE 8122, 812205 (2011); Swanson, MIT Lincoln Lab
//! TR-854 (1989); Zhang, You & Chu, Light Sci. Appl. 3, e213 (2014); Persson
//! et al., Opt. Express 20, 22334 (2012); JOSA A 21(10), 1996 (2004); Allen et
//! al., Phys. Rev. A 45, 8185 (1992).

use oc_wave::beams::{gaussian, gaussian_radius_mm, rayleigh_range_mm, GaussianBeam};
use oc_wave::czt::fraunhofer_zoom;
use oc_wave::elements::{apply_aperture, apply_thin_lens, apply_transmittance, Shape};
use oc_wave::propagate::critical_distance_mm;
use oc_wave::special::{bessel_j, fresnel_cs};
use oc_wave::{
    next_fast_len, propagate, AsmOptions, Cx, Fft2, Field2, Grid2, Propagator, SamplingWarning,
    WaveReport,
};
use oc_core::modulator::{
    dmd_projector_efficiency, facet_normal, on_axis_illumination, photopic_average, popoff_blazed_number, reflect,
    sinc as msinc, unit_cell_orders, DmdSpec, DmdState, LcosBeam, LcosSpec, ModulatorKind, OrderModel, Pattern,
    PatternGenerator, PhaseLut, SpatialModulator, TiltAxis, UnitCell,
};
use oc_core::Vec3;
use oc_wave::modulator::lcos_reflect;
use serde_json::{json, Map, Value};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, Box<dyn std::error::Error>>;

/// Zeros of J₁ (DLMF 10.21, Table 10.21.1): the Airy dark rings.
const J1_ZEROS: [f64; 4] = [
    3.831_705_970_207_512,
    7.015_586_669_815_619,
    10.173_468_135_062_722,
    13.323_691_936_314_223,
];

// ---------------------------------------------------------------------------
// Output plumbing
// ---------------------------------------------------------------------------

struct Dataset {
    dir: PathBuf,
    name: String,
    arrays: Map<String, Value>,
    t0: Instant,
}

impl Dataset {
    fn new(dir: &Path, name: &str) -> Self {
        eprintln!("[gallery] {name} ...");
        Self {
            dir: dir.to_path_buf(),
            name: name.to_string(),
            arrays: Map::new(),
            t0: Instant::now(),
        }
    }

    fn array(
        &mut self,
        key: &str,
        data: &[f64],
        shape: &[usize],
        axes: &[&str],
        units: &str,
        desc: &str,
    ) -> R<()> {
        let count: usize = shape.iter().product();
        if count != data.len() {
            return Err(format!("{}: array {key} has {} values, shape {shape:?}", self.name, data.len()).into());
        }
        if !data.iter().all(|v| v.is_finite()) {
            return Err(format!("{}: array {key} has a non-finite value", self.name).into());
        }
        let file = format!("{}__{}.f64", self.name, key);
        let mut bytes = Vec::with_capacity(8 * data.len());
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(self.dir.join(&file), bytes)?;
        self.arrays.insert(
            key.to_string(),
            json!({
                "file": file,
                "dtype": "<f8",
                "shape": shape,
                "order": "C (row-major, last index fastest)",
                "axes": axes,
                "units": units,
                "description": desc,
            }),
        );
        Ok(())
    }

    fn finish(self, mut meta: Value) -> R<f64> {
        let secs = self.t0.elapsed().as_secs_f64();
        let obj = meta.as_object_mut().ok_or("dataset metadata must be an object")?;
        obj.insert("dataset".into(), json!(self.name));
        obj.insert("generator".into(), json!("crates/oc-wave/examples/gallery.rs"));
        obj.insert("compute_seconds".into(), json!(secs));
        obj.insert("arrays".into(), Value::Object(self.arrays));
        let path = self.dir.join(format!("{}.json", self.name));
        std::fs::write(&path, serde_json::to_string_pretty(&meta)?)?;
        eprintln!("[gallery] {} done in {secs:.2} s", self.name);
        Ok(secs)
    }
}

/// Every warning a report carried, tagged with where it came from.
fn collect_warnings(ctx: &str, r: &WaveReport, acc: &mut Vec<Value>) {
    for w in &r.warnings {
        acc.push(json!({ "context": ctx, "warning": serde_json::to_value(w).unwrap_or(Value::Null) }));
    }
}

fn method_of(r: &WaveReport) -> String {
    r.steps
        .first()
        .map(|s| format!("{:?}", s.method))
        .unwrap_or_else(|| "none".into())
}

fn ledger_closure(r: &WaveReport) -> f64 {
    let l = &r.ledger;
    if l.launched > 0.0 {
        l.numerical.abs() / l.launched
    } else {
        l.numerical.abs()
    }
}

fn uniform(g: Grid2, lambda_um: f64) -> R<Field2> {
    Ok(Field2::from_fn(g, lambda_um, |_, _| Cx::ONE)?)
}

fn logspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| a * (b / a).powf(k as f64 / (n - 1) as f64))
        .collect()
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n).map(|k| a + (b - a) * k as f64 / (n - 1) as f64).collect()
}

fn fmax(it: impl Iterator<Item = f64>) -> f64 {
    it.fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Closed forms
// ---------------------------------------------------------------------------

/// `2J₁(v)/v` (Goodman; Born & Wolf), 1 at the origin.
fn airy_amp(v: f64) -> f64 {
    if v == 0.0 {
        1.0
    } else {
        2.0 * bessel_j(1, v) / v
    }
}

/// Heurtley (1973): the first Rayleigh–Sommerfeld solution on the axis of a
/// circular aperture of radius `a` in a unit plane wave,
/// `U(z) = e^{ikz} − (z/√(z² + a²)) e^{ik√(z² + a²)}`.
fn heurtley(lambda_um: f64, a: f64, z: f64) -> Cx {
    let k = 2.0 * PI / (lambda_um / 1000.0);
    let rr = (z * z + a * a).sqrt();
    Cx::expi(k * z) - Cx::expi(k * rr) * (z / rr)
}

/// Goodman's Fresnel integral for one axis of a `w`-wide slit at `z`:
/// `I(x) = [(C(α₂) − C(α₁)) + i(S(α₂) − S(α₁))]/√2`,
/// `α₁ = −√(2/λz)(w/2 + x)`, `α₂ = √(2/λz)(w/2 − x)`. The square aperture's
/// field is `e^{ikz}/i · I(x) I(y)`.
fn fresnel_slit(w: f64, lambda_mm: f64, z: f64, x: f64) -> Cx {
    let s = (2.0 / (lambda_mm * z)).sqrt();
    let (c1, s1) = fresnel_cs(-s * (0.5 * w + x));
    let (c2, s2) = fresnel_cs(s * (0.5 * w - x));
    Cx::new(c2 - c1, s2 - s1) / 2f64.sqrt()
}

fn sinc(t: f64) -> f64 {
    if t == 0.0 {
        1.0
    } else {
        (PI * t).sin() / (PI * t)
    }
}

/// `1/e²` radius from the second moment along x, `2√⟨(x − x̄)²⟩`, over the
/// whole 2D irradiance.
fn second_moment_radius_x(u: &Field2) -> f64 {
    let g = u.grid;
    let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let (x, _) = g.xy(i, j);
            let w = u.at(i, j).norm_sqr();
            s0 += w;
            s1 += w * x;
            s2 += w * x * x;
        }
    }
    let m = s1 / s0;
    2.0 * (s2 / s0 - m * m).sqrt()
}

/// Least-squares fit of `w² = c₀ + c₁t + c₂t²` (a Gaussian beam's radius is
/// exactly this in `t = z − z_ref`), returning (waist position t₀, waist w₀).
fn fit_waist(t: &[f64], w: &[f64]) -> (f64, f64) {
    // Normal equations, 3×3, by Cramer's rule.
    let mut s = [0.0f64; 5];
    let mut b = [0.0f64; 3];
    for (&ti, &wi) in t.iter().zip(w) {
        let y = wi * wi;
        let mut p = 1.0;
        for (k, sk) in s.iter_mut().enumerate() {
            *sk += p;
            if k < 3 {
                b[k] += p * y;
            }
            p *= ti;
        }
    }
    let m = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    let det3 = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det3(m);
    let mut c = [0.0; 3];
    for (col, ck) in c.iter_mut().enumerate() {
        let mut mm = m;
        for row in 0..3 {
            mm[row][col] = b[row];
        }
        *ck = det3(mm) / d;
    }
    let t0 = -c[1] / (2.0 * c[2]);
    let w0 = (c[0] - c[1] * c[1] / (4.0 * c[2])).sqrt();
    (t0, w0)
}

// ---------------------------------------------------------------------------
// F1: the Airy pattern
// ---------------------------------------------------------------------------

fn f1_airy(dir: &Path, fft: &mut Fft2) -> R<Value> {
    let mut ds = Dataset::new(dir, "f1_airy");
    let (lam, d, n, z) = (0.5, 0.2, 1024usize, 1000.0);
    let dx = d / 64.0;
    let lm = lam / 1000.0;
    let lz = lm * z;
    let k = 2.0 * PI / lm;
    let g = Grid2::square(n, dx)?;
    let mut u = uniform(g, lam)?;
    let mut ra = WaveReport::default();
    apply_aperture(&mut u, &Shape::Disc { c: [0.0, 0.0], r: 0.5 * d }, false, &mut ra)?;
    let area = PI * d * d / 4.0;
    let pk = area / lz; // analytic peak amplitude
    let i0 = pk * pk;
    let r1 = J1_ZEROS[0] * lz / (PI * d);
    let want = |x: f64, y: f64| {
        let r2 = x * x + y * y;
        let v = PI * d * r2.sqrt() / lz;
        Cx::expi(k * z + k * r2 / (2.0 * z)) / Cx::new(0.0, lz) * (area * airy_amp(v))
    };
    let mut warnings = Vec::new();

    // (a) the zoomed far field, ±4 first-zero radii, 401².
    let m = 401usize;
    let out = Grid2::square(m, 8.0 * r1 / (m - 1) as f64)?;
    let (v, rep) = fraunhofer_zoom(&u, z, out, fft)?;
    collect_warnings("fraunhofer_zoom 401²", &rep, &mut warnings);
    let img: Vec<f64> = v.data.iter().map(|c| c.norm_sqr() / i0).collect();
    let coords: Vec<f64> = (0..m).map(|i| out.coord(0, i)).collect();
    let mut worst_img: f64 = 0.0;
    for j in 0..m {
        for i in 0..m {
            let (x, y) = out.xy(i, j);
            worst_img = worst_img.max((v.at(i, j) - want(x, y)).abs() / pk);
        }
    }
    ds.array("image_intensity", &img, &[m, m], &["y", "x"], "|U|^2 / I0, I0 = (pi D^2 / 4 / (lambda z))^2", "simulated far-field irradiance on the zoomed 401x401 grid, normalized by the analytic peak")?;
    ds.array("image_coord_mm", &coords, &[m], &["x or y"], "mm", "sample coordinates of the zoomed grid (both axes)")?;

    // (b) the profile along +x through the centre.
    let jc = m / 2;
    let (mut pv, mut psim, mut pan, mut perr, mut pierr) = (vec![], vec![], vec![], vec![], vec![]);
    for i in m / 2..m {
        let x = out.coord(0, i);
        let vv = PI * d * x / lz;
        let s = v.at(i, jc);
        let a = want(x, 0.0);
        pv.push(vv);
        psim.push(s.norm_sqr() / i0);
        pan.push(a.norm_sqr() / i0);
        perr.push((s - a).abs() / pk);
        pierr.push((s.norm_sqr() - a.norm_sqr()).abs() / i0);
    }
    let worst_profile = fmax(perr.iter().copied());
    let worst_profile_int = fmax(pierr.iter().copied());
    // The simulated irradiance at the sample nearest each dark ring.
    let at_zeros: Vec<Value> = J1_ZEROS[..3]
        .iter()
        .map(|&vz| {
            let (idx, _) = pv
                .iter()
                .enumerate()
                .map(|(i, &x)| (i, (x - vz).abs()))
                .fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
            json!({ "v_zero": vz, "v_sample": pv[idx], "sim_intensity": psim[idx], "analytic_intensity": pan[idx] })
        })
        .collect();
    let dense_v = linspace(0.0, 4.0 * J1_ZEROS[0], 3001);
    let dense_i: Vec<f64> = dense_v.iter().map(|&v| airy_amp(v).powi(2)).collect();
    ds.array("profile_v", &pv, &[pv.len()], &["sample"], "1", "v = pi D r / (lambda z) of the samples along +x, y = 0")?;
    ds.array("profile_sim", &psim, &[psim.len()], &["sample"], "|U|^2 / I0", "simulated irradiance along +x")?;
    ds.array("profile_complex_err", &perr, &[perr.len()], &["sample"], "|U_sim - U_analytic| / |U_analytic(0)|", "complex error along +x")?;
    ds.array("dense_v", &dense_v, &[dense_v.len()], &["v"], "1", "v grid for the analytic curve")?;
    ds.array("dense_airy", &dense_i, &[dense_i.len()], &["v"], "1", "analytic [2 J1(v)/v]^2 (crate bessel_j)")?;

    // (c) encircled energy on a finer zoom, 801² over the same ±4 r1.
    let me = 801usize;
    let oe = Grid2::square(me, 8.0 * r1 / (me - 1) as f64)?;
    let (ve, repe) = fraunhofer_zoom(&u, z, oe, fft)?;
    collect_warnings("fraunhofer_zoom 801² (encircled energy)", &repe, &mut warnings);
    let da = oe.cell_area_mm2();
    let h = 0.5 * oe.pitch_mm[0];
    // (radius of the cell centre, power in the cell, x, y), sorted by radius.
    let mut pts: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(me * me);
    for j in 0..me {
        for i in 0..me {
            let (x, y) = oe.xy(i, j);
            pts.push(((x * x + y * y).sqrt(), ve.at(i, j).norm_sqr() * da, x, y));
        }
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut cum = Vec::with_capacity(pts.len());
    let mut acc = 0.0;
    for p in &pts {
        acc += p.1;
        cum.push(acc);
    }
    // Power inside the circle of radius r: cells wholly inside count in full;
    // a cell the circle crosses counts by its exact covered fraction
    // (`Shape::cell_overlap`), so the estimate has no pixel-staircase error.
    let half_diag = h * 2f64.sqrt();
    let ee_at = |r: f64| -> f64 {
        let inner = pts.partition_point(|p| p.0 + half_diag <= r);
        let outer = pts.partition_point(|p| p.0 - half_diag < r);
        let mut e = if inner == 0 { 0.0 } else { cum[inner - 1] };
        let disc = Shape::Disc { c: [0.0, 0.0], r };
        for p in &pts[inner..outer] {
            e += p.1 * disc.cell_overlap(p.2 - h, p.2 + h, p.3 - h, p.3 + h) / da;
        }
        e
    };
    let ee_v = linspace(0.0, 4.0 * J1_ZEROS[0], 401);
    let ee_sim: Vec<f64> = ee_v.iter().map(|&v| ee_at(v * lz / (PI * d)) / area).collect();
    let ee_an: Vec<f64> = ee_v
        .iter()
        .map(|&v| 1.0 - bessel_j(0, v).powi(2) - bessel_j(1, v).powi(2))
        .collect();
    let ee_ring = ee_at(r1) / area;
    let ee_ring_an = 1.0 - bessel_j(0, J1_ZEROS[0]).powi(2) - bessel_j(1, J1_ZEROS[0]).powi(2);
    let ee_ring_of_sampled = ee_at(r1) / u.power();
    let ee_worst = fmax(ee_sim.iter().zip(&ee_an).map(|(a, b)| (a - b).abs()));
    ds.array("ee_v", &ee_v, &[ee_v.len()], &["radius"], "1", "v of the encircling radius")?;
    ds.array("ee_sim", &ee_sim, &[ee_sim.len()], &["radius"], "fraction of pi D^2/4", "simulated encircled energy (sum of |U|^2 dA inside r on the 801x801 zoom) / aperture area")?;
    ds.array("ee_analytic", &ee_an, &[ee_an.len()], &["radius"], "1", "1 - J0(v)^2 - J1(v)^2 (crate bessel_j)")?;

    let errors = json!({
        "profile_worst_complex_err_rel_peak": worst_profile,
        "profile_worst_intensity_err_rel_peak": worst_profile_int,
        "image_worst_complex_err_rel_peak": worst_img,
        "intensity_at_dark_rings": at_zeros,
        "ee_first_ring_sim": ee_ring,
        "ee_first_ring_analytic": ee_ring_an,
        "ee_first_ring_abs_err": (ee_ring - ee_ring_an).abs(),
        "ee_first_ring_of_sampled_field_power": ee_ring_of_sampled,
        "ee_worst_abs_err_0_to_4r1": ee_worst,
    });
    eprintln!("  F1 errors: {errors}");
    let meta = json!({
        "title": "Airy pattern of a circular aperture",
        "parameters": {
            "D_mm": d, "lambda_um": lam, "dx_mm": dx, "D_over_dx": 64, "n": n, "z_mm": z,
            "window_mm": n as f64 * dx, "first_zero_radius_mm": r1,
            "source_fresnel_number_r2_over_lambda_z": (0.5 * d).powi(2) / lz,
            "zoom_image": { "n": m, "pitch_mm": out.pitch_mm[0], "half_width_mm": 4.0 * r1 },
            "zoom_encircled_energy": { "n": me, "pitch_mm": oe.pitch_mm[0] },
            "aperture_power_area_mm2": area,
            "sampled_field_power": u.power(),
        },
        "method": {
            "aperture": "apply_aperture(Shape::Disc), exact area-weighted cell coverage",
            "propagation": method_of(&rep),
            "call": "oc_wave::czt::fraunhofer_zoom (Bluestein chirp-z, separable)",
            "step_note": rep.steps.first().map(|s| s.note.clone()),
        },
        "aperture_ledger": serde_json::to_value(ra.ledger)?,
        "reference": "U = exp(ikz) exp(ik r^2/2z)/(i lambda z) * (pi D^2/4) * 2 J1(v)/v, v = pi D r/(lambda z); EE = 1 - J0^2 - J1^2",
        "sources": ["Goodman", "Born & Wolf", "DLMF"],
        "warnings": warnings,
        "errors": errors,
    });
    ds.finish(meta)?;
    Ok(errors)
}

// ---------------------------------------------------------------------------
// F2: Heurtley's exact on-axis field
// ---------------------------------------------------------------------------

fn disc_field(n: usize, dx: f64, lam: f64, a: f64) -> R<(Field2, WaveReport)> {
    let mut u = uniform(Grid2::square(n, dx)?, lam)?;
    let mut r = WaveReport::default();
    apply_aperture(&mut u, &Shape::Disc { c: [0.0, 0.0], r: a }, false, &mut r)?;
    Ok((u, r))
}

fn f2_heurtley(dir: &Path, fft: &mut Fft2) -> R<Value> {
    let mut ds = Dataset::new(dir, "f2_heurtley");
    let (lam, a) = (0.5, 0.1);
    let lm = lam / 1000.0;
    let mut warnings = Vec::new();

    // (a) 20 log-spaced Fresnel numbers from 20 to 0.2 at a/Δx = 256.
    let nfs = logspace(20.0, 0.2, 20);
    let (fine, _) = disc_field(1024, a / 256.0, lam, a)?;
    let (mut zs, mut sim_re, mut sim_im, mut an_re, mut an_im, mut err) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut methods = Vec::new();
    let mut closure: f64 = 0.0;
    for &nf in &nfs {
        let z = a * a / (lm * nf);
        let (v, r) = propagate(&fine, z, Propagator::RsDirect, fft)?;
        collect_warnings(&format!("RsDirect a/dx=256, N_F={nf:.4}"), &r, &mut warnings);
        methods.push(method_of(&r));
        closure = closure.max(ledger_closure(&r));
        let s = v.at(512, 512);
        let h = heurtley(lam, a, z);
        zs.push(z);
        sim_re.push(s.re);
        sim_im.push(s.im);
        an_re.push(h.re);
        an_im.push(h.im);
        err.push((s - h).abs());
    }
    let worst_a = fmax(err.iter().copied());
    let worst_int = fmax(
        sim_re
            .iter()
            .zip(&sim_im)
            .zip(an_re.iter().zip(&an_im))
            .map(|((sr, si), (ar, ai))| ((sr * sr + si * si) - (ar * ar + ai * ai)).abs()),
    );
    ds.array("nf", &nfs, &[nfs.len()], &["point"], "1", "Fresnel number a^2/(lambda z) of the simulated points")?;
    ds.array("z_mm", &zs, &[zs.len()], &["point"], "mm", "distance of the simulated points")?;
    ds.array("sim_re", &sim_re, &[nfs.len()], &["point"], "unit-plane-wave amplitude", "Re U(0, z), direct Rayleigh-Sommerfeld, a/dx = 256")?;
    ds.array("sim_im", &sim_im, &[nfs.len()], &["point"], "unit-plane-wave amplitude", "Im U(0, z)")?;
    ds.array("heurtley_re", &an_re, &[nfs.len()], &["point"], "unit-plane-wave amplitude", "Re of Heurtley's exact value at the same z")?;
    ds.array("heurtley_im", &an_im, &[nfs.len()], &["point"], "unit-plane-wave amplitude", "Im of Heurtley's exact value")?;
    ds.array("abs_err", &err, &[nfs.len()], &["point"], "unit-plane-wave amplitude", "|U_sim - U_Heurtley|")?;

    let dense_nf = logspace(20.0, 0.2, 6001);
    let dense_exact: Vec<f64> = dense_nf
        .iter()
        .map(|&nf| heurtley(lam, a, a * a / (lm * nf)).norm_sqr())
        .collect();
    let dense_par: Vec<f64> = dense_nf
        .iter()
        .map(|&nf| 4.0 * (0.5 * PI * nf).sin().powi(2))
        .collect();
    ds.array("dense_nf", &dense_nf, &[dense_nf.len()], &["nf"], "1", "Fresnel numbers of the analytic curves")?;
    ds.array("dense_heurtley_int", &dense_exact, &[dense_nf.len()], &["nf"], "|U|^2 / |U_inc|^2", "Heurtley's exact on-axis irradiance")?;
    ds.array("dense_paraxial_int", &dense_par, &[dense_nf.len()], &["nf"], "|U|^2 / |U_inc|^2", "paraxial (Fresnel) on-axis irradiance 4 sin^2(pi N_F / 2)")?;
    // How far the paraxial field e^{ikz}(1 − e^{iπN_F}) is from Heurtley's,
    // with the common e^{ikz} removed and k(R − z) written as ka²/(R + z),
    // so neither side loses digits to the absolute phase.
    let dense_par_err: Vec<f64> = dense_nf
        .iter()
        .map(|&nf| {
            let z = a * a / (lm * nf);
            let rr = (z * z + a * a).sqrt();
            let k = 2.0 * PI / lm;
            (Cx::expi(PI * nf) - Cx::expi(k * a * a / (rr + z)) * (z / rr)).abs()
        })
        .collect();
    ds.array("dense_paraxial_abs_err", &dense_par_err, &[dense_nf.len()], &["nf"], "unit-plane-wave amplitude", "|U_Fresnel - U_Heurtley| on axis, U_Fresnel = exp(ikz)(1 - exp(i pi N_F))")?;

    // (b) convergence in a/Δx, and Richardson extrapolation.
    let levels = [32usize, 64, 128, 256];
    let conv_nf = [10.0, 5.0, 2.5];
    let mut conv = vec![0.0; conv_nf.len() * levels.len()];
    let mut rich = vec![0.0; conv_nf.len() * (levels.len() - 1)];
    let mut vals = vec![Cx::ZERO; conv_nf.len() * levels.len()];
    for (li, &m) in levels.iter().enumerate() {
        let n = 4 * m;
        let (u, _) = disc_field(n, a / m as f64, lam, a)?;
        for (ni, &nf) in conv_nf.iter().enumerate() {
            let z = a * a / (lm * nf);
            let (v, r) = propagate(&u, z, Propagator::RsDirect, fft)?;
            collect_warnings(&format!("RsDirect a/dx={m}, N_F={nf}"), &r, &mut warnings);
            let s = v.at(n / 2, n / 2);
            vals[ni * levels.len() + li] = s;
            conv[ni * levels.len() + li] = (s - heurtley(lam, a, z)).abs();
        }
    }
    let mut orders = Vec::new();
    for (ni, &nf) in conv_nf.iter().enumerate() {
        let z = a * a / (lm * nf);
        let h = heurtley(lam, a, z);
        for li in 1..levels.len() {
            let uf = vals[ni * levels.len() + li];
            let uc = vals[ni * levels.len() + li - 1];
            let ur = (uf * 4.0 - uc) / 3.0;
            rich[ni * (levels.len() - 1) + li - 1] = (ur - h).abs();
        }
        let ratios: Vec<f64> = (1..levels.len())
            .map(|li| conv[ni * levels.len() + li - 1] / conv[ni * levels.len() + li])
            .collect();
        orders.push(json!({ "nf": nf, "error_ratio_per_halving": ratios,
            "observed_order": ratios.iter().map(|r| r.log2()).collect::<Vec<_>>() }));
    }
    let lv: Vec<f64> = levels.iter().map(|&m| m as f64).collect();
    ds.array("conv_a_over_dx", &lv, &[levels.len()], &["level"], "1", "a/dx of the convergence levels (n = 4 a/dx)")?;
    ds.array("conv_nf", &conv_nf, &[conv_nf.len()], &["case"], "1", "Fresnel numbers of the convergence study")?;
    ds.array("conv_abs_err", &conv, &[conv_nf.len(), levels.len()], &["case", "level"], "unit-plane-wave amplitude", "|U_sim - U_Heurtley| on axis, direct RS")?;
    ds.array("rich_abs_err", &rich, &[conv_nf.len(), levels.len() - 1], &["case", "level (finer of the pair: 64, 128, 256)"], "unit-plane-wave amplitude", "|(4U(dx/2) - U(dx))/3 - U_Heurtley|")?;

    let errors = json!({
        "points_worst_abs_err_a_dx_256": worst_a,
        "points_worst_intensity_err_a_dx_256": worst_int,
        "convergence": orders,
        "conv_abs_err": conv,
        "richardson_abs_err": rich,
        "richardson_worst": fmax(rich.iter().copied()),
        "ledger_worst_numerical_fraction": closure,
    });
    eprintln!("  F2 errors: {errors}");
    let meta = json!({
        "title": "Heurtley's exact on-axis field of a circular aperture",
        "parameters": { "a_mm": a, "lambda_um": lam, "fine": { "a_over_dx": 256, "n": 1024, "window_mm": 4.0 * a },
            "nf_range": [20.0, 0.2], "z_range_mm": [a * a / (lm * 20.0), a * a / (lm * 0.2)],
            "convergence_levels_a_over_dx": levels, "convergence_n": levels.iter().map(|m| 4 * m).collect::<Vec<_>>() },
        "method": { "propagation": methods.first().cloned(), "all_methods": methods,
            "aperture": "apply_aperture(Shape::Disc), exact area-weighted cell coverage",
            "sampling_note": "the direct-RS Undersampled warning tests the kernel at the window's largest offset (n-1)dx; the on-axis value only uses offsets <= a, where the kernel's local frequency a/(lambda sqrt(z^2+a^2)) is below 1/(2dx) at every level" },
        "reference": "Heurtley 1973: U(0,z) = exp(ikz) - z/sqrt(z^2+a^2) exp(ik sqrt(z^2+a^2)); paraxial |U|^2 = 4 sin^2(pi N_F/2)",
        "sources": ["Heurtley 1973", "Shen 2006", "Goodman"],
        "warnings": warnings,
        "errors": errors,
    });
    ds.finish(meta)?;
    Ok(errors)
}

// ---------------------------------------------------------------------------
// F3: Fresnel diffraction of a square
// ---------------------------------------------------------------------------

struct SquareRun {
    n: usize,
    dx: f64,
    method: String,
    zc: f64,
    field: Field2,
    analytic_line: Vec<Cx>, // I(x) per grid coordinate
    warnings: Vec<Value>,
    closure: f64,
    ledger: Value,
}

/// A `w`-wide square, `m` samples across (edges on cell boundaries), carried
/// `z` by the Fresnel method in its regime (Voelz & Roggemann 2009).
fn run_square(m: usize, n: usize, w: f64, lam: f64, z: f64, fft: &mut Fft2) -> R<SquareRun> {
    let lm = lam / 1000.0;
    let dx = w / m as f64;
    let g = Grid2::square(n, dx)?;
    let mut u = uniform(g, lam)?;
    let mut r0 = WaveReport::default();
    apply_aperture(&mut u, &Shape::Rect { c: [0.0, 0.0], h: [0.5 * w, 0.5 * w] }, false, &mut r0)?;
    let lp = next_fast_len(2 * n) as f64 * dx;
    let zc = critical_distance_mm(dx, lp, lm);
    let how = if z <= zc { Propagator::FresnelTf } else { Propagator::FresnelIr };
    let (v, rep) = propagate(&u, z, how, fft)?;
    let mut warnings = Vec::new();
    collect_warnings(&format!("{how:?}, m={m}, n={n}, z={z}"), &rep, &mut warnings);
    if rep.warnings.iter().any(|w| matches!(w, SamplingWarning::OutsideRegime { .. })) {
        return Err(format!("square m={m}: {how:?} ran outside its regime").into());
    }
    let analytic_line: Vec<Cx> = (0..n).map(|i| fresnel_slit(w, lm, z, g.coord(0, i))).collect();
    Ok(SquareRun {
        n,
        dx,
        method: method_of(&rep),
        zc,
        field: v,
        analytic_line,
        warnings,
        closure: ledger_closure(&rep),
        ledger: serde_json::to_value(rep.ledger)?,
    })
}

fn f3_fresnel_square(dir: &Path, fft: &mut Fft2) -> R<Value> {
    let mut ds = Dataset::new(dir, "f3_fresnel_square");
    let (lam, w, z) = (0.5, 0.5, 50.0);
    let lm = lam / 1000.0;
    let k = 2.0 * PI / lm;
    let pre = Cx::expi(k * z) / Cx::I;
    let mut warnings = Vec::new();
    let mut runs_meta = Vec::new();
    let mut fine_errors = Value::Null;
    for m in [201usize, 101] {
        let n = 5 * m + (5 * m) % 2;
        let run = run_square(m, n, w, lam, z, fft)?;
        warnings.extend(run.warnings.iter().cloned());
        let ix = &run.analytic_line;
        let pk = fmax(ix.iter().map(|c| c.abs())).powi(2); // max |U| over the plane
        let g = run.field.grid;
        let mut worst2d: f64 = 0.0;
        for j in 0..n {
            for i in 0..n {
                let want = pre * ix[i] * ix[j];
                worst2d = worst2d.max((run.field.at(i, j) - want).abs() / pk);
            }
        }
        let jc = n / 2;
        let (mut xs, mut isim, mut ian, mut err) = (vec![], vec![], vec![], vec![]);
        for i in 0..n {
            let want = pre * ix[i] * ix[jc];
            let s = run.field.at(i, jc);
            xs.push(g.coord(0, i));
            isim.push(s.norm_sqr());
            ian.push(want.norm_sqr());
            err.push((s - want).abs() / pk);
        }
        let worst_line = fmax(err.iter().copied());
        let worst_line_int = fmax(isim.iter().zip(&ian).map(|(a, b)| (a - b).abs())) / (pk * pk);
        let tag = format!("m{m}");
        ds.array(&format!("{tag}_line_x_mm"), &xs, &[n], &["x"], "mm", "x of the samples along y = 0")?;
        ds.array(&format!("{tag}_line_sim"), &isim, &[n], &["x"], "|U|^2 / |U_inc|^2", "simulated irradiance along y = 0")?;
        ds.array(&format!("{tag}_line_analytic"), &ian, &[n], &["x"], "|U|^2 / |U_inc|^2", "Fresnel-integral irradiance along y = 0 (crate fresnel_cs)")?;
        ds.array(&format!("{tag}_line_complex_err"), &err, &[n], &["x"], "|dU| / max|U_analytic|", "complex error along y = 0")?;
        if m == 201 {
            // The irradiance map, cropped to |x|, |y| ≤ 0.5 mm.
            let idx: Vec<usize> = (0..n).filter(|&i| g.coord(0, i).abs() <= 0.5 + 1e-12).collect();
            let mut img = Vec::with_capacity(idx.len() * idx.len());
            for &j in &idx {
                for &i in &idx {
                    img.push(run.field.at(i, j).norm_sqr());
                }
            }
            let cx: Vec<f64> = idx.iter().map(|&i| g.coord(0, i)).collect();
            ds.array("image_intensity", &img, &[idx.len(), idx.len()], &["y", "x"], "|U|^2 / |U_inc|^2", "simulated irradiance, m = 201, cropped to |x|,|y| <= 0.5 mm")?;
            ds.array("image_coord_mm", &cx, &[cx.len()], &["x or y"], "mm", "coordinates of the cropped image (both axes)")?;
            fine_errors = json!({
                "worst_complex_err_rel_peak_2d": worst2d,
                "worst_complex_err_rel_peak_line": worst_line,
                "worst_intensity_err_rel_peak_line": worst_line_int,
            });
        }
        runs_meta.push(json!({
            "samples_across": m, "n": run.n, "dx_mm": run.dx, "window_mm": run.n as f64 * run.dx,
            "padded_window_mm": next_fast_len(2 * run.n) as f64 * run.dx, "critical_distance_mm": run.zc,
            "method": run.method, "worst_complex_err_rel_peak_2d": worst2d,
            "worst_complex_err_rel_peak_line": worst_line, "worst_intensity_err_rel_peak_line": worst_line_int,
            "ledger_numerical_fraction": run.closure,
            "ledger": run.ledger,
        }));
    }
    let ratio = runs_meta[1]["worst_complex_err_rel_peak_2d"].as_f64().unwrap_or(f64::NAN)
        / runs_meta[0]["worst_complex_err_rel_peak_2d"].as_f64().unwrap_or(f64::NAN);
    let errors = json!({ "m201": fine_errors, "error_ratio_101_to_201_2d": ratio, "runs": runs_meta });
    eprintln!("  F3 errors: {errors}");
    let meta = json!({
        "title": "Fresnel diffraction of a square aperture",
        "parameters": { "w_mm": w, "lambda_um": lam, "z_mm": z, "fresnel_number_half_width": (0.5 * w).powi(2) / (lm * z) },
        "method": { "rule": "transfer function if z <= z_c = L dx / lambda (L = padded window), impulse response beyond (Voelz & Roggemann 2009)",
            "aperture": "apply_aperture(Shape::Rect), edges on cell boundaries (odd samples across)" },
        "reference": "U = exp(ikz)/i I(x) I(y), I(x) = [(C(a2)-C(a1)) + i(S(a2)-S(a1))]/sqrt2, a1 = -sqrt(2/(lambda z))(w/2+x), a2 = sqrt(2/(lambda z))(w/2-x)",
        "sources": ["Goodman", "Voelz 2009", "DLMF"],
        "warnings": warnings,
        "errors": errors,
    });
    ds.finish(meta)?;
    Ok(errors)
}

// ---------------------------------------------------------------------------
// F4: from Fresnel to Fraunhofer
// ---------------------------------------------------------------------------

fn f4_fresnel_to_fraunhofer(dir: &Path, fft: &mut Fft2) -> R<Value> {
    let mut ds = Dataset::new(dir, "f4_fresnel_to_fraunhofer");
    let (lam, w): (f64, f64) = (0.5, 0.5);
    let lm = lam / 1000.0;
    let k = 2.0 * PI / lm;
    let nfs: [f64; 6] = [10.0, 2.5, 1.0, 0.4, 0.1, 0.02];
    let n = 2048usize;
    let mut warnings = Vec::new();
    let mut cases = Vec::new();
    for (ci, &nf) in nfs.iter().enumerate() {
        let z = (0.5 * w).powi(2) / (lm * nf);
        let lzw = lm * z / w;
        // Display: the geometric half-width plus 2.5 diffraction widths λz/w,
        // at least 0.4 mm. The window holds it with a 10% margin, and the
        // aperture gets as many samples across as n = 2048 then allows (odd,
        // so its edges fall on cell boundaries).
        let xd = (0.5 * w + 2.5 * lzw).max(0.4);
        let mut m = (n as f64 * w / (2.0 * 1.1 * xd)).floor() as usize;
        if m % 2 == 0 {
            m -= 1;
        }
        let run = run_square(m, n, w, lam, z, fft)?;
        warnings.extend(run.warnings.iter().cloned());
        let g = run.field.grid;
        let jc = n / 2;
        let i0 = run.analytic_line[jc];
        let pre = Cx::expi(k * z) / Cx::I;
        let idx: Vec<usize> = (0..n).filter(|&i| g.coord(0, i).abs() <= xd + 1e-12).collect();
        let line_an: Vec<Cx> = idx.iter().map(|&i| pre * run.analytic_line[i] * i0).collect();
        let pk = fmax(line_an.iter().map(|c| c.abs()));
        let imax = pk * pk;
        let (mut xs, mut isim, mut ian, mut ifr, mut err) = (vec![], vec![], vec![], vec![], vec![]);
        // Fraunhofer: U = e^{ikz}e^{ikx²/2z}/(iλz) · w² sinc(wx/λz) sinc(0).
        let lz = lm * z;
        for (q, &i) in idx.iter().enumerate() {
            let x = g.coord(0, i);
            let s = run.field.at(i, jc);
            xs.push(x);
            isim.push(s.norm_sqr() / imax);
            ian.push(line_an[q].norm_sqr() / imax);
            ifr.push((w * w / lz * sinc(w * x / lz)).powi(2) / imax);
            err.push((s - line_an[q]).abs() / pk);
        }
        let worst = fmax(err.iter().copied());
        let worst_int = fmax(isim.iter().zip(&ian).map(|(a, b)| (a - b).abs()));
        let fr_vs_fresnel = fmax(ian.iter().zip(&ifr).map(|(a, b)| (a - b).abs()));
        let tag = format!("c{ci}");
        ds.array(&format!("{tag}_x_mm"), &xs, &[xs.len()], &["x"], "mm", "x along y = 0")?;
        ds.array(&format!("{tag}_sim"), &isim, &[xs.len()], &["x"], "|U|^2 / max|U_analytic|^2 on the line", "simulated irradiance along y = 0")?;
        ds.array(&format!("{tag}_analytic"), &ian, &[xs.len()], &["x"], "same normalization", "Fresnel-integral irradiance (crate fresnel_cs)")?;
        ds.array(&format!("{tag}_fraunhofer"), &ifr, &[xs.len()], &["x"], "same normalization", "Fraunhofer limit (w^2/(lambda z))^2 sinc^2(w x/(lambda z))")?;
        ds.array(&format!("{tag}_complex_err"), &err, &[xs.len()], &["x"], "|dU| / max|U_analytic| on the line", "complex error along y = 0")?;
        cases.push(json!({
            "index": ci, "nf": nf, "z_mm": z, "lambda_z_over_w_mm": lzw, "display_half_range_mm": xd,
            "samples_across": m, "n": n, "dx_mm": run.dx, "window_mm": n as f64 * run.dx,
            "padded_window_mm": next_fast_len(2 * n) as f64 * run.dx, "critical_distance_mm": run.zc,
            "z_over_zc": z / run.zc, "method": run.method,
            "worst_complex_err_rel_peak_line": worst, "worst_intensity_err_rel_peak_line": worst_int,
            "fraunhofer_vs_fresnel_max_intensity_diff_rel_peak": fr_vs_fresnel,
            "ledger_numerical_fraction": run.closure,
            "ledger": run.ledger,
        }));
        eprintln!("  F4 N_F={nf}: m={m}, method {}, worst {worst:.3e}, intensity {worst_int:.3e}", run.method);
    }
    let errors = json!({ "cases": cases });
    let meta = json!({
        "title": "From Fresnel to Fraunhofer: a square aperture",
        "parameters": { "w_mm": w, "lambda_um": lam, "fresnel_numbers_half_width": nfs, "n": n },
        "method": { "rule": "Fresnel transfer function if z <= z_c, impulse response beyond (Voelz & Roggemann 2009); window and samples per case as listed",
            "aperture": "apply_aperture(Shape::Rect), edges on cell boundaries" },
        "reference": "Goodman: U = exp(ikz)/i I(x) I(y) through the Fresnel integrals; Fraunhofer limit w^2/(lambda z) sinc(w x/(lambda z)) sinc(w y/(lambda z))",
        "sources": ["Goodman", "Voelz 2009", "DLMF"],
        "warnings": warnings,
        "errors": errors,
    });
    ds.finish(meta)?;
    Ok(errors)
}

// ---------------------------------------------------------------------------
// F5: a Gaussian beam through a thin lens
// ---------------------------------------------------------------------------

fn f5_gaussian_lens(dir: &Path, fft: &mut Fft2) -> R<Value> {
    let mut ds = Dataset::new(dir, "f5_gaussian_lens");
    let (lam, w0, f, s) = (0.6328, 0.5, 200.0, 100.0);
    let lm = lam / 1000.0;
    let zr = rayleigh_range_mm(w0, lam, 1.0);
    // Self (1983).
    let s2 = 1.0 / (1.0 / f - 1.0 / (s + zr * zr / (s - f)));
    let w2 = w0 / ((1.0 - s / f).powi(2) + (zr / f).powi(2)).sqrt();
    let zr2 = PI * w2 * w2 / lm;
    // Cross-check through the q-parameter (q = z − i z_R, a lens maps 1/q to 1/q − 1/f).
    let q_after = Cx::ONE / (Cx::ONE / Cx::new(s, -zr) - Cx::real(1.0 / f));
    let s2_q = -q_after.re;
    let w2_q = (-q_after.im * lm / PI).sqrt();

    let g = Grid2::square(512, 0.008)?;
    let u0 = gaussian(g, lam, 1.0, w0, 0.0, 1.0, [0.0, 0.0])?;
    let how = Propagator::AngularSpectrum(AsmOptions::default());
    let mut warnings = Vec::new();
    let (mut ul, rl) = propagate(&u0, s, how, fft)?;
    collect_warnings("ASM to the lens", &rl, &mut warnings);
    let mut rlens = WaveReport::default();
    apply_thin_lens(&mut ul, f, [0.0, 0.0], &mut rlens)?;
    collect_warnings("thin lens", &rlens, &mut warnings);

    // Planes: 2 mm steps to the lens, then s″/100 steps to s + 1.6 s″, so the
    // lens and the predicted waist are both sampled.
    let mut zs: Vec<f64> = (0..=50).map(|q| s * q as f64 / 50.0).collect();
    zs.extend((1..=160).map(|q| s + s2 * q as f64 / 100.0));
    let nx = g.n[0];
    let jc = g.n[1] / 2;
    let mut slice = Vec::with_capacity(zs.len() * nx);
    let (mut wsim, mut wan) = (vec![], vec![]);
    let mut closure: f64 = 0.0;
    let mut max_removed: f64 = 0.0;
    let mut methods = std::collections::BTreeSet::new();
    // The paraxial control: the same planes by the Fresnel method in its
    // regime (transfer function to z_c, impulse response beyond). A Gaussian
    // times a parabolic lens phase is an exact Fresnel solution, so Self's
    // formulas hold for it exactly; whatever residual the exact angular
    // spectrum shows beyond this one is non-paraxial.
    let zc_f = critical_distance_mm(g.pitch_mm[0], next_fast_len(2 * g.n[0]) as f64 * g.pitch_mm[0], lm);
    let mut wpar = Vec::new();
    let mut methods_par = std::collections::BTreeSet::new();
    for &z in &zs {
        let (src, dz) = if z <= s { (&u0, z) } else { (&ul, z - s) };
        let hp = if dz <= zc_f { Propagator::FresnelTf } else { Propagator::FresnelIr };
        let (vp, rp) = propagate(src, dz, hp, fft)?;
        collect_warnings(&format!("paraxial control {hp:?} z={z:.3}"), &rp, &mut warnings);
        methods_par.insert(method_of(&rp));
        wpar.push(second_moment_radius_x(&vp));
    }
    for &z in &zs {
        let (v, r) = if z <= s {
            propagate(&u0, z, how, fft)?
        } else {
            propagate(&ul, z - s, how, fft)?
        };
        collect_warnings(&format!("ASM z={z:.3}"), &r, &mut warnings);
        methods.insert(method_of(&r));
        closure = closure.max(ledger_closure(&r));
        max_removed = max_removed.max(r.ledger.band_limit_removed / r.ledger.launched);
        for i in 0..nx {
            slice.push(v.at(i, jc).norm_sqr());
        }
        wsim.push(second_moment_radius_x(&v));
        wan.push(if z <= s {
            gaussian_radius_mm(w0, lam, 1.0, z)
        } else {
            w2 * (1.0 + ((z - s - s2) / zr2).powi(2)).sqrt()
        });
    }
    let rel: Vec<f64> = wsim.iter().zip(&wan).map(|(a, b)| (a - b) / b).collect();
    let rel_par: Vec<f64> = wpar.iter().zip(&wan).map(|(a, b)| (a - b) / b).collect();
    let worst_rel_par = fmax(rel_par.iter().map(|r| r.abs()));
    let (tp, wp): (Vec<f64>, Vec<f64>) = zs
        .iter()
        .zip(&wpar)
        .filter(|(z, _)| **z > s)
        .map(|(z, w)| (*z - s - s2, *w))
        .unzip();
    let (t0_par, w0_par) = fit_waist(&tp, &wp);
    ds.array("w_rel_residual_paraxial", &rel_par, &[zs.len()], &["plane"], "1", "(w_Fresnel - w_analytic) / w_analytic, the paraxial control (Fresnel TF to z_c, IR beyond)")?;
    let worst_rel = fmax(rel.iter().map(|r| r.abs()));
    let worst_before = fmax(zs.iter().zip(&rel).filter(|(z, _)| **z <= s).map(|(_, r)| r.abs()));
    let worst_after = fmax(zs.iter().zip(&rel).filter(|(z, _)| **z > s).map(|(_, r)| r.abs()));
    let iw = zs.iter().position(|&z| (z - (s + s2)).abs() < 1e-9).ok_or("waist plane missing")?;
    let waist_rel = rel[iw];
    // Measured waist: the parabola w² = c₀ + c₁t + c₂t² through every plane after the lens.
    let (ta, wa): (Vec<f64>, Vec<f64>) = zs
        .iter()
        .zip(&wsim)
        .filter(|(z, _)| **z > s)
        .map(|(z, w)| (*z - s - s2, *w))
        .unzip();
    let (t0, w0_fit) = fit_waist(&ta, &wa);
    // Phase flatness at the waist: arg U(w₀″, 0) − arg U(0, 0).
    let (vw, _) = propagate(&ul, s2, how, fft)?;
    let c = g.n[0] / 2;
    let i_edge = c + (w2 / g.pitch_mm[0]).round() as usize;
    let dph = (vw.at(i_edge, c) * Cx::expi(-vw.at(c, c).arg())).arg().abs();
    let xs: Vec<f64> = (0..nx).map(|i| g.coord(0, i)).collect();
    ds.array("z_mm", &zs, &[zs.len()], &["plane"], "mm", "distance from the input waist")?;
    ds.array("x_mm", &xs, &[nx], &["x"], "mm", "x of the samples")?;
    ds.array("slice_intensity", &slice, &[zs.len(), nx], &["plane", "x"], "|U|^2, beam of unit power", "irradiance along y = 0 at each plane (unnormalized)")?;
    ds.array("w_sim_mm", &wsim, &[zs.len()], &["plane"], "mm", "2 sqrt(<x^2>) from the full 2D irradiance")?;
    ds.array("w_analytic_mm", &wan, &[zs.len()], &["plane"], "mm", "Kogelnik-Li w(z) before the lens; Self's transformed beam after")?;
    ds.array("w_rel_residual", &rel, &[zs.len()], &["plane"], "1", "(w_sim - w_analytic) / w_analytic")?;
    let dense_z = linspace(0.0, s + 1.6 * s2, 2001);
    let dense_w: Vec<f64> = dense_z
        .iter()
        .map(|&z| if z <= s { gaussian_radius_mm(w0, lam, 1.0, z) } else { w2 * (1.0 + ((z - s - s2) / zr2).powi(2)).sqrt() })
        .collect();
    ds.array("dense_z_mm", &dense_z, &[dense_z.len()], &["z"], "mm", "z of the analytic envelope")?;
    ds.array("dense_w_mm", &dense_w, &[dense_z.len()], &["z"], "mm", "analytic w(z)")?;

    let errors = json!({
        "radius_worst_rel_residual": worst_rel,
        "radius_worst_rel_residual_before_lens": worst_before,
        "radius_worst_rel_residual_after_lens": worst_after,
        "radius_rel_residual_at_predicted_waist": waist_rel,
        "fitted_waist_position_minus_self_mm": t0,
        "fitted_waist_radius_mm": w0_fit,
        "fitted_waist_radius_rel_err": (w0_fit - w2) / w2,
        "waist_phase_at_w0pp_rad": dph,
        "paraxial_control_worst_rel_residual": worst_rel_par,
        "paraxial_control_fitted_waist_position_minus_self_mm": t0_par,
        "paraxial_control_fitted_waist_radius_rel_err": (w0_par - w2) / w2,
        "paraxial_control_methods": methods_par.iter().cloned().collect::<Vec<_>>(),
        "paraxial_control_critical_distance_mm": zc_f,
        "lens_plane_w2_over_2f_mm": gaussian_radius_mm(w0, lam, 1.0, s).powi(2) / (2.0 * f),
        "self_vs_q_parameter_s2_rel": (s2 - s2_q).abs() / s2,
        "self_vs_q_parameter_w2_rel": (w2 - w2_q).abs() / w2,
        "band_limit_removed_max_fraction": max_removed,
        "ledger_worst_numerical_fraction": closure,
    });
    eprintln!("  F5 errors: {errors}");
    let meta = json!({
        "title": "Gaussian beam through a thin lens",
        "parameters": { "w0_mm": w0, "lambda_um": lam, "s_mm": s, "f_mm": f, "zR_mm": zr,
            "grid_n": 512, "dx_mm": 0.008, "window_mm": 512.0 * 0.008,
            "s2_mm": s2, "w0pp_mm": w2, "zRpp_mm": zr2, "lens_z_mm": s, "waist_z_mm": s + s2, "z_end_mm": s + 1.6 * s2,
            "critical_distance_mm": critical_distance_mm(0.008, next_fast_len(1024) as f64 * 0.008, lm) },
        "method": { "propagation": methods.into_iter().collect::<Vec<_>>(),
            "note": "each plane is one step from the input waist (z <= s) or from the lens plane (z > s), never accumulated",
            "lens": "apply_thin_lens, t = exp(-ik r^2 / 2f)" },
        "reference": "Kogelnik & Li 1966: w(z) = w0 sqrt(1+(z/zR)^2); Self 1983: 1/(s + zR^2/(s-f)) + 1/s'' = 1/f, w0'' = w0 / sqrt((1-s/f)^2 + (zR/f)^2)",
        "sources": ["Kogelnik 1966", "Self 1983", "Matsushima 2009", "Goodman"],
        "warnings": warnings,
        "errors": errors,
    });
    ds.finish(meta)?;
    Ok(errors)
}

// ---------------------------------------------------------------------------
// F6: the Talbot carpet
// ---------------------------------------------------------------------------

/// Length of `[a, b] ∩ ⋃ₖ [kp − h, kp + h]`.
fn grating_cover(a: f64, b: f64, p: f64, h: f64) -> f64 {
    let k0 = ((a - h) / p).floor() as i64 - 1;
    let k1 = ((b + h) / p).ceil() as i64 + 1;
    (k0..=k1)
        .map(|k| {
            let c = k as f64 * p;
            ((b.min(c + h)) - (a.max(c - h))).max(0.0)
        })
        .sum()
}

fn f6_talbot(dir: &Path, fft: &mut Fft2) -> R<Value> {
    let mut ds = Dataset::new(dir, "f6_talbot");
    let (lam, p) = (0.5, 0.05);
    let lm = lam / 1000.0;
    let k = 2.0 * PI / lm;
    let spp = 64usize;
    let periods = 4usize;
    let nx = spp * periods;
    let dx = p / spp as f64;
    let g = Grid2::new([nx, 4], [dx, dx], [0.0, 0.0])?;
    let periodic = Propagator::AngularSpectrum(AsmOptions { band_limit: false, pad: 1 });
    // Exact: λ/(1 − √(1 − λ²/p²)), written without the cancellation.
    let q = (lm / p).powi(2);
    let zt = lm * (1.0 + (1.0 - q).sqrt()) / q;
    let zt_par = 2.0 * p * p / lm;
    let mut warnings = Vec::new();

    // The binary grating: open where |x − kp| < p/4, each cell weighted by its
    // exact open fraction.
    let t: Vec<Cx> = (0..g.len())
        .map(|idx| {
            let x = g.coord(0, idx % nx);
            Cx::real(grating_cover(x - 0.5 * dx, x + 0.5 * dx, p, 0.25 * p) / dx)
        })
        .collect();
    let mut u = uniform(g, lam)?;
    let mut rt = WaveReport::default();
    apply_transmittance(&mut u, &t, &mut rt)?;

    // 5 µm between planes: order m's paraxial phase repeats every z_T/m²,
    // 9.8 µm for the highest order the sampling carries (m = 32).
    let nz = 2201usize;
    let zs = linspace(0.0, 1.1 * zt_par, nz);
    let mut carpet = Vec::with_capacity(nz * nx);
    let mut closure: f64 = 0.0;
    for &z in &zs {
        let (v, r) = propagate(&u, z, periodic, fft)?;
        collect_warnings(&format!("binary z={z:.4}"), &r, &mut warnings);
        closure = closure.max(ledger_closure(&r));
        for i in 0..nx {
            carpet.push(v.at(i, 0).norm_sqr());
        }
    }
    let i_in: Vec<f64> = (0..nx).map(|i| u.at(i, 0).norm_sqr()).collect();
    let imax_in = fmax(i_in.iter().copied());
    let profile = |z: f64, fft: &mut Fft2| -> R<(Vec<Cx>, String)> {
        let (v, r) = propagate(&u, z, periodic, fft)?;
        Ok(((0..nx).map(|i| v.at(i, 0)).collect(), method_of(&r)))
    };
    let (b_zt, method) = profile(zt, fft)?;
    let (b_zt_par, _) = profile(zt_par, fft)?;
    let (b_half, _) = profile(0.5 * zt, fft)?;
    let int_err = |a: &[Cx]| fmax(a.iter().zip(&i_in).map(|(c, i)| (c.norm_sqr() - i).abs())) / imax_in;
    let int_rms = |a: &[Cx]| {
        (a.iter().zip(&i_in).map(|(c, i)| (c.norm_sqr() - i).powi(2)).sum::<f64>() / nx as f64).sqrt() / imax_in
    };
    // Half-period shift at z_T/2: compare with the input moved by p/2 (32 samples).
    let half_err = fmax((0..nx).map(|i| (b_half[i].norm_sqr() - i_in[(i + nx - spp / 2) % nx]).abs())) / imax_in;
    let bin_complex = fmax(b_zt.iter().enumerate().map(|(i, c)| (*c * Cx::expi(-k * zt) - u.at(i, 0)).abs()));

    // The cosine grating, three plane waves: exact revival at z_T.
    let tc = |x: f64| 0.5 * (1.0 + (2.0 * PI * x / p).cos());
    let uc = Field2::from_fn(g, lam, |x, _| Cx::real(tc(x)))?;
    let (vc, rc) = propagate(&uc, zt, periodic, fft)?;
    collect_warnings("cosine z_T", &rc, &mut warnings);
    let cos_rev = fmax(uc.data.iter().zip(&vc.data).map(|(a, b)| (*a - *b * Cx::expi(-k * zt)).abs()));
    let (vh, _) = propagate(&uc, 0.5 * zt, periodic, fft)?;
    let mut cos_half: f64 = 0.0;
    for j in 0..g.n[1] {
        for i in 0..nx {
            let (x, _) = g.xy(i, j);
            let want = Cx::real(tc(x - 0.5 * p)) * Cx::expi(k * 0.5 * zt);
            cos_half = cos_half.max((vh.at(i, j) - want).abs());
        }
    }
    let (vcp, _) = propagate(&uc, zt_par, periodic, fft)?;
    let cos_rev_par = fmax(uc.data.iter().zip(&vcp.data).map(|(a, b)| (a.norm_sqr() - b.norm_sqr()).abs()));

    let xs: Vec<f64> = (0..nx).map(|i| g.coord(0, i)).collect();
    let to_int = |a: &[Cx]| a.iter().map(|c| c.norm_sqr()).collect::<Vec<f64>>();
    ds.array("z_mm", &zs, &[nz], &["plane"], "mm", "z of the carpet planes")?;
    ds.array("x_mm", &xs, &[nx], &["x"], "mm", "x of the samples (4 periods)")?;
    ds.array("carpet", &carpet, &[nz, nx], &["plane", "x"], "|U|^2 / |U_inc|^2", "binary-grating irradiance")?;
    ds.array("profile_input", &i_in, &[nx], &["x"], "|U|^2", "irradiance just after the grating (area-weighted edges)")?;
    ds.array("profile_zt_exact", &to_int(&b_zt), &[nx], &["x"], "|U|^2", "binary grating at the exact z_T")?;
    ds.array("profile_zt_half", &to_int(&b_half), &[nx], &["x"], "|U|^2", "binary grating at z_T/2 (exact z_T)")?;

    let errors = json!({
        "cosine_revival_worst_complex_err": cos_rev,
        "cosine_half_talbot_shift_worst_complex_err": cos_half,
        "cosine_at_paraxial_zt_worst_intensity_err": cos_rev_par,
        "binary_revival_exact_zt_worst_intensity_err_rel_max": int_err(&b_zt),
        "binary_revival_exact_zt_rms_intensity_err_rel_max": int_rms(&b_zt),
        "binary_revival_paraxial_zt_worst_intensity_err_rel_max": int_err(&b_zt_par),
        "binary_revival_exact_zt_worst_complex_err": bin_complex,
        "binary_half_talbot_shift_worst_intensity_err_rel_max": half_err,
        "ledger_worst_numerical_fraction": closure,
    });
    eprintln!("  F6 errors: {errors}");
    let meta = json!({
        "title": "Talbot carpet of a binary amplitude grating",
        "parameters": { "p_mm": p, "lambda_um": lam, "duty": 0.5, "samples_per_period": spp, "periods_in_window": periods,
            "dx_mm": dx, "grid": [nx, 4], "zT_paraxial_mm": zt_par, "zT_exact_mm": zt, "zT_exact_minus_paraxial_mm": zt - zt_par,
            "highest_order_sampled": spp / 2, "nz": nz, "z_max_mm": 1.1 * zt_par },
        "method": { "propagation": method, "options": "AsmOptions { band_limit: false, pad: 1 }: periodic window, exact for a field periodic in it",
            "grating": "apply_transmittance with exact area-weighted open fraction per cell" },
        "reference": "Wen, Zhang & Xiao 2013: revival at z_T = lambda/(1 - sqrt(1 - lambda^2/p^2)) for three plane waves (cosine grating), paraxially 2p^2/lambda; half-period shift at z_T/2",
        "sources": ["Wen 2013", "Goodman"],
        "warnings": warnings,
        "errors": errors,
    });
    ds.finish(meta)?;
    Ok(errors)
}

// ---------------------------------------------------------------------------
// F7: sampling regimes and the power ledger
// ---------------------------------------------------------------------------

fn f7_regimes_ledger(dir: &Path, fft: &mut Fft2) -> R<Value> {
    let mut ds = Dataset::new(dir, "f7_regimes_ledger");
    let mut warnings = Vec::new();

    // (a) Fresnel TF and IR against the analytic Gaussian across z_c.
    let (lam, w0, dx, n) = (0.5, 0.1, 0.005, 256usize);
    let lm = lam / 1000.0;
    let g = Grid2::square(n, dx)?;
    let beam = GaussianBeam::new(w0, lam, 1.0, 1.0);
    let u = beam.sample(g, 0.0)?;
    let lp = next_fast_len(2 * n) as f64 * dx;
    let zc = critical_distance_mm(dx, lp, lm);
    let ratios = logspace(0.05, 20.0, 41);
    let (mut e_tf, mut e_ir, mut out_tf, mut out_ir) = (vec![], vec![], vec![], vec![]);
    let mut wrap_flags = Vec::new();
    for &rz in &ratios {
        let z = rz * zc;
        let mut row = [0.0f64; 2];
        let mut outside = [0.0f64; 2];
        for (mi, how) in [Propagator::FresnelTf, Propagator::FresnelIr].into_iter().enumerate() {
            let (v, r) = propagate(&u, z, how, fft)?;
            let pk = beam.value(0.0, 0.0, z).abs();
            let mut worst: f64 = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let (x, y) = g.xy(i, j);
                    worst = worst.max((v.at(i, j) - beam.value(x, y, z)).abs() / pk);
                }
            }
            row[mi] = worst;
            outside[mi] = if r.warnings.iter().any(|w| matches!(w, SamplingWarning::OutsideRegime { .. })) { 1.0 } else { 0.0 };
            for w in &r.warnings {
                if matches!(w, SamplingWarning::WrapAround { .. }) {
                    wrap_flags.push(json!({ "z_over_zc": rz, "method": format!("{how:?}"), "warning": serde_json::to_value(w)? }));
                }
            }
        }
        e_tf.push(row[0]);
        e_ir.push(row[1]);
        out_tf.push(outside[0]);
        out_ir.push(outside[1]);
    }
    let in_regime_worst_tf = fmax(ratios.iter().zip(&e_tf).filter(|(r, _)| **r <= 1.0).map(|(_, e)| *e));
    let in_regime_worst_ir = fmax(ratios.iter().zip(&e_ir).filter(|(r, _)| **r >= 1.0).map(|(_, e)| *e));
    let at_zc = ratios.iter().position(|r| (r - 1.0).abs() < 1e-9);
    ds.array("a_z_over_zc", &ratios, &[ratios.len()], &["point"], "1", "z / z_c")?;
    ds.array("a_err_tf", &e_tf, &[ratios.len()], &["point"], "max|dU| / |U_analytic(0,0,z)|", "Fresnel transfer function vs the analytic Gaussian, worst over the window")?;
    ds.array("a_err_ir", &e_ir, &[ratios.len()], &["point"], "max|dU| / |U_analytic(0,0,z)|", "Fresnel impulse response vs the analytic Gaussian")?;
    ds.array("a_outside_tf", &out_tf, &[ratios.len()], &["point"], "0/1", "1 where the report carried OutsideRegime for the transfer function")?;
    ds.array("a_outside_ir", &out_ir, &[ratios.len()], &["point"], "0/1", "1 where the report carried OutsideRegime for the impulse response")?;

    // (b) The ledger of a band-limited angular-spectrum step, 1 mm to 3 m.
    let gb = Grid2::square(128, 0.01)?;
    let mut ub = uniform(gb, 0.5)?;
    let mut rap = WaveReport::default();
    apply_aperture(&mut ub, &Shape::Rect { c: [0.0, 0.0], h: [0.2, 0.2] }, false, &mut rap)?;
    let zcb = critical_distance_mm(0.01, next_fast_len(256) as f64 * 0.01, lm);
    let zb = logspace(1.0, 3000.0, 49);
    let (mut det, mut band, mut left, mut evan, mut num) = (vec![], vec![], vec![], vec![], vec![]);
    let mut bl_count = 0usize;
    for &z in &zb {
        let (_, r) = propagate(&ub, z, Propagator::AngularSpectrum(AsmOptions::default()), fft)?;
        if r.warnings.iter().any(|w| matches!(w, SamplingWarning::BandLimited { .. })) {
            bl_count += 1;
        }
        for w in &r.warnings {
            if !matches!(w, SamplingWarning::BandLimited { .. }) {
                warnings.push(json!({ "context": format!("ASM ledger z={z:.3}"), "warning": serde_json::to_value(w)? }));
            }
        }
        let l = r.ledger;
        det.push(l.detected / l.launched);
        band.push(l.band_limit_removed / l.launched);
        left.push(l.left_window / l.launched);
        evan.push(l.evanescent / l.launched);
        num.push(l.numerical / l.launched);
        let other = l.absorbed_aperture + l.absorbed_element + l.beyond_nyquist + l.other_family + l.untracked_orders;
        if other != 0.0 {
            return Err(format!("unexpected ledger entries at z = {z}: {l:?}").into());
        }
    }
    ds.array("b_z_mm", &zb, &[zb.len()], &["point"], "mm", "propagation distance")?;
    ds.array("b_detected", &det, &[zb.len()], &["point"], "fraction of launched", "ledger.detected / launched")?;
    ds.array("b_band_limit_removed", &band, &[zb.len()], &["point"], "fraction of launched", "ledger.band_limit_removed / launched")?;
    ds.array("b_left_window", &left, &[zb.len()], &["point"], "fraction of launched", "ledger.left_window / launched")?;
    ds.array("b_evanescent", &evan, &[zb.len()], &["point"], "fraction of launched", "ledger.evanescent / launched")?;
    ds.array("b_numerical", &num, &[zb.len()], &["point"], "fraction of launched", "ledger.numerical / launched (signed)")?;

    let errors = json!({
        "a_critical_distance_mm": zc,
        "a_tf_worst_for_z_le_zc": in_regime_worst_tf,
        "a_ir_worst_for_z_ge_zc": in_regime_worst_ir,
        "a_tf_at_zc": at_zc.map(|i| e_tf[i]),
        "a_ir_at_zc": at_zc.map(|i| e_ir[i]),
        "a_tf_at_20zc": e_tf.last(),
        "a_ir_at_0p05zc": e_ir.first(),
        "a_outside_regime_flags_agree_with_zc": ratios.iter().zip(out_tf.iter().zip(&out_ir)).all(|(r, (t, i))| {
            (*r > 1.0 + 1e-9) == (*t == 1.0) && (*r < 1.0 - 1e-9) == (*i == 1.0)
        }),
        "b_critical_distance_mm": zcb,
        "b_max_abs_numerical_fraction": fmax(num.iter().map(|v| v.abs())),
        "b_max_evanescent_fraction": fmax(evan.iter().copied()),
        "b_band_limited_warnings": bl_count,
        "b_band_limit_removed_at_3000mm": band.last(),
        "b_left_window_at_3000mm": left.last(),
        "b_detected_at_3000mm": det.last(),
    });
    eprintln!("  F7 errors: {errors}");
    let meta = json!({
        "title": "Sampling regimes and the power ledger",
        "parameters": {
            "a": { "w0_mm": w0, "lambda_um": lam, "n": n, "dx_mm": dx, "padded_window_mm": lp, "zR_mm": beam.rayleigh_range_mm(), "critical_distance_mm": zc },
            "b": { "aperture": "0.4 mm square", "lambda_um": 0.5, "n": 128, "dx_mm": 0.01, "padded_window_mm": next_fast_len(256) as f64 * 0.01, "critical_distance_mm": zcb,
                   "aperture_ledger": serde_json::to_value(rap.ledger)? },
        },
        "method": { "a": ["FresnelTf", "FresnelIr"], "b": "AngularSpectrum { band_limited: true }, pad 2" },
        "reference": "(a) Kogelnik & Li Gaussian, an exact solution of the Fresnel (paraxial) equation; (b) the ledger identity launched = detected + band_limit_removed + evanescent + left_window + numerical",
        "sources": ["Voelz 2009", "Matsushima 2009", "Kogelnik 1966"],
        "wraparound_warnings_a": wrap_flags,
        "warnings": warnings,
        "errors": errors,
    });
    ds.finish(meta)?;
    Ok(errors)
}

// ---------------------------------------------------------------------------
// F8: the DMD as a blazed grating, and its projector efficiency
// ---------------------------------------------------------------------------

fn dmd_device(pitch_um: f64, tilt_deg: f64, axis: TiltAxis, fill: f64) -> SpatialModulator {
    SpatialModulator {
        pixels: [1, 1],
        pitch_um,
        kind: ModulatorKind::Dmd(DmdSpec {
            tilt_deg,
            tilt_axis: axis,
            mirror_fill: fill,
            reflectance: 1.0,
            window_transmittance: 1.0,
        }),
        pattern: Pattern::Uniform { value: 1 },
    }
}

fn dmd_spec_of(d: &SpatialModulator) -> DmdSpec {
    match &d.kind {
        ModulatorKind::Dmd(s) => s.clone(),
        ModulatorKind::Lcos(_) => unreachable!("built as a DMD"),
    }
}

/// Popoff et al.'s geometry: light in the plane across a diagonal hinge, at
/// `alpha_deg` from the normal (as in tests/dmd.rs).
fn across_hinge(alpha_deg: f64) -> Vec3 {
    let a = alpha_deg.to_radians();
    let s = std::f64::consts::FRAC_1_SQRT_2;
    Vec3::new(-s * a.sin(), s * a.sin(), -a.cos())
}

fn f8_dmd(dir: &Path, _fft: &mut Fft2) -> R<Value> {
    let mut ds = Dataset::new(dir, "f8_dmd");
    let harvey = OrderModel::HarveyRenormalized;
    let modulated = LcosBeam::Modulated;

    // (a) The orders of an all-ON 7.56 µm, 12° diagonal DMD at 633 nm under
    // the illumination its ON mirrors send along the normal.
    let lam = 0.633;
    let (pitch_a, fno_a) = (7.56, 2.4);
    let d = dmd_device(pitch_a, 12.0, TiltAxis::Diagonal, 0.94f64.sqrt());
    let spec = dmd_spec_of(&d);
    let din = on_axis_illumination(&spec);
    let t = unit_cell_orders(&d, &UnitCell::uniform(1), lam, din, harvey, modulated)?;
    let env = reflect(din, facet_normal(&spec, DmdState::On));
    let na_proj = 1.0 / (2.0 * fno_a);
    let incidence_deg = (-din.z).acos().to_degrees();
    let (mut al, mut be, mut eff, mut om, mut on) = (vec![], vec![], vec![], vec![], vec![]);
    for o in &t.orders {
        al.push(o.dir.x);
        be.push(o.dir.y);
        eff.push(o.efficiency / t.total);
        om.push(o.m as f64);
        on.push(o.n as f64);
    }
    let b = t.brightest().ok_or("no orders")?;
    let in_cone = t.within(Vec3::Z, na_proj) / t.total;
    let k = al.len();
    ds.array("a_alpha", &al, &[k], &["order"], "direction cosine", "alpha of each propagating order")?;
    ds.array("a_beta", &be, &[k], &["order"], "direction cosine", "beta of each propagating order")?;
    ds.array("a_share", &eff, &[k], &["order"], "fraction of reflected power", "Harvey-renormalized efficiency / total reflected")?;
    ds.array("a_m", &om, &[k], &["order"], "1", "order index m")?;
    ds.array("a_n", &on, &[k], &["order"], "1", "order index n")?;

    // (b) Projector efficiency against wavelength, TI DLPA083B's devices.
    type Dev = (&'static str, f64, f64, TiltAxis, f64, f64, f64);
    let devs: [Dev; 5] = [
        ("13.68 µm, 12°, f/2.4", 13.68, 12.0, TiltAxis::Diagonal, 2.4, 0.92, 0.89),
        ("10.8 µm, 12°, f/2.4", 10.8, 12.0, TiltAxis::Diagonal, 2.4, 0.92, 0.87),
        ("9.0 µm, 14.5°, f/2.0", 9.0, 14.5, TiltAxis::Diagonal, 2.0, 0.97, 0.89),
        ("7.56 µm, 12°, f/2.4", 7.56, 12.0, TiltAxis::Diagonal, 2.4, 0.94, 0.84),
        ("5.4 µm, 17° side, f/1.7", 5.4, 17.0, TiltAxis::Side, 1.7, 0.93, 0.86),
    ];
    let lams = linspace(0.420, 0.680, 131);
    let mut curves = Vec::with_capacity(devs.len() * lams.len());
    let mut rows = Vec::new();
    for (name, pitch, tilt, axis, fno, ti_fill, ti_eff) in devs {
        let dv = dmd_device(pitch, tilt, axis, ti_fill.sqrt());
        for &l in &lams {
            curves.push(dmd_projector_efficiency(&dv, l, fno, fno, 64, harvey)?);
        }
        let ph = photopic_average(420.0, 680.0, 5.0, |l| dmd_projector_efficiency(&dv, l, fno, fno, 64, harvey))?;
        rows.push(json!({ "device": name, "pitch_um": pitch, "tilt_deg": tilt, "axis": format!("{axis:?}"), "f_number": fno,
            "ti_on_state_fill": ti_fill, "mirror_fill": ti_fill.sqrt(), "ti_photopic": ti_eff, "model_photopic": ph,
            "delta_pp": 100.0 * (ph - ti_eff) }));
        eprintln!("  F8 {name}: model {:.2}% vs TI {:.0}%", 100.0 * ph, 100.0 * ti_eff);
    }
    // The row the tests name as the one discrepancy beyond ±3 pp.
    let d54 = dmd_device(5.4, 17.0, TiltAxis::Side, 0.93f64.sqrt());
    let ph54 = photopic_average(420.0, 680.0, 5.0, |l| dmd_projector_efficiency(&d54, l, 2.4, 2.4, 64, harvey))?;
    let lnm: Vec<f64> = lams.iter().map(|l| l * 1000.0).collect();
    ds.array("b_lambda_nm", &lnm, &[lnm.len()], &["wavelength"], "nm", "wavelengths")?;
    ds.array("b_efficiency", &curves, &[devs.len(), lams.len()], &["device", "wavelength"], "fraction of reflected power", "dmd_projector_efficiency, matched illumination and projection cones, 64 directions, Harvey-renormalized")?;

    // (c) Popoff's blazed number against incidence, and the brightest
    // order's share of the reflected power from the order model.
    let lam_c = 0.633;
    let ac = linspace(-45.0, 45.0, 361);
    let mu76: Vec<f64> = ac.iter().map(|&a| popoff_blazed_number(7.6, lam_c, 12.0, a)).collect();
    let mu108: Vec<f64> = ac.iter().map(|&a| popoff_blazed_number(10.8, lam_c, 12.0, a)).collect();
    let acoarse = linspace(-45.0, 45.0, 37);
    let mut share = Vec::new();
    for pitch in [7.6, 10.8] {
        let dv = dmd_device(pitch, 12.0, TiltAxis::Diagonal, 0.95);
        for &a in &acoarse {
            let tt = unit_cell_orders(&dv, &UnitCell::uniform(1), lam_c, across_hinge(a), harvey, modulated)?;
            share.push(tt.brightest().map(|o| o.efficiency).unwrap_or(0.0) / tt.total);
        }
    }
    let published = [(7.6, 0.0, 0.06), (7.6, 25.0, 0.04), (10.8, 0.0, 0.97), (10.8, 25.0, 1.00)];
    let pub_cmp: Vec<Value> = published
        .iter()
        .map(|&(p, a, v)| json!({ "pitch_um": p, "alpha_deg": a, "published": v, "model": popoff_blazed_number(p, lam_c, 12.0, a) }))
        .collect();
    ds.array("c_alpha_deg", &ac, &[ac.len()], &["angle"], "deg", "incidence angle alpha' in the plane across the hinge")?;
    ds.array("c_mu_7p6", &mu76, &[ac.len()], &["angle"], "1", "popoff_blazed_number, 7.6 um, 12 deg, 633 nm")?;
    ds.array("c_mu_10p8", &mu108, &[ac.len()], &["angle"], "1", "popoff_blazed_number, 10.8 um, 12 deg, 633 nm")?;
    ds.array("c_alpha_coarse_deg", &acoarse, &[acoarse.len()], &["angle"], "deg", "angles of the order-model points")?;
    ds.array("c_brightest_share", &share, &[2, acoarse.len()], &["pitch (7.6, 10.8)", "angle"], "fraction of reflected power", "brightest order's Harvey-renormalized share, all-ON, fill 0.95")?;

    let errors = json!({
        "a_orders": k,
        "a_incidence_deg": incidence_deg,
        "a_envelope_centre": [env.x, env.y],
        "a_zero_order_direction": [din.x, din.y],
        "a_brightest": { "m": b.m, "n": b.n, "alpha": b.dir.x, "beta": b.dir.y, "share": b.efficiency / t.total },
        "a_share_in_projection_cone": in_cone,
        "a_projection_na": na_proj,
        "b_rows": rows,
        "b_5p4um_f2p4": { "ti_photopic": 0.80, "model_photopic": ph54, "delta_pp": 100.0 * (ph54 - 0.80) },
        "c_published_vs_model": pub_cmp,
        "c_worst_abs_diff_published": fmax(published.iter().map(|&(p, a, v)| (popoff_blazed_number(p, lam_c, 12.0, a) - v).abs())),
    });
    eprintln!("  F8 errors: {errors}");
    let meta = json!({
        "title": "The DMD as a blazed grating, and its projector efficiency",
        "parameters": {
            "a": { "pitch_um": pitch_a, "tilt_deg": 12.0, "axis": "Diagonal", "mirror_fill": 0.94f64.sqrt(), "lambda_um": lam,
                   "incident_direction": [din.x, din.y, din.z], "projection_f_number": fno_a },
            "b": { "lambda_range_nm": [420, 680], "samples_per_cone": 64, "photopic_step_nm": 5.0, "order_model": "HarveyRenormalized",
                   "reflectance": 1.0, "window_transmittance": 1.0 },
            "c": { "lambda_um": lam_c, "tilt_deg": 12.0, "order_model_fill": 0.95 },
        },
        "method": { "orders": "oc_core::modulator::unit_cell_orders (closed-form Fourier series of the unit cell, Harvey-renormalized)",
            "efficiency": "dmd_projector_efficiency (TI DLPA083B 4.2 method), photopic_average (CIE 1931 V(lambda))",
            "blazed_number": "popoff_blazed_number (Popoff et al. 2024 Eq. 2)" },
        "reference": "TI DLPA083B Table 6-1 (vendor simulation); Popoff et al. 2024 Fig. 5; Harvey & Vernold 1998 order grid",
        "sources": ["Popoff 2024", "TI DLPA083B", "Harvey 1998", "Harvey 2011"],
        "warnings": [],
        "errors": errors,
    });
    ds.finish(meta)?;
    Ok(errors)
}

// ---------------------------------------------------------------------------
// F9: the phase-only LCoS SLM
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn lcos_device(pixels: u32, pitch_um: f64, levels: u16, fill: f64, gap: (f64, f64), fringing_um: f64, pattern: Pattern) -> SpatialModulator {
    SpatialModulator {
        pixels: [pixels, pixels],
        pitch_um,
        kind: ModulatorKind::Lcos(LcosSpec {
            director_deg: 0.0,
            lut: PhaseLut::Linear { levels, max_phase_rad: 2.0 * PI },
            design_lambda_um: 0.633,
            pixel_fill: fill,
            gap_amplitude: gap.0,
            gap_phase_rad: gap.1,
            fringing_um,
            reflectance: 1.0,
        }),
        pattern,
    }
}

fn staircase(n: usize) -> UnitCell {
    UnitCell { period: [n, 1], values: (0..n as u16).collect() }
}

fn order_eff(t: &oc_core::modulator::OrderTable, m: i32, n: i32) -> f64 {
    t.orders.iter().find(|o| o.m == m && o.n == n).map(|o| o.efficiency).unwrap_or(0.0)
}

/// c₁ of an `n`-pixel blazed staircase whose phase is averaged over a window
/// `w` pixels wide, by direct quadrature of `(1/Λ)∫ e^{iφ_w(x)} e^{−i2πx/Λ} dx`
/// with `φ_w` written out piecewise: the same model as `lcos_reflect`'s
/// fringing, reached without sampling a field (as in tests/slm.rs).
fn fringed_c1(n: usize, w: f64) -> Cx {
    let lev = |j: i64| 2.0 * PI * (j.rem_euclid(n as i64)) as f64 / n as f64;
    let phi = |x: f64| -> f64 {
        if w == 0.0 {
            return lev(x.floor() as i64);
        }
        let (a, b) = (x - 0.5 * w, x + 0.5 * w);
        let mut s = 0.0;
        let mut k = a.floor() as i64;
        while (k as f64) < b {
            let lo = a.max(k as f64);
            let hi = b.min(k as f64 + 1.0);
            s += lev(k) * (hi - lo);
            k += 1;
        }
        s / w
    };
    let m = 200_000;
    let mut c = Cx::ZERO;
    for i in 0..m {
        let x = (i as f64 + 0.5) * n as f64 / m as f64;
        c += Cx::expi(phi(x) - 2.0 * PI * x / n as f64);
    }
    c / m as f64
}

/// Modified Bessel functions I₀, I₁ by their power series (fine for x ≲ 30).
fn bessel_i01(x: f64) -> (f64, f64) {
    let h = 0.5 * x;
    let (mut t0, mut t1) = (1.0, h);
    let (mut s0, mut s1) = (t0, t1);
    for k in 1..200 {
        let kf = k as f64;
        t0 *= h * h / (kf * kf);
        t1 *= h * h / (kf * (kf + 1.0));
        s0 += t0;
        s1 += t1;
        if t0 < 1e-17 * s0 && t1 < 1e-17 * s1 {
            break;
        }
    }
    (s0, s1)
}

fn f9_slm(dir: &Path, fft: &mut Fft2) -> R<Value> {
    let mut ds = Dataset::new(dir, "f9_slm");
    let normal = Vec3::new(0.0, 0.0, -1.0);
    let lam = 0.633;
    let mut warnings = Vec::new();

    // (a) N-level blazed gratings, closed-form orders (TEA).
    let ns: [usize; 16] = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64];
    let (rho, gap) = (0.9, 0.5);
    let (mut e1, mut e1g, mut e0g) = (vec![], vec![], vec![]);
    let (mut w1, mut w1g, mut w0g): (f64, f64, f64) = (0.0, 0.0, 0.0);
    for &n in &ns {
        let d = lcos_device(1, 8.0, n as u16, 1.0, (0.0, 0.0), 0.0, Pattern::Uniform { value: 0 });
        let t = unit_cell_orders(&d, &staircase(n), lam, normal, OrderModel::Tea, LcosBeam::Modulated)?;
        let v = order_eff(&t, 1, 0);
        w1 = w1.max((v - msinc(1.0 / n as f64).powi(2)).abs());
        e1.push(v);
        let dg = lcos_device(1, 8.0, n as u16, rho, (gap, 0.0), 0.0, Pattern::Uniform { value: 0 });
        let tg = unit_cell_orders(&dg, &staircase(n), lam, normal, OrderModel::Tea, LcosBeam::Modulated)?;
        let (v1, v0) = (order_eff(&tg, 1, 0), order_eff(&tg, 0, 0));
        w1g = w1g.max((v1 - rho.powi(4) * msinc(rho / n as f64).powi(2)).abs());
        w0g = w0g.max((v0 - gap * gap * (1.0 - rho * rho).powi(2)).abs());
        e1g.push(v1);
        e0g.push(v0);
    }
    let nsf: Vec<f64> = ns.iter().map(|&n| n as f64).collect();
    let dense_n = logspace(2.0, 64.0, 400);
    let dn1: Vec<f64> = dense_n.iter().map(|&n| msinc(1.0 / n).powi(2)).collect();
    let dn1g: Vec<f64> = dense_n.iter().map(|&n| rho.powi(4) * msinc(rho / n).powi(2)).collect();
    ds.array("a_levels", &nsf, &[ns.len()], &["N"], "1", "phase levels per period (period = N pixels)")?;
    ds.array("a_eta1", &e1, &[ns.len()], &["N"], "fraction of incident power", "first order, fill 1, no dead space (unit_cell_orders, TEA)")?;
    ds.array("a_eta1_dead", &e1g, &[ns.len()], &["N"], "fraction of incident power", "first order, pixel fill 0.9, gap amplitude 0.5")?;
    ds.array("a_eta0_dead", &e0g, &[ns.len()], &["N"], "fraction of incident power", "zero order, pixel fill 0.9, gap amplitude 0.5")?;
    ds.array("a_dense_n", &dense_n, &[dense_n.len()], &["N"], "1", "N for the analytic curves")?;
    ds.array("a_dense_sinc2", &dn1, &[dense_n.len()], &["N"], "1", "sinc^2(1/N)")?;
    ds.array("a_dense_dead", &dn1g, &[dense_n.len()], &["N"], "1", "rho^4 sinc^2(rho/N)")?;

    // (b) Fringing: the sampled field projected on the first order over the
    // interior periods, against direct quadrature (tests/slm.rs).
    let (nlev, per, np, pitch_um) = (8u16, 32usize, 48u32, 8.0);
    let wpx: Vec<f64> = (0..=24).map(|k| k as f64 * 0.0625).collect();
    let (mut eta_s, mut eta_q_at) = (vec![], vec![]);
    for &w in &wpx {
        let d = lcos_device(np, pitch_um, nlev, 1.0, (0.0, 0.0), w * pitch_um, Pattern::Explicit(
            (0..np as usize * np as usize).map(|k| ((k % np as usize) % nlev as usize) as u16).collect(),
        ));
        let p = pitch_um / 1000.0;
        let dx = p / per as f64;
        let ns_ = np as usize * per;
        let g = Grid2::new([ns_, 8], [dx, dx], [0.5 * dx, 0.5 * dx])?;
        let u = Field2::from_fn(g, lam, |_, _| Cx::ONE)?;
        let (v, r) = lcos_reflect(&u, &d, LcosBeam::Modulated)?;
        collect_warnings(&format!("lcos_reflect fringing w={w} px"), &r, &mut warnings);
        let lam_g = nlev as f64 * p;
        let period = nlev as usize * per;
        let left = -(np as f64) * p / 2.0;
        let mut c = Cx::ZERO;
        let mut count = 0usize;
        for j in 0..g.n[1] {
            for i in period..ns_ - period {
                let (x, _) = g.xy(i, j);
                c += v.at(i, j) * Cx::expi(-2.0 * PI * (x - left) / lam_g);
                count += 1;
            }
        }
        eta_s.push((c / count as f64).norm_sqr());
        eta_q_at.push(fringed_c1(nlev as usize, w).norm_sqr());
    }
    let wq: Vec<f64> = (0..=60).map(|k| k as f64 * 0.025).collect();
    let eta_q: Vec<f64> = wq.iter().map(|&w| fringed_c1(nlev as usize, w).norm_sqr()).collect();
    let fr_worst = fmax(eta_s.iter().zip(&eta_q_at).map(|(a, b)| (a - b).abs()));
    ds.array("b_w_px", &wpx, &[wpx.len()], &["w"], "pixels", "fringing width over pitch, sampled points")?;
    ds.array("b_eta_sampled", &eta_s, &[wpx.len()], &["w"], "fraction", "first-order efficiency of the sampled field (lcos_reflect, 32 samples/pixel, interior 4 of 6 periods)")?;
    ds.array("b_w_quad", &wq, &[wq.len()], &["w"], "pixels", "fringing width for the quadrature curve")?;
    ds.array("b_eta_quadrature", &eta_q, &[wq.len()], &["w"], "fraction", "direct quadrature of the same window-averaged phase model")?;

    // (c) A Gaussian through an l = 1 spiral, focused.
    let npx = 64u32;
    let dv = lcos_device(npx, 8.0, 256, 1.0, (0.0, 0.0), 0.0, Pattern::Generated(PatternGenerator::Vortex { charge: 1 }));
    let perv = 4usize;
    let dx = 8e-3 / perv as f64;
    let nsv = npx as usize * perv;
    let g = Grid2::new([nsv, nsv], [dx, dx], [0.5 * dx, 0.5 * dx])?;
    let w0 = 0.12;
    let u = Field2::from_fn(g, lam, |x, y| Cx::real((-(x * x + y * y) / (w0 * w0)).exp()))?;
    let (v, rv) = lcos_reflect(&u, &dv, LcosBeam::Modulated)?;
    collect_warnings("lcos_reflect vortex", &rv, &mut warnings);
    let zf = 500.0;
    let (far, rf) = propagate(&v, zf, Propagator::Fraunhofer { pad: 16 }, fft)?;
    collect_warnings("Fraunhofer focus (stands in for an f = 500 mm lens)", &rf, &mut warnings);
    let lm = lam / 1000.0;
    let k = 2.0 * PI / lm;
    // Remove e^{ikz} e^{ikr²/2z}/i: the back-focal-plane field of a lens of
    // focal length z with the SLM in its front focal plane, F{U}(x/λz)/(λz).
    let fg = far.grid;
    let c = fg.n[0] / 2;
    let half = 3.0;
    let idx: Vec<usize> = (0..fg.n[0]).filter(|&i| fg.coord(0, i).abs() <= half + 1e-12).collect();
    let peak = far.data.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max);
    let (mut img, mut ph) = (vec![], vec![]);
    for &j in &idx {
        for &i in &idx {
            let (x, y) = fg.xy(i, j);
            let bf = far.at(i, j) * Cx::expi(-k * zf - k * (x * x + y * y) / (2.0 * zf)) * Cx::I;
            img.push(bf.norm_sqr() / peak);
            ph.push(bf.arg());
        }
    }
    let coords: Vec<f64> = idx.iter().map(|&i| fg.coord(0, i)).collect();
    let axis = far.at(c, c).norm_sqr() / peak;
    // Winding around a ring of 18 samples' radius (0.7 mm, the bright ring).
    let ring: Vec<Cx> = (0..128)
        .map(|q| {
            let a = 2.0 * PI * q as f64 / 128.0;
            let (i, j) = ((c as f64 + 18.0 * a.cos()).round() as usize, (c as f64 + 18.0 * a.sin()).round() as usize);
            far.at(i, j)
        })
        .collect();
    let wind: f64 = (0..128).map(|q| (ring[(q + 1) % 128] / ring[q]).arg()).sum();
    let winding = (wind / (2.0 * PI)).round();
    // Azimuthal average against the closed form: |∫e^{−r²/w₀²} J₁(br) r dr|²
    // ∝ b² e^{−2q}[I₀(q) − I₁(q)]², q = b²w₀²/8, b = 2πR/(λz).
    let pitch_f = fg.pitch_mm[0];
    let nb = (half / pitch_f).floor() as usize;
    let (mut acc, mut cnt) = (vec![0.0; nb + 1], vec![0usize; nb + 1]);
    for j in 0..fg.n[1] {
        for i in 0..fg.n[0] {
            let (x, y) = fg.xy(i, j);
            let r = (x * x + y * y).sqrt();
            let bin = (r / pitch_f).round() as usize;
            if bin <= nb {
                acc[bin] += far.at(i, j).norm_sqr() / peak;
                cnt[bin] += 1;
            }
        }
    }
    let rad: Vec<f64> = (0..=nb).map(|b| b as f64 * pitch_f).collect();
    let prof: Vec<f64> = acc.iter().zip(&cnt).map(|(a, n)| if *n > 0 { a / *n as f64 } else { 0.0 }).collect();
    let analytic = |r: f64| {
        let b = 2.0 * PI * r / (lm * zf);
        let q = b * b * w0 * w0 / 8.0;
        let (i0, i1) = bessel_i01(q);
        b * b * (-2.0 * q).exp() * (i0 - i1).powi(2)
    };
    let dense_r = linspace(0.0, half, 1201);
    let an: Vec<f64> = dense_r.iter().map(|&r| analytic(r)).collect();
    let an_max = fmax(an.iter().copied());
    let an_n: Vec<f64> = an.iter().map(|v| v / an_max).collect();
    let ia = an.iter().enumerate().fold((0, 0.0), |a, (i, v)| if *v > a.1 { (i, *v) } else { a }).0;
    let dr = dense_r[1] - dense_r[0];
    let r_an = dense_r[ia] + 0.5 * dr * (an[ia - 1] - an[ia + 1]) / (an[ia - 1] - 2.0 * an[ia] + an[ia + 1]);
    // Simulated ring radius: parabola through the brightest annulus and its neighbours.
    let ib = prof.iter().enumerate().fold((0, 0.0), |a, (i, v)| if *v > a.1 { (i, *v) } else { a }).0;
    let r_sim = if ib > 0 && ib < nb {
        let (ym, y0, yp) = (prof[ib - 1], prof[ib], prof[ib + 1]);
        rad[ib] + 0.5 * pitch_f * (ym - yp) / (ym - 2.0 * y0 + yp)
    } else {
        rad[ib]
    };
    let prof_max = fmax(prof.iter().copied());
    let prof_n: Vec<f64> = prof.iter().map(|v| v / prof_max).collect();
    let prof_err = fmax(rad.iter().zip(&prof_n).map(|(&r, &p)| (p - analytic(r) / an_max).abs()));
    ds.array("c_coord_mm", &coords, &[coords.len()], &["x or y"], "mm", "back-focal-plane coordinates (both axes)")?;
    ds.array("c_intensity", &img, &[idx.len(), idx.len()], &["y", "x"], "I / I_peak", "focused irradiance")?;
    ds.array("c_phase", &ph, &[idx.len(), idx.len()], &["y", "x"], "rad", "phase of the back-focal-plane field (quadratic Fraunhofer phase removed)")?;
    ds.array("c_radius_mm", &rad, &[rad.len()], &["r"], "mm", "annulus radius (one far-field sample wide)")?;
    ds.array("c_profile_sim", &prof_n, &[rad.len()], &["r"], "normalized", "azimuthal mean of the simulated irradiance")?;
    ds.array("c_dense_r_mm", &dense_r, &[dense_r.len()], &["r"], "mm", "radius for the closed form")?;
    ds.array("c_profile_analytic", &an_n, &[dense_r.len()], &["r"], "normalized", "Gaussian times exp(i phi), focused: b^2 e^{-2q}[I0(q) - I1(q)]^2")?;

    let errors = json!({
        "a_worst_abs_err_sinc2": w1,
        "a_worst_abs_err_dead_space_first_order": w1g,
        "a_worst_abs_err_dead_space_zero_order": w0g,
        "a_dead_space_zero_order": gap * gap * (1.0 - rho * rho).powi(2),
        "b_worst_abs_diff_sampled_vs_quadrature": fr_worst,
        "b_eta_at_w": wpx.iter().zip(&eta_s).filter(|(w, _)| [0.0, 0.25, 0.5, 1.0, 1.5].iter().any(|q| (*w - q).abs() < 1e-9))
            .map(|(w, e)| json!({ "w_px": w, "eta": e })).collect::<Vec<_>>(),
        "c_on_axis_over_peak": axis,
        "c_winding_number": winding,
        "c_ring_radius_sim_mm": r_sim,
        "c_ring_radius_analytic_mm": r_an,
        "c_ring_radius_rel_err": (r_sim - r_an) / r_an,
        "c_profile_worst_abs_diff_normalized": prof_err,
        "c_far_pitch_mm": pitch_f,
    });
    eprintln!("  F9 errors: {errors}");
    let meta = json!({
        "title": "Phase-only LCoS SLM",
        "parameters": {
            "a": { "pitch_um": 8.0, "lambda_um": lam, "design_lambda_um": 0.633, "levels": ns, "dead_space": { "pixel_fill": rho, "gap_amplitude": gap, "gap_phase_rad": 0.0 } },
            "b": { "levels": nlev, "pitch_um": pitch_um, "pixels": np, "samples_per_pixel": per, "projection": "interior 4 of 6 periods", "quadrature_samples": 200000 },
            "c": { "pixels": npx, "pitch_um": 8.0, "lut_levels": 256, "samples_per_pixel": perv, "w0_mm": w0, "lambda_um": lam, "focal_mm": zf, "pad": 16,
                   "far_pitch_mm": pitch_f, "displayed_half_width_mm": half },
        },
        "method": { "a": "unit_cell_orders (TEA), closed-form Fourier series of the cell", "b": "lcos_reflect with fringing_um = w (window-averaged phase), projected on exp(i2 pi x/Lambda)",
            "c": "lcos_reflect (Vortex charge 1), then propagate(Fraunhofer { pad: 16 }); e^{ikz} e^{ikr^2/2z}/i removed" },
        "reference": "Swanson 1989 sinc^2(1/N); dead space |g|^2(1-rho^2)^2 and rho^4 sinc^2(rho/N); fringing model of JOSA A 21, 1996 (2004) by quadrature; Allen et al. 1992 dark core",
        "sources": ["Swanson 1989", "Zhang 2014", "Persson 2012", "JOSA A 2004", "Allen 1992"],
        "validity_flags_vortex_reflect": serde_json::to_value(&rv.validity)?,
        "approximations_vortex_reflect": rv.approximated.clone(),
        "warnings": warnings,
        "errors": errors,
    });
    ds.finish(meta)?;
    Ok(errors)
}

fn main() -> R<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: cargo run -p oc-wave --release --example gallery -- <outdir>")?,
    );
    std::fs::create_dir_all(&dir)?;
    let only: Option<String> = std::env::args().nth(2);
    let t0 = Instant::now();
    let mut fft = Fft2::new();
    let mut timings = Map::new();
    type Job = fn(&Path, &mut Fft2) -> R<Value>;
    let jobs: [(&str, Job); 9] = [
        ("f1_airy", f1_airy),
        ("f2_heurtley", f2_heurtley),
        ("f3_fresnel_square", f3_fresnel_square),
        ("f4_fresnel_to_fraunhofer", f4_fresnel_to_fraunhofer),
        ("f5_gaussian_lens", f5_gaussian_lens),
        ("f6_talbot", f6_talbot),
        ("f7_regimes_ledger", f7_regimes_ledger),
        ("f8_dmd", f8_dmd),
        ("f9_slm", f9_slm),
    ];
    for (name, job) in jobs {
        if let Some(o) = &only {
            if !name.starts_with(o.as_str()) {
                continue;
            }
        }
        let t = Instant::now();
        job(&dir, &mut fft)?;
        timings.insert(name.into(), json!(t.elapsed().as_secs_f64()));
    }
    let total = t0.elapsed().as_secs_f64();
    let summary = json!({
        "generator": "crates/oc-wave/examples/gallery.rs",
        "command": "cargo run -p oc-wave --release --example gallery -- <outdir>",
        "seconds_total": total,
        "seconds_per_dataset": timings,
        "max_wave_samples": oc_wave::MAX_WAVE_SAMPLES,
    });
    let path = if only.is_some() { "run_partial.json" } else { "run.json" };
    std::fs::write(dir.join(path), serde_json::to_string_pretty(&summary)?)?;
    eprintln!("[gallery] all done in {total:.1} s");
    Ok(())
}
