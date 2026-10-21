//! Scalar wave optics across the Python boundary (`oc-wave`, ADR 0003).
//!
//! The rest of this crate exchanges JSON. A field is a million complex numbers,
//! so it crosses as `bytes` instead: interleaved little-endian `f64`, real part
//! first, row-major with `x` fastest, which is exactly numpy's `complex128` in C
//! order, so the Python side reads it with `numpy.frombuffer` and needs no numpy
//! crate here. The report crosses as JSON, as everything else does.

use oc_wave::czt::fraunhofer_zoom;
use oc_wave::{propagate, AsmOptions, Cx, Field2, Fft2, Grid2, Propagator};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn to_py<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn read_field(
    data: &[u8],
    n: (usize, usize),
    pitch_mm: (f64, f64),
    center_mm: (f64, f64),
    lambda_um: f64,
    n_medium: f64,
    carrier: (f64, f64),
) -> PyResult<Field2> {
    let grid = Grid2::new([n.0, n.1], [pitch_mm.0, pitch_mm.1], [center_mm.0, center_mm.1])
        .map_err(to_py)?;
    if data.len() != grid.len() * 16 {
        return Err(PyValueError::new_err(format!(
            "{} bytes do not hold a {}×{} complex128 field ({} bytes)",
            data.len(),
            n.0,
            n.1,
            grid.len() * 16
        )));
    }
    let mut u = Field2::zeros(grid, lambda_um).map_err(to_py)?;
    for (z, c) in u.data.iter_mut().zip(data.chunks_exact(16)) {
        let re = f64::from_le_bytes(c[..8].try_into().expect("8 bytes"));
        let im = f64::from_le_bytes(c[8..].try_into().expect("8 bytes"));
        *z = Cx::new(re, im);
    }
    if !u.data.iter().all(|z| z.is_finite()) {
        return Err(PyValueError::new_err("the field has a non-finite sample"));
    }
    let mut u = u.in_medium(n_medium).map_err(to_py)?;
    u.carrier = [carrier.0, carrier.1];
    Ok(u)
}

fn field_bytes<'py>(py: Python<'py>, u: &Field2) -> Bound<'py, PyBytes> {
    let mut out = Vec::with_capacity(u.data.len() * 16);
    for z in &u.data {
        out.extend_from_slice(&z.re.to_le_bytes());
        out.extend_from_slice(&z.im.to_le_bytes());
    }
    PyBytes::new(py, &out)
}

type FieldOut<'py> = (
    Bound<'py, PyBytes>,
    (usize, usize),
    (f64, f64),
    (f64, f64),
    String,
);

/// Propagate a sampled field `z_mm` along its plane's normal.
///
/// `method` is one of `"auto"`, `"asm"` (angular spectrum; `band_limit` and
/// `pad` apply), `"fresnel_tf"`, `"fresnel_ir"`, `"fraunhofer"` (`pad`
/// applies) and `"rs_direct"`. Returns the field's bytes, its grid
/// (`n`, `pitch_mm`, `center_mm`, which change for Fraunhofer) and the report
/// as JSON.
#[pyfunction]
#[pyo3(signature = (data, n, pitch_mm, lambda_um, z_mm, method = "auto", n_medium = 1.0,
                    center_mm = (0.0, 0.0), carrier = (0.0, 0.0), pad = 2, band_limit = true))]
#[allow(clippy::too_many_arguments)]
pub fn wave_propagate<'py>(
    py: Python<'py>,
    data: &[u8],
    n: (usize, usize),
    pitch_mm: (f64, f64),
    lambda_um: f64,
    z_mm: f64,
    method: &str,
    n_medium: f64,
    center_mm: (f64, f64),
    carrier: (f64, f64),
    pad: usize,
    band_limit: bool,
) -> PyResult<FieldOut<'py>> {
    let u = read_field(data, n, pitch_mm, center_mm, lambda_um, n_medium, carrier)?;
    let how = match method {
        "auto" => Propagator::Auto,
        "asm" => Propagator::AngularSpectrum(AsmOptions { band_limit, pad }),
        "fresnel_tf" => Propagator::FresnelTf,
        "fresnel_ir" => Propagator::FresnelIr,
        "fraunhofer" => Propagator::Fraunhofer { pad },
        "rs_direct" => Propagator::RsDirect,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}: auto, asm, fresnel_tf, fresnel_ir, fraunhofer, rs_direct"
            )))
        }
    };
    let (v, report) = propagate(&u, z_mm, how, &mut Fft2::new()).map_err(to_py)?;
    let json = serde_json::to_string(&report).map_err(to_py)?;
    Ok((
        field_bytes(py, &v),
        (v.grid.n[0], v.grid.n[1]),
        (v.grid.pitch_mm[0], v.grid.pitch_mm[1]),
        (v.grid.center_mm[0], v.grid.center_mm[1]),
        json,
    ))
}

/// The Fraunhofer field `z_mm` ahead on any output grid (`out_n`,
/// `out_pitch_mm`, `out_center_mm`), by the chirp-z transform. Returns the
/// field's bytes and the report as JSON.
#[pyfunction]
#[pyo3(signature = (data, n, pitch_mm, lambda_um, z_mm, out_n, out_pitch_mm, out_center_mm = (0.0, 0.0),
                    n_medium = 1.0, center_mm = (0.0, 0.0), carrier = (0.0, 0.0)))]
#[allow(clippy::too_many_arguments)]
pub fn wave_fraunhofer_zoom<'py>(
    py: Python<'py>,
    data: &[u8],
    n: (usize, usize),
    pitch_mm: (f64, f64),
    lambda_um: f64,
    z_mm: f64,
    out_n: (usize, usize),
    out_pitch_mm: (f64, f64),
    out_center_mm: (f64, f64),
    n_medium: f64,
    center_mm: (f64, f64),
    carrier: (f64, f64),
) -> PyResult<(Bound<'py, PyBytes>, String)> {
    let u = read_field(data, n, pitch_mm, center_mm, lambda_um, n_medium, carrier)?;
    let out = Grid2::new(
        [out_n.0, out_n.1],
        [out_pitch_mm.0, out_pitch_mm.1],
        [out_center_mm.0, out_center_mm.1],
    )
    .map_err(to_py)?;
    let (v, report) = fraunhofer_zoom(&u, z_mm, out, &mut Fft2::new()).map_err(to_py)?;
    let json = serde_json::to_string(&report).map_err(to_py)?;
    Ok((field_bytes(py, &v), json))
}

fn parse_device(device_json: &str) -> PyResult<oc_core::modulator::SpatialModulator> {
    let d: oc_core::modulator::SpatialModulator = serde_json::from_str(device_json).map_err(to_py)?;
    d.validate().map_err(to_py)?;
    Ok(d)
}

fn parse_model(model: &str) -> PyResult<oc_core::modulator::OrderModel> {
    match model {
        "harvey" => Ok(oc_core::modulator::OrderModel::HarveyRenormalized),
        "tea" => Ok(oc_core::modulator::OrderModel::Tea),
        other => Err(PyValueError::new_err(format!("unknown order model {other:?}: harvey, tea"))),
    }
}

/// The closed-form diffraction orders of a periodic cell on a DMD or LCoS
/// (`oc_core::modulator::unit_cell_orders`), as JSON: `{"total",
/// "propagating", "orders": [{"m", "n", "dir", "amplitude": [re, im],
/// "efficiency"}]}`. `device_json` is a `SpatialModulator`; `d_in` a unit
/// direction travelling toward −Z; `model` `"harvey"` or `"tea"`; `beam`
/// `"modulated"` or `"unmodulated"` (LCoS only).
#[pyfunction]
#[pyo3(signature = (device_json, cell_period, cell_values, lambda_um, d_in, model = "harvey", beam = "modulated"))]
pub fn modulator_orders(
    device_json: &str,
    cell_period: (usize, usize),
    cell_values: Vec<u16>,
    lambda_um: f64,
    d_in: (f64, f64, f64),
    model: &str,
    beam: &str,
) -> PyResult<String> {
    use oc_core::modulator::{unit_cell_orders, LcosBeam, UnitCell};
    let d = parse_device(device_json)?;
    let beam = match beam {
        "modulated" => LcosBeam::Modulated,
        "unmodulated" => LcosBeam::Unmodulated,
        other => return Err(PyValueError::new_err(format!("unknown beam {other:?}"))),
    };
    let cell = UnitCell { period: [cell_period.0, cell_period.1], values: cell_values };
    let t = unit_cell_orders(&d, &cell, lambda_um, oc_core::Vec3::new(d_in.0, d_in.1, d_in.2), parse_model(model)?, beam)
        .map_err(to_py)?;
    let orders: Vec<serde_json::Value> = t
        .orders
        .iter()
        .map(|o| {
            serde_json::json!({
                "m": o.m, "n": o.n, "dir": [o.dir.x, o.dir.y, o.dir.z],
                "coefficient": [o.coefficient.re, o.coefficient.im], "efficiency": o.efficiency,
            })
        })
        .collect();
    serde_json::to_string(&serde_json::json!({"total": t.total, "propagating": t.propagating, "orders": orders}))
        .map_err(to_py)
}

/// The diffraction efficiency of an all-ON DMD in a projector, TI DLPA083B's
/// method (`oc_core::modulator::dmd_projector_efficiency`).
#[pyfunction]
#[pyo3(signature = (device_json, lambda_um, illum_fno, proj_fno, samples = 64, model = "harvey"))]
pub fn dmd_projector_efficiency(
    device_json: &str,
    lambda_um: f64,
    illum_fno: f64,
    proj_fno: f64,
    samples: usize,
    model: &str,
) -> PyResult<f64> {
    let d = parse_device(device_json)?;
    oc_core::modulator::dmd_projector_efficiency(&d, lambda_um, illum_fno, proj_fno, samples, parse_model(model)?)
        .map_err(to_py)
}

/// Gerchberg–Saxton phase design between a source plane and its Fraunhofer
/// far field (`oc_wave::design::gerchberg_saxton`): returns the phase and the
/// far-field error after each iteration.
#[pyfunction]
#[pyo3(signature = (source, target, n, iterations, start = None))]
pub fn gerchberg_saxton(
    source: Vec<f64>,
    target: Vec<f64>,
    n: (usize, usize),
    iterations: usize,
    start: Option<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let r = oc_wave::design::gerchberg_saxton(&source, &target, [n.0, n.1], iterations, start.as_deref())
        .map_err(to_py)?;
    Ok((r.phase, r.error))
}
