//! PyO3 bindings — the `_native` submodule of the `optiland_canvas` package.
//!
//! This layer is deliberately thin: **all** of the optical geometry lives in
//! `oc-core` (the [`SeqSystem`](oc_core::SeqSystem) bridge). Python only turns
//! the neutral sequential prescription into an Optiland `Optic` and back, so
//! there is exactly one implementation of "what a lens is" and it is in Rust.
//!
//! The exchanges crossing this boundary are all JSON strings:
//! - [`scene_to_seq`] `.ocanvas` → sequential-prescription JSON (+ fidelity notes)
//! - [`seq_to_scene`] sequential-prescription JSON → `.ocanvas` (version 1)
//! - [`seq_to_scene3`] sequential-prescription JSON → `.ocanvas` (version 2)
//! - [`validate`] parse-and-version-check an `.ocanvas` document
//! - [`scene3_sequence`] `.ocanvas` (v1 or v2) → `oc-seq3` axial-ray sequence
//!   prescription for one source/detector pair — the 3D, trace-derived sibling
//!   of [`scene_to_seq`]
//!
//! Every entry point that takes a document reads it through
//! [`oc_core::document::read_document`], so v1 and v2 are read the same way
//! everywhere, and a document that cannot be read raises one of three
//! exceptions saying which way it failed (see [`doc_err`]).

use oc_core::document::{read_document, Document, DocumentError, READABLE_VERSIONS};
use oc_core::{SeqSystem, OCANVAS_FORMAT, OCANVAS_VERSION};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

create_exception!(
    _native,
    OcanvasError,
    PyValueError,
    "An .ocanvas document could not be used. A ValueError, so code that caught \
     ValueError before these existed still catches them."
);
create_exception!(
    _native,
    NotADocumentError,
    OcanvasError,
    "The bytes are not JSON, or JSON that is not an .ocanvas document."
);
create_exception!(
    _native,
    UnsupportedVersionError,
    OcanvasError,
    "An .ocanvas document of a version this build does not read."
);
create_exception!(
    _native,
    InvalidSceneError,
    OcanvasError,
    "An .ocanvas document of a readable version whose scene does not hold up: \
     a field of the wrong type, a non-finite number, a dangling reference."
);
create_exception!(
    _native,
    NotExportableError,
    OcanvasError,
    "A valid scene with optical content the sequential export cannot represent."
);
create_exception!(
    _native,
    NotImportableError,
    OcanvasError,
    "A sequential prescription with content a canvas scene cannot be built from:      a mirror, a virtual image, a glass with no data."
);

/// The exception a document failure raises: which of the three it was.
fn doc_err(e: DocumentError) -> PyErr {
    let msg = e.to_string();
    match e {
        DocumentError::NotADocument(_) => NotADocumentError::new_err(msg),
        DocumentError::UnsupportedVersion { .. } => UnsupportedVersionError::new_err(msg),
        DocumentError::InvalidScene(_) => InvalidSceneError::new_err(msg),
    }
}

/// Extract the rotationally-symmetric sequential prescription from an
/// `.ocanvas` document. Returns `SeqSystem` JSON, whose `notes` array reports
/// anything that could not be represented exactly and whose `stop` is the
/// declared aperture stop's surface index, or null when nothing declares one.
///
/// A version 2 scene exports the path from `source_id` to `detector_id`,
/// traced (see `oc_trace::seq_export`). Without ids it exports its one
/// declared path, or its one source and one area detector. A version 1 scene
/// has one implicit path, and naming ids for it is an error.
///
/// Raises `NotExportableError`, with every reason, when the path has content a
/// sequential system cannot hold, cannot be found, or the stop declarations
/// contradict each other.
#[pyfunction]
#[pyo3(signature = (ocanvas_json, source_id = None, detector_id = None))]
fn scene_to_seq(
    ocanvas_json: &str,
    source_id: Option<u64>,
    detector_id: Option<u64>,
) -> PyResult<String> {
    match read_document(ocanvas_json).map_err(doc_err)? {
        Document::V1(scene) => {
            if source_id.is_some() || detector_id.is_some() {
                return Err(PyValueError::new_err(
                    "a version 1 scene has one implicit path and no ids to choose one by",
                ));
            }
            Ok(scene.to_seq().to_json())
        }
        Document::V2(scene) | Document::V3(scene) => {
            let (source, detector) = match (source_id, detector_id) {
                (Some(s), Some(d)) => (oc_core::id::SourceId(s), oc_core::id::DetectorId(d)),
                (None, None) => oc_trace::default_path(&scene).map_err(export_err)?,
                _ => {
                    return Err(PyValueError::new_err(
                        "name both source_id and detector_id, or neither",
                    ))
                }
            };
            oc_trace::export_seq(&scene, source, detector)
                .map(|s| s.to_json())
                .map_err(export_err)
        }
    }
}

fn export_err(e: oc_trace::ExportError) -> PyErr {
    NotExportableError::new_err(e.to_string())
}

/// Rebuild an `.ocanvas` document from a sequential-prescription JSON string.
#[pyfunction]
fn seq_to_scene(seq_json: &str) -> PyResult<String> {
    let sys = SeqSystem::from_json(seq_json).map_err(to_py_err)?;
    Ok(sys.to_scene().to_json())
}

/// Build a version 2 `.ocanvas` document from a sequential-prescription JSON
/// string: the lens as revolved bodies, the stop as a declared aperture, one
/// source per field and a detector on the image plane (see
/// `oc_core::seq_import`). Returns the document and the notes on what it
/// holds only approximately. Raises `NotImportableError`, with every reason,
/// for a prescription it cannot build.
#[pyfunction]
#[pyo3(signature = (seq_json, rays_per_field = 64, pixels = 64))]
fn seq_to_scene3(
    seq_json: &str,
    rays_per_field: u32,
    pixels: usize,
) -> PyResult<(String, Vec<String>)> {
    let sys = SeqSystem::from_json(seq_json).map_err(to_py_err)?;
    let opts = oc_core::seq_import::ImportOptions {
        rays_per_field,
        pixels,
    };
    let built = oc_core::seq_import::import_seq(&sys, &opts)
        .map_err(|e| NotImportableError::new_err(e.to_string()))?;
    Ok((built.scene.to_json(), built.notes))
}

/// Round-trip an `.ocanvas` document through the scene model (parse + re-emit),
/// normalizing it and stamping the current header of **its own** version: a
/// v1 document stays v1 and a v2 document stays v2.
#[pyfunction]
fn normalize(ocanvas_json: &str) -> PyResult<String> {
    Ok(read_document(ocanvas_json).map_err(doc_err)?.to_json())
}

/// Read and validate an `.ocanvas` document of any readable version, and
/// return that version. Raises `NotADocumentError`, `UnsupportedVersionError`
/// or `InvalidSceneError` otherwise.
#[pyfunction]
fn validate(ocanvas_json: &str) -> PyResult<u32> {
    Ok(read_document(ocanvas_json).map_err(doc_err)?.version())
}

/// Extract the ordered surface chain the axial ray of `source_id` threads on
/// its way to `detector_id`, from an `.ocanvas` document (v1 documents migrate
/// on load). Returns `oc-seq3` prescription JSON.
///
/// Raises `ValueError` when the document does not parse, the ids are unknown,
/// the axial ray never reaches the detector (the message lists every branch's
/// fate), or it reaches it along more than one chain (the message lists the
/// chains).
#[pyfunction]
#[pyo3(signature = (ocanvas_json, source_id, detector_id, revision = 0))]
fn scene3_sequence(
    ocanvas_json: &str,
    source_id: u64,
    detector_id: u64,
    revision: u64,
) -> PyResult<String> {
    let (scene, _report) = read_document(ocanvas_json)
        .and_then(Document::into_scene3)
        .map_err(doc_err)?;
    oc_trace::sequence3::extract_sequence3(
        &scene,
        oc_core::id::SourceId(source_id),
        oc_core::id::DetectorId(detector_id),
        revision,
    )
    .map(|p| p.to_json())
    .map_err(to_py_err)
}

/// [`scene3_sequence`] for a declared primary-path intent: resolves the
/// intent's source/detector pair and stamps the intent id into the
/// prescription.
#[pyfunction]
#[pyo3(signature = (ocanvas_json, intent_id, revision = 0))]
fn scene3_sequence_for_intent(
    ocanvas_json: &str,
    intent_id: u64,
    revision: u64,
) -> PyResult<String> {
    let (scene, _report) = read_document(ocanvas_json)
        .and_then(Document::into_scene3)
        .map_err(doc_err)?;
    oc_trace::sequence3::extract_for_intent(
        &scene,
        oc_core::id::PathIntentId(intent_id),
        revision,
    )
    .map(|p| p.to_json())
    .map_err(to_py_err)
}

/// Evaluate a thin-film stack, for cross-checking against `optiland.thin_film`.
///
/// `stack_json` is a serialized `oc_core::thinfilm::Coating`. `n_far` is the
/// medium behind the stack, which only a `Medium`-backed coating reads; it
/// defaults to `n_incident`, a zero-thickness surface in one medium. Returns JSON
/// `{"rs": [...], "rp": [...], "ts": [...], "tp": [...]}`, one entry per
/// `(wavelength, angle)` pair, both flattened row-major over wavelengths then
/// angles. Raises `ValueError` if the stack does not parse or a layer's optical
/// constants have no value at one of the wavelengths.
///
/// This exists for the parity suite. The canvas evaluates coatings in Rust and
/// Optiland evaluates them in Python, and the only way "they agree" means
/// anything is if something runs both and subtracts.
#[pyfunction]
#[pyo3(signature = (stack_json, lambda_um, aoi_deg, n_incident, n_far=None))]
fn thinfilm_rt(
    stack_json: &str,
    lambda_um: Vec<f64>,
    aoi_deg: Vec<f64>,
    n_incident: f64,
    n_far: Option<f64>,
) -> PyResult<String> {
    let n_far = n_far.unwrap_or(n_incident);
    let coating: oc_core::thinfilm::Coating =
        serde_json::from_str(stack_json).map_err(to_py_err)?;
    let mut cols: [Vec<f64>; 4] = Default::default();
    for &nm in &lambda_um {
        for &deg in &aoi_deg {
            let rt = oc_core::thinfilm::coating_rt_at(
                &coating,
                n_incident,
                n_far,
                nm,
                deg.to_radians().cos(),
            )
            .map_err(|e| PyValueError::new_err(format!("at {nm} um, {deg} deg: {e}")))?;
            cols[0].push(rt.r_s);
            cols[1].push(rt.r_p);
            cols[2].push(rt.t_s);
            cols[3].push(rt.t_p);
        }
    }
    let arr = |v: &Vec<f64>| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    Ok(format!(
        "{{\"rs\":[{}],\"rp\":[{}],\"ts\":[{}],\"tp\":[{}]}}",
        arr(&cols[0]),
        arr(&cols[1]),
        arr(&cols[2]),
        arr(&cols[3])
    ))
}

/// Evaluate a thin-film stack as **complex amplitudes**, for cross-checking the
/// polarization conventions against `optiland.thin_film`.
///
/// Returns JSON with eight arrays: `rs_re`, `rs_im`, `rp_re`, `rp_im` and the
/// same for `ts`/`tp`, flattened the same way as [`thinfilm_rt`], plus `scale`,
/// the factor turning `|t|²` into a transmittance, and `psi`, `delta` (the
/// ellipsometric angles, `FilmAmp::ellipsometric`) and `t_retardance`
/// (`arg t_p − arg t_s`), radians.
///
/// These are in the **kernel's ray frame** (see `oc_core::polarization`), not
/// Macleod's: `r_p` carries the opposite sign, `t_p` is the whole field rather
/// than its tangential part, and every amplitude is conjugated into the
/// `exp(-iωt)` convention. The parity test reconciles those three steps
/// explicitly rather than comparing numbers that happen to be close.
#[pyfunction]
fn thinfilm_amplitudes(
    stack_json: &str,
    lambda_um: Vec<f64>,
    aoi_deg: Vec<f64>,
    n_incident: f64,
    n_exit: f64,
) -> PyResult<String> {
    let coating: oc_core::thinfilm::Coating =
        serde_json::from_str(stack_json).map_err(to_py_err)?;
    let mut cols: [Vec<f64>; 12] = Default::default();
    for &nm in &lambda_um {
        for &deg in &aoi_deg {
            let a = oc_core::thinfilm::coating_amp_at(
                &coating,
                n_incident,
                n_exit,
                nm,
                deg.to_radians().cos(),
            )
            .map_err(|e| PyValueError::new_err(format!("at {nm} um, {deg} deg: {e}")))?;
            for (i, z) in [a.r_s, a.r_p, a.t_s, a.t_p].into_iter().enumerate() {
                cols[2 * i].push(z.re);
                cols[2 * i + 1].push(z.im);
            }
            cols[8].push(a.t_power_scale);
            let (psi, delta) = a.ellipsometric();
            cols[9].push(psi);
            cols[10].push(delta);
            cols[11].push(a.transmission_retardance());
        }
    }
    let arr = |v: &Vec<f64>| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let names = [
        "rs_re", "rs_im", "rp_re", "rp_im", "ts_re", "ts_im", "tp_re", "tp_im", "scale", "psi", "delta",
        "t_retardance",
    ];
    let body = names
        .iter()
        .zip(cols.iter())
        .map(|(n, v)| format!("\"{n}\":[{}]", arr(v)))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!("{{{body}}}"))
}

/// Each layer's complex index `n + ik` at each wavelength, as the kernel
/// samples it, for feeding the oracles.
///
/// Returns JSON `{"n_re": [[...]], "n_im": [[...]], "sub": [[re, im], ...]}`,
/// one row per wavelength and one column per layer in the **written** order;
/// `sub` is the substrate's index per wavelength for a `Plate` or `Opaque`
/// backing, and `null` for `Medium`, whose far side is the scene's. The parity suites hand
/// these numbers to `tmm` and Optiland unchanged, so that a disagreement is the
/// solver's and never a second implementation of the material lookup.
#[pyfunction]
fn thinfilm_indices(stack_json: &str, lambda_um: Vec<f64>) -> PyResult<String> {
    let coating: oc_core::thinfilm::Coating =
        serde_json::from_str(stack_json).map_err(to_py_err)?;
    let mut re_rows = Vec::with_capacity(lambda_um.len());
    let mut im_rows = Vec::with_capacity(lambda_um.len());
    let substrate = match &coating.backing {
        oc_core::thinfilm::CoatingBacking::Medium => None,
        oc_core::thinfilm::CoatingBacking::Plate { substrate, .. }
        | oc_core::thinfilm::CoatingBacking::Opaque { substrate } => Some(substrate),
    };
    let mut sub_rows = Vec::with_capacity(lambda_um.len());
    for &um in &lambda_um {
        if let Some(m) = substrate {
            let n = m
                .index_at(um)
                .ok_or_else(|| PyValueError::new_err(format!("no optical constants for the substrate at {um} um")))?;
            sub_rows.push(format!("[{},{}]", n.re, n.im));
        }
        let mut re = Vec::with_capacity(coating.layers.len());
        let mut im = Vec::with_capacity(coating.layers.len());
        for l in &coating.layers {
            let n = l.material.index_at(um).ok_or_else(|| {
                PyValueError::new_err(format!("no optical constants for {} at {um} um", l.material.name))
            })?;
            re.push(n.re.to_string());
            im.push(n.im.to_string());
        }
        re_rows.push(format!("[{}]", re.join(",")));
        im_rows.push(format!("[{}]", im.join(",")));
    }
    let sub = if substrate.is_some() { format!("[{}]", sub_rows.join(",")) } else { "null".to_string() };
    Ok(format!(
        "{{\"n_re\":[{}],\"n_im\":[{}],\"sub\":{}}}",
        re_rows.join(","),
        im_rows.join(","),
        sub
    ))
}

/// A coating's absorption by layer and substrate (coating-fidelity spec
/// CF-04), for cross-checking against `tmm.absorp_in_each_layer`.
///
/// Returns JSON `{"a_s": [[...]], "a_p": [[...]], "sub_s": [...], "sub_p":
/// [...], "rs": [...], ...}`: for each `(wavelength, angle)` pair, flattened
/// as [`thinfilm_rt`] flattens them, the fraction of the incident power each
/// layer absorbs (written order), what the substrate takes, and the powers.
#[pyfunction]
#[pyo3(signature = (stack_json, lambda_um, aoi_deg, n_incident, n_far=None, plate_incoherent=false))]
fn thinfilm_absorption(
    stack_json: &str,
    lambda_um: Vec<f64>,
    aoi_deg: Vec<f64>,
    n_incident: f64,
    n_far: Option<f64>,
    plate_incoherent: bool,
) -> PyResult<String> {
    let coating: oc_core::thinfilm::Coating =
        serde_json::from_str(stack_json).map_err(to_py_err)?;
    let n_far = n_far.unwrap_or(n_incident);
    let ctx = oc_core::thinfilm::FilmContext {
        plate: if plate_incoherent {
            oc_core::fidelity::PlateModel::Incoherent
        } else {
            oc_core::fidelity::PlateModel::PhaseFree
        },
        ..Default::default()
    };
    let (mut a_s, mut a_p, mut sub_s, mut sub_p) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut rs, mut rp, mut ts, mut tp) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &um in &lambda_um {
        for &deg in &aoi_deg {
            let (p, _) = oc_core::thinfilm::coating_power_flow_eval(&coating, n_incident, n_far, um, deg.to_radians().cos(), &ctx)
                .map_err(|e| PyValueError::new_err(format!("at {um} um, {deg} deg: {e}")))?;
            a_s.push(p.a_layer_s.clone());
            a_p.push(p.a_layer_p.clone());
            sub_s.push(p.a_substrate_s);
            sub_p.push(p.a_substrate_p);
            rs.push(p.rt.r_s);
            rp.push(p.rt.r_p);
            ts.push(p.rt.t_s);
            tp.push(p.rt.t_p);
        }
    }
    serde_json::to_string(&serde_json::json!({
        "a_s": a_s, "a_p": a_p, "sub_s": sub_s, "sub_p": sub_p,
        "rs": rs, "rp": rp, "ts": ts, "tp": tp,
    }))
    .map_err(to_py_err)
}

/// The built-in film catalogue as JSON: `[{"key": ..., "material": {...}}]`,
/// each material in the document's own form (curves, formula, form,
/// provenance, nominal), for the parity suites and for building stacks.
#[pyfunction]
fn film_catalog_json() -> String {
    let rows: Vec<String> = oc_core::film_catalog::builtin()
        .into_iter()
        .map(|(key, m)| format!("{{\"key\":\"{key}\",\"material\":{}}}", serde_json::to_string(&m).expect("serializes")))
        .collect();
    format!("[{}]", rows.join(","))
}

/// The `.ocanvas` version [`seq_to_scene`] writes. The browser writes the
/// later one; [`READABLE_VERSIONS`] is every version this build reads.
#[pyfunction]
fn format_version() -> u32 {
    OCANVAS_VERSION
}

/// The `.ocanvas` format magic string.
#[pyfunction]
fn format_magic() -> &'static str {
    OCANVAS_FORMAT
}

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Since pyo3 0.28 a `#[pymodule]` declares free-threaded support by default
/// (`Py_MOD_GIL_NOT_USED`), where older versions declared nothing. The 0.23 → 0.29
/// security upgrade therefore turned that claim on without touching this file,
/// so it is audited here rather than left to a default nobody read.
///
/// The claim holds. Every function below is a pure `&str` → `String` call into
/// `oc-core`: no `#[pyclass]`, no statics, no interior mutability, no `unsafe`,
/// and no state that outlives a call. There is nothing for two threads to race
/// on.
///
/// It is also moot for the wheel we ship: `abi3-py39` targets the stable ABI, and
/// free-threaded CPython has its own (`cp313t`), so this build is loaded by
/// GIL-enabled interpreters only. If that changes, the audit above is the thing
/// to recheck, and `#[pymodule(gil_used = true)]` is the opt-out.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scene_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(seq_to_scene, m)?)?;
    m.add_function(wrap_pyfunction!(seq_to_scene3, m)?)?;
    m.add_function(wrap_pyfunction!(scene3_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(scene3_sequence_for_intent, m)?)?;
    m.add_function(wrap_pyfunction!(thinfilm_rt, m)?)?;
    m.add_function(wrap_pyfunction!(thinfilm_amplitudes, m)?)?;
    m.add_function(wrap_pyfunction!(thinfilm_indices, m)?)?;
    m.add_function(wrap_pyfunction!(film_catalog_json, m)?)?;
    m.add_function(wrap_pyfunction!(thinfilm_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(format_version, m)?)?;
    m.add_function(wrap_pyfunction!(format_magic, m)?)?;
    m.add("OCANVAS_VERSION", OCANVAS_VERSION)?;
    m.add("OCANVAS_FORMAT", OCANVAS_FORMAT)?;
    m.add("READABLE_VERSIONS", READABLE_VERSIONS.to_vec())?;
    let py = m.py();
    m.add("OcanvasError", py.get_type::<OcanvasError>())?;
    m.add("NotADocumentError", py.get_type::<NotADocumentError>())?;
    m.add(
        "UnsupportedVersionError",
        py.get_type::<UnsupportedVersionError>(),
    )?;
    m.add("InvalidSceneError", py.get_type::<InvalidSceneError>())?;
    m.add("NotExportableError", py.get_type::<NotExportableError>())?;
    m.add("NotImportableError", py.get_type::<NotImportableError>())?;
    Ok(())
}
