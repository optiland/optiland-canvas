//! PyO3 bindings — the `_native` submodule of the `optiland_canvas` package.
//!
//! This layer is deliberately thin: **all** of the optical geometry lives in
//! `oc-core` (the [`SeqSystem`](oc_core::SeqSystem) bridge). Python only turns
//! the neutral sequential prescription into an Optiland `Optic` and back, so
//! there is exactly one implementation of "what a lens is" and it is in Rust.
//!
//! The three exchanges crossing this boundary are all JSON strings:
//! - [`scene_to_seq`] `.ocanvas` → sequential-prescription JSON (+ fidelity notes)
//! - [`seq_to_scene`] sequential-prescription JSON → `.ocanvas`
//! - [`validate`] parse-and-version-check an `.ocanvas` document

use oc_core::{Scene, SeqSystem, OCANVAS_FORMAT, OCANVAS_VERSION};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Extract the rotationally-symmetric sequential prescription from an
/// `.ocanvas` document. Returns `SeqSystem` JSON, whose `notes` array reports
/// anything that could not be represented exactly.
#[pyfunction]
fn scene_to_seq(ocanvas_json: &str) -> PyResult<String> {
    let scene = Scene::from_json(ocanvas_json).map_err(to_py_err)?;
    Ok(scene.to_seq().to_json())
}

/// Rebuild an `.ocanvas` document from a sequential-prescription JSON string.
#[pyfunction]
fn seq_to_scene(seq_json: &str) -> PyResult<String> {
    let sys = SeqSystem::from_json(seq_json).map_err(to_py_err)?;
    Ok(sys.to_scene().to_json())
}

/// Round-trip an `.ocanvas` document through the scene model (parse + re-emit),
/// normalizing it and stamping the current format header.
#[pyfunction]
fn normalize(ocanvas_json: &str) -> PyResult<String> {
    let scene = Scene::from_json(ocanvas_json).map_err(to_py_err)?;
    Ok(scene.to_json())
}

/// Parse and version-check an `.ocanvas` document, raising `ValueError` if it is
/// not a readable document. Returns its declared version.
#[pyfunction]
fn validate(ocanvas_json: &str) -> PyResult<u32> {
    let scene = Scene::from_json(ocanvas_json).map_err(to_py_err)?;
    Ok(scene.version)
}

/// The `.ocanvas` schema version this build writes.
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scene_to_seq, m)?)?;
    m.add_function(wrap_pyfunction!(seq_to_scene, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(format_version, m)?)?;
    m.add_function(wrap_pyfunction!(format_magic, m)?)?;
    m.add("OCANVAS_VERSION", OCANVAS_VERSION)?;
    m.add("OCANVAS_FORMAT", OCANVAS_FORMAT)?;
    Ok(())
}
