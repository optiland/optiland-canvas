//! Write the development fixture catalogue the browser loads.
//!
//! ```sh
//! cargo run -p oc-openuc2 --example emit_dev_catalog -- web/public/catalog/openuc2
//! ```
//!
//! **This emits invented data, not the openUC2 catalogue.** The real one is
//! produced by the module repository's compiler (`build_canvas_catalog.py`, PR
//! M1) from actual `module.yaml` descriptors, and drops into the same directory
//! with the same shape. This exists so the openUC2 mode has something to show
//! before that lands, and so the loading path is exercised end to end rather
//! than mocked.
//!
//! Everything comes from `test_support`, so the fixture the UI loads and the
//! fixture the Rust tests assert against cannot drift apart.

use std::fs;
use std::path::PathBuf;

use oc_openuc2::test_support;

fn main() -> std::io::Result<()> {
    let out: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "web/public/catalog/openuc2".to_string())
        .into();
    let assets = out.join("assets");
    fs::create_dir_all(&assets)?;

    fs::write(out.join("catalog.json"), test_support::minimal_catalog_json())?;
    for (name, body) in [
        ("source.ocanvas", test_support::source_fragment()),
        ("lens.ocanvas", test_support::lens_fragment()),
        ("mirror.ocanvas", test_support::mirror_fragment()),
        ("detector.ocanvas", test_support::detector_fragment()),
        ("aperture.ocanvas", test_support::aperture_fragment()),
    ] {
        fs::write(assets.join(name), body)?;
    }

    println!("wrote the development fixture catalogue to {}", out.display());
    println!("this is placeholder data, not real openUC2 module data");
    Ok(())
}
