//! Write the committed folded-singlet demo setup (§16.4).
//!
//! ```sh
//! cargo run -p oc-openuc2 --example emit_folded_singlet --features test-support \
//!   -- examples/openuc2/folded-singlet-demo.uc2canvas
//! ```
//!
//! Not named `emit_demo_setup`: Windows' installer-detection heuristic sees
//! "setup" in an executable's name and demands elevation, so that binary cannot
//! be run by `cargo run` at all on a developer machine.
//!
//! The demo is defined once, in `test_support::folded_singlet_demo`, and emitted
//! from there. `the_committed_demo_still_matches_the_code` re-reads the file this
//! writes and checks it against that definition, so the fixture cannot quietly
//! drift away from the layout the tests prove works.
//!
//! The setup it writes is pinned to the **development fixture** catalogue, whose
//! commit is all zeros. It resolves against that fixture and nothing else; the
//! real catalogue (PR M1) carries real UUIDs, and a setup for it is a different
//! file.

use std::fs;
use std::path::PathBuf;

use oc_openuc2::test_support;

fn main() -> std::io::Result<()> {
    let out: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/openuc2/folded-singlet-demo.uc2canvas".to_string())
        .into();
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }

    let demo = test_support::folded_singlet_demo();
    // Deterministic bytes: every map is ordered and no timestamp is stamped, so
    // re-running this on an unchanged tree produces an unchanged file (§12.3.9).
    fs::write(&out, demo.save())?;

    println!("wrote the folded-singlet demo to {}", out.display());
    println!(
        "{} instances, {} BOM lines",
        demo.setup().instances.len(),
        demo.bom().lines.len()
    );
    Ok(())
}
