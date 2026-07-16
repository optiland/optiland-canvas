//! The folded-singlet demo (§16.4), headless and built from catalogue JSON.
//!
//! The C2 acceptance scenario, and the first end-to-end evidence that a user can
//! assemble openUC2 modules and get real light out. Everything here goes through
//! the public API: if this compiles, the crate is usable from outside.
//!
//! ```text
//!   grid    [0,0,0]      [1,0,0]       [2,0,0]
//!           source  -->  singlet  -->  45° mirror
//!                                          |
//!                                          v  (folded onto -y)
//!                                      detector  [2,-1,0], rotated 270°
//! ```
//!
//! The detector module is rotated on purpose: its sensor is authored across the
//! module's own axis, so catching a folded beam requires the placement rotation
//! to reach the detector's endpoints. A module whose optics did not rotate with
//! it would miss the beam entirely.
//!
//! It is rotated *270°* rather than 90° because of where its window is. A bare
//! line detector has no sides, so both rotations lay the same sensor across the
//! same beam and trace identically — but the module takes light in through its
//! `-X` face, and only 270° turns that face toward the mirror. At 90° the camera
//! is mounted backwards: the numbers come out right and the port graph reports
//! its input unconnected, which is precisely the mistake ports exist to catch.

use oc_openuc2::test_support::*;
use oc_openuc2::{OpenUc2Project, ProjectWarning, Quadrant, SceneObjectId};
use oc_trace::{trace, TraceConfig};

/// Bounded to the scene: the demo spans ~150 mm, and the 1e4 mm default would
/// draw escaping rays far past anything worth asserting about.
fn config() -> TraceConfig {
    TraceConfig {
        bounds: 400.0,
        ..TraceConfig::default()
    }
}

/// Assemble the demo from the catalogue.
///
/// The same definition the committed `.uc2canvas` fixture is emitted from, so
/// everything asserted here is asserted about the file a user actually opens.
fn folded_singlet() -> OpenUc2Project {
    folded_singlet_demo()
}

/// The committed demo setup (§16.4).
fn committed_demo() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/openuc2/folded-singlet-demo.uc2canvas");
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "the committed demo is missing at {}: {e}\n\
             regenerate it with:\n  cargo run -p oc-openuc2 --example emit_folded_singlet \
             --features test-support -- examples/openuc2/folded-singlet-demo.uc2canvas",
            path.display()
        )
    })
}

#[test]
fn the_demo_assembles_from_catalogue_json() {
    let p = folded_singlet();
    assert_eq!(p.setup().instances.len(), 4);
    // Source, lens, mirror, detector: two elements, one source, one detector.
    assert_eq!(p.scene().elements.len(), 2, "the singlet and the mirror");
    assert_eq!(p.scene().sources.len(), 1);
    assert_eq!(p.scene().detectors.len(), 1);
}

#[test]
fn light_reaches_the_detector() {
    let p = folded_singlet();
    let out = trace(p.scene(), &config());
    let det = &out.detectors[0];
    assert!(det.hits > 0, "no light reached the detector");
    assert!(det.total_flux > 0.0);
    // Every launched ray is a (spatial sample x wavelength) pair: 7 x 3.
    assert!(det.hits >= 21, "expected the whole beam, got {} hits", det.hits);
}

#[test]
fn the_beam_is_actually_folded() {
    // The detector sits off the source's axis, so flux arriving at all means the
    // mirror really did turn the beam through 90 degrees.
    let p = folded_singlet();
    let out = trace(p.scene(), &config());

    // The detector spans x = 80..120 at y = -50, well off the y = 0 source axis.
    let d = &p.scene().detectors[0];
    assert!((d.a.y + 50.0).abs() < 1e-12 && (d.b.y + 50.0).abs() < 1e-12);
    assert!(out.detectors[0].total_flux > 0.0);

    // And some ray really travels in -y to get there.
    let folded = out
        .segments
        .iter()
        .any(|s| (s.b - s.a).normalize().y < -0.99);
    assert!(folded, "no segment travels along -y");
}

#[test]
fn the_singlet_refracts_and_disperses() {
    let p = folded_singlet();
    let out = trace(p.scene(), &config());

    // All three source lines propagate...
    let lines: std::collections::BTreeSet<u64> = out
        .segments
        .iter()
        .map(|s| (s.lambda_um * 1e4).round() as u64)
        .collect();
    assert_eq!(lines.len(), 3, "expected F, d, and C lines");

    // ...and they do not land in the same place, which is dispersion doing its
    // job rather than a monochromatic trace wearing three labels.
    let centroid = |lambda: f64| -> f64 {
        let hits: Vec<_> = out
            .segments
            .iter()
            .filter(|s| (s.lambda_um - lambda).abs() < 1e-6)
            .collect();
        hits.iter().map(|s| s.b.x).sum::<f64>() / hits.len() as f64
    };
    assert!(
        (centroid(0.4861) - centroid(0.6563)).abs() > 1e-9,
        "F and C should not overlap exactly"
    );
}

#[test]
fn the_detector_modules_endpoints_follow_its_placement() {
    // A `Detector` carries bare endpoints and no transform of its own, so it is
    // the one geometry a placement has to move directly rather than compose onto.
    // Assert the endpoints, not the flux: the beam is converging toward focus, so
    // a sensor lying *along* it is still crossed by rays on their way in and
    // would collect a plausible-looking number.
    let mut p = folded_singlet();
    let detector = p
        .setup()
        .instances
        .iter()
        .find(|i| i.fallback.module_id == "detector-line-1x1")
        .map(|i| i.instance_id.clone())
        .expect("the detector instance");

    // Rotated 270°, the sensor lies across the folded beam: horizontal at y = -50,
    // spanning the module's cell.
    let d = p.scene().detectors[0].clone();
    assert!((d.a - oc_core::Vec2::new(80.0, -50.0)).length() < 1e-12, "{d:?}");
    assert!((d.b - oc_core::Vec2::new(120.0, -50.0)).length() < 1e-12, "{d:?}");
    let across = trace(p.scene(), &config()).detectors[0].total_flux;

    // Unrotated, the very same module's sensor runs along the beam instead:
    // vertical at x = 100.
    p.rotate_module(&detector, Quadrant::R0).unwrap();
    let d = p.scene().detectors[0].clone();
    assert!((d.a - oc_core::Vec2::new(100.0, -70.0)).length() < 1e-12, "{d:?}");
    assert!((d.b - oc_core::Vec2::new(100.0, -30.0)).length() < 1e-12, "{d:?}");
    let along = trace(p.scene(), &config()).detectors[0].total_flux;

    // And it intercepts far less of the beam that way.
    assert!(
        across > 2.0 * along,
        "across the beam collected {across}, along it {along}"
    );
}

#[test]
fn every_port_in_the_demo_is_connected() {
    // The chain is source -> lens -> mirror -> detector, and on a 50 mm grid each
    // pair of neighbours mates on the cell boundary between them. Six ports, three
    // connections, nothing left over.
    let p = folded_singlet();
    let ports = p.ports();
    assert_eq!(ports.len(), 6);
    assert!(
        ports.iter().all(|q| q.status.is_connected()),
        "unconnected: {:?}",
        ports
            .iter()
            .filter(|q| !q.status.is_connected())
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_camera_mounted_backwards_is_reported_but_still_traces() {
    // 90° instead of 270°. A bare line detector has no sides, so the beam lands on
    // it exactly as before — the flux is identical, and nothing about the picture
    // or the numbers looks wrong. The port graph is the only thing that can say
    // the module is facing away from the light.
    let mut p = folded_singlet();
    let detector = p
        .setup()
        .instances
        .iter()
        .find(|i| i.fallback.module_id == "detector-line-1x1")
        .map(|i| i.instance_id.clone())
        .expect("the detector instance");

    let before = trace(p.scene(), &config()).detectors[0].total_flux;
    p.rotate_module(&detector, Quadrant::R90).unwrap();
    let after = trace(p.scene(), &config()).detectors[0].total_flux;
    assert_eq!(before, after, "the physics cannot tell the difference");

    let free: Vec<_> = p
        .ports()
        .into_iter()
        .filter(|q| !q.status.is_connected())
        .collect();
    assert_eq!(free.len(), 2, "the mirror's output and the camera's window");
    assert!(free.iter().any(|q| q.instance == detector));
}

#[test]
fn the_committed_demo_still_matches_the_code() {
    // The fixture is generated, not hand-written, so this is what stops it drifting
    // away from the demo every other test here proves works. If this fails, the
    // emitter needs re-running rather than the file needs editing.
    assert_eq!(
        committed_demo(),
        folded_singlet().save(),
        "examples/openuc2/folded-singlet-demo.uc2canvas is stale; re-run \
         `cargo run -p oc-openuc2 --example emit_folded_singlet --features test-support \
         -- examples/openuc2/folded-singlet-demo.uc2canvas`"
    );
}

#[test]
fn the_committed_demo_opens_and_traces() {
    // The user-facing path: open the file, get the light. Nothing here builds the
    // demo in code — it comes off disk, resolves against the catalogue, and traces.
    let p = OpenUc2Project::load(pilot_catalog(), &committed_demo()).expect("the demo opens");
    assert_eq!(p.setup().instances.len(), 4);

    let out = trace(p.scene(), &config());
    assert!(out.detectors[0].hits >= 21, "the demo's light reaches its detector");
    assert!(p.ports().iter().all(|q| q.status.is_connected()));
    assert_eq!(p.bom().lines.len(), 4);
}

#[test]
fn the_demo_round_trips_through_uc2canvas() {
    let p = folded_singlet();
    let saved = p.save();
    let reopened = OpenUc2Project::load(pilot_catalog(), &saved).expect("reopens");

    assert_eq!(reopened.save(), saved, "the file is stable");
    assert_eq!(reopened.setup().instances.len(), 4);
    // The reopened scene is not merely similar: it is the same scene, because it
    // is derived from the same setup.
    assert_eq!(reopened.scene().to_json(), p.scene().to_json());

    let before = trace(p.scene(), &config()).detectors[0].total_flux;
    let after = trace(reopened.scene(), &config()).detectors[0].total_flux;
    assert_eq!(before, after);
}

#[test]
fn the_flat_export_traces() {
    let p = folded_singlet();
    let export = p.export_flat_ocanvas().expect("exports");
    let scene = oc_core::Scene::from_json(&export.ocanvas).expect("valid .ocanvas");

    // The exported file stands alone: it traces to the same numbers without the
    // catalogue or the setup being anywhere in sight.
    let direct = trace(p.scene(), &config()).detectors[0].total_flux;
    let flat = trace(&scene, &config()).detectors[0].total_flux;
    assert!(flat > 0.0);
    assert_eq!(flat, direct);
}

#[test]
fn the_export_declares_its_idealized_parts() {
    // The source and the detector are idealized, and the demo must say so rather
    // than let the numbers look better founded than they are.
    let p = folded_singlet();
    let export = p.export_flat_ocanvas().expect("exports");
    let idealized: Vec<_> = export
        .warnings
        .iter()
        .filter(|w| matches!(w, ProjectWarning::IdealizedOptics { .. }))
        .collect();
    assert_eq!(idealized.len(), 2, "the source and the detector");
}

#[test]
fn the_demo_produces_a_bom() {
    let p = folded_singlet();
    let bom = p.bom();
    assert_eq!(bom.lines.len(), 4, "one line per distinct product");
    assert!(bom.lines.iter().all(|l| l.quantity == 1 && l.resolved));
    assert_eq!(bom.total_eur, 40.0 + 24.0 + 18.0 + 120.0);
    assert_eq!(bom.custom_object_count, 0);
}

#[test]
fn selecting_any_object_selects_its_whole_module() {
    let mut p = folded_singlet();
    // The mirror is one element among several objects; picking it must name the
    // module, not the element.
    let mirror = p
        .setup()
        .instances
        .iter()
        .find(|i| i.fallback.module_id == "mirror-flat-45-1x1")
        .map(|i| i.instance_id.clone())
        .unwrap();
    let element = p.instance(&mirror).unwrap().inserted.elements[0];
    assert_eq!(
        p.select_object(SceneObjectId::Element(element.0)),
        Some(mirror.as_str())
    );
}

#[test]
fn removing_the_lens_leaves_the_rest_of_the_demo_intact() {
    // The whole id-remap contract, seen from the outside: drop a module from the
    // middle and the survivors must still be themselves.
    let mut p = folded_singlet();
    let lens = p
        .setup()
        .instances
        .iter()
        .find(|i| i.fallback.module_id == "lens-pcx-nbk7-1x1")
        .map(|i| i.instance_id.clone())
        .unwrap();

    p.remove_module(&lens).unwrap();

    assert_eq!(p.scene().elements.len(), 1, "only the mirror is left");
    assert_eq!(p.setup().instances.len(), 3);
    // Light still gets to the detector: without the lens it is collimated rather
    // than focused, but the fold is untouched.
    let out = trace(p.scene(), &config());
    assert!(out.detectors[0].hits > 0, "the fold still works");
    assert_eq!(p.bom().lines.len(), 3);
}
