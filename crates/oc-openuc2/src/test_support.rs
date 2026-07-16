//! A small in-memory catalogue with the pilot module roles (§16.2).
//!
//! Fragments are built as real [`Scene`]s and serialized, so a test travels the
//! same path a shipped module does: `.ocanvas` text in, parsed to a fragment,
//! placed. Nothing here shortcuts into the adapter's internals.
//!
//! The modules are laid out so the folded-singlet demo works on the 50 mm grid:
//! a three-line collimated source, an N-BK7 singlet, a 45° fold mirror, a line
//! detector across the module's own axis, an aperture, and an empty cube that
//! carries no optics at all.

use oc_core::{
    spherical_edge, Detector, Edge, Element, ElementId, Isometry2, Material, Scene, Source,
    SourceKind, Spectrum, Vec2,
};

pub const SOURCE_UUID: &str = "11111111-1111-4111-8111-000000000001";
pub const LENS_UUID: &str = "11111111-1111-4111-8111-000000000002";
pub const MIRROR_UUID: &str = "11111111-1111-4111-8111-000000000003";
pub const DETECTOR_UUID: &str = "11111111-1111-4111-8111-000000000004";
pub const CUBE_UUID: &str = "11111111-1111-4111-8111-000000000005";

/// Sag of a spherical surface at height `y`.
fn sag(c: f64, y: f64) -> f64 {
    let d = 1.0 - c * c * y * y;
    c * y * y / (1.0 + d.max(0.0).sqrt())
}

/// A three-line collimated source aimed along the module's +x axis.
pub fn source_fragment() -> String {
    let mut scene = Scene::new();
    scene.sources.push(Source {
        kind: SourceKind::Collimated { half_width: 6.0 },
        transform: Isometry2::from_pos(Vec2::new(-15.0, 0.0)),
        // F, d, and C lines: enough to show dispersion through the singlet.
        spectrum: Spectrum::Lines(vec![(0.4861, 1.0), (0.5876, 1.0), (0.6563, 1.0)]),
        flux: 1.0,
        rays: 7,
    });
    scene.to_json()
}

/// A biconvex N-BK7 singlet centred on the module anchor.
///
/// Deliberately a `Material::Catalog` glass: the fragment carries the builtin
/// catalogue's N-BK7, so placing it exercises the glass-id remap that a naive
/// insertion would get wrong.
pub fn lens_fragment() -> String {
    let mut scene = Scene::new();
    let nbk7 = scene.catalog.find("N-BK7").expect("builtin N-BK7");
    let (r1, r2, thickness, half_ap) = (100.0, -100.0, 5.0, 12.0);
    let (zf, zr) = (-thickness / 2.0, thickness / 2.0);
    let front = spherical_edge(Vec2::new(zf, 0.0), Vec2::X, r1, half_ap);
    let rear = spherical_edge(Vec2::new(zr, 0.0), Vec2::X, r2, half_ap);
    let ftip = zf + sag(1.0 / r1, half_ap);
    let rtip = zr + sag(1.0 / r2, half_ap);
    let boundary = vec![
        front,
        Edge::Line {
            a: Vec2::new(ftip, half_ap),
            b: Vec2::new(rtip, half_ap),
        },
        rear,
        Edge::Line {
            a: Vec2::new(rtip, -half_ap),
            b: Vec2::new(ftip, -half_ap),
        },
    ];
    scene.add_element(Element::new(
        ElementId(0),
        boundary,
        Material::Catalog(nbk7),
    ));
    scene.to_json()
}

/// A flat mirror at 45°, folding the module's +x axis onto -y.
pub fn mirror_fragment() -> String {
    let mut scene = Scene::new();
    let (hx, hy) = (0.75, 14.0);
    let boundary = vec![
        Edge::Line {
            a: Vec2::new(-hx, -hy),
            b: Vec2::new(hx, -hy),
        },
        Edge::Line {
            a: Vec2::new(hx, -hy),
            b: Vec2::new(hx, hy),
        },
        Edge::Line {
            a: Vec2::new(hx, hy),
            b: Vec2::new(-hx, hy),
        },
        Edge::Line {
            a: Vec2::new(-hx, hy),
            b: Vec2::new(-hx, -hy),
        },
    ];
    let mut e = Element::new(
        ElementId(0),
        boundary,
        Material::Mirror { reflectivity: 0.96 },
    );
    // The module's own pose, which placement composes onto rather than replaces.
    e.transform = Isometry2::new(Vec2::ZERO, std::f64::consts::FRAC_PI_4);
    scene.add_element(e);
    scene.to_json()
}

/// A line detector across the module's axis, facing -x.
///
/// Odd bin count: the demo is symmetric about the axis, and an even count would
/// put the axial ray exactly on a bin boundary.
pub fn detector_fragment() -> String {
    let mut scene = Scene::new();
    scene
        .detectors
        .push(Detector::new(Vec2::new(0.0, -20.0), Vec2::new(0.0, 20.0), 15));
    scene.to_json()
}

/// An absorbing aperture with a clear centre: two blades, not a hole.
pub fn aperture_fragment() -> String {
    let mut scene = Scene::new();
    for sign in [1.0f64, -1.0] {
        let (inner, outer) = (6.0 * sign, 20.0 * sign);
        let boundary = vec![
            Edge::Line {
                a: Vec2::new(-1.0, inner),
                b: Vec2::new(1.0, inner),
            },
            Edge::Line {
                a: Vec2::new(1.0, inner),
                b: Vec2::new(1.0, outer),
            },
            Edge::Line {
                a: Vec2::new(1.0, outer),
                b: Vec2::new(-1.0, outer),
            },
            Edge::Line {
                a: Vec2::new(-1.0, outer),
                b: Vec2::new(-1.0, inner),
            },
        ];
        scene.add_element(Element::new(ElementId(0), boundary, Material::Absorber));
    }
    scene.to_json()
}

/// A catalogue over the pilot modules, with fragment assets attached.
pub fn pilot_catalog() -> crate::Catalog {
    let mut cat = crate::Catalog::from_json(&minimal_catalog_json()).expect("fixture parses");
    cat.attach_fragment("assets/source.ocanvas", &source_fragment())
        .expect("source fragment");
    cat.attach_fragment("assets/lens.ocanvas", &lens_fragment())
        .expect("lens fragment");
    cat.attach_fragment("assets/mirror.ocanvas", &mirror_fragment())
        .expect("mirror fragment");
    cat.attach_fragment("assets/detector.ocanvas", &detector_fragment())
        .expect("detector fragment");
    cat
}

/// The catalogue document for [`pilot_catalog`].
pub fn minimal_catalog_json() -> String {
    // Written as JSON rather than built through the structs: this is the
    // generated artifact the browser and the tests both consume, so the fixture
    // should break if the shape drifts.
    format!(
        r##"{{
  "format": "openuc2-canvas-catalog",
  "version": "1.0.0",
  "source": {{
    "repository": "openUC2/openUC2-optikit-modules",
    "commit": "0000000000000000000000000000000000000000"
  }},
  "descriptor_schema_version": "2.0.0",
  "grid_profiles": [
    {{
      "id": "openuc2-core-50",
      "pitch_mm": [50.0, 50.0, 55.0],
      "cell_size_mm": [50.0, 50.0, 50.0],
      "mounting_plane_offset_mm": 5.0,
      "allowed_rotations_deg": [0, 90, 180, 270]
    }}
  ],
  "modules": [
    {{
      "id": "lens-pcx-nbk7-1x1",
      "uuid": "{LENS_UUID}",
      "version": "1.0.0",
      "name": "N-BK7 singlet lens module",
      "category": "lens",
      "group": "lenses",
      "tags": ["lens", "singlet", "nbk7"],
      "color": "#7CC142",
      "footprint": {{"anchor_cell": [0, 0, 0], "cells": [[0, 0, 0]]}},
      "ports": [
        {{"id": "optical-in", "type": "optical", "role": "input", "face": "-X",
          "pose": {{"position_mm": [-25.0, 0.0, 25.0], "direction": [-1.0, 0.0, 0.0]}}}},
        {{"id": "optical-out", "type": "optical", "role": "output", "face": "+X",
          "pose": {{"position_mm": [25.0, 0.0, 25.0], "direction": [1.0, 0.0, 0.0]}}}}
      ],
      "capabilities": {{"layout": "ready", "optical": "validated", "cad": "reference", "control": "absent"}},
      "assets": {{"icon": "assets/lens.svg", "canvas_fragment": "assets/lens.ocanvas"}},
      "commercial": {{"vendor": "openUC2", "sku": "UC2-LENS-01", "price_eur": 24.0}},
      "source_path": "modules/lenses/lens-pcx-nbk7-1x1/module.yaml"
    }},
    {{
      "id": "source-collimated-3line-1x1",
      "uuid": "{SOURCE_UUID}",
      "version": "1.0.0",
      "name": "Three-line collimated source module",
      "category": "light-source",
      "group": "light-sources",
      "tags": ["source", "collimated"],
      "footprint": {{"anchor_cell": [0, 0, 0], "cells": [[0, 0, 0]]}},
      "capabilities": {{"layout": "ready", "optical": "idealized", "cad": "absent", "control": "absent"}},
      "assets": {{"canvas_fragment": "assets/source.ocanvas"}},
      "commercial": {{"vendor": "openUC2", "sku": "UC2-SRC-01", "price_eur": 40.0}}
    }},
    {{
      "id": "mirror-flat-45-1x1",
      "uuid": "{MIRROR_UUID}",
      "version": "1.0.0",
      "name": "45 degree fold mirror module",
      "category": "mirror",
      "group": "mirrors",
      "tags": ["mirror", "fold"],
      "footprint": {{"anchor_cell": [0, 0, 0], "cells": [[0, 0, 0]]}},
      "capabilities": {{"layout": "ready", "optical": "validated", "cad": "reference", "control": "absent"}},
      "assets": {{"canvas_fragment": "assets/mirror.ocanvas"}},
      "commercial": {{"vendor": "openUC2", "sku": "UC2-MIR-01", "price_eur": 18.0}}
    }},
    {{
      "id": "detector-line-1x1",
      "uuid": "{DETECTOR_UUID}",
      "version": "1.0.0",
      "name": "Camera cross-section module",
      "category": "detector",
      "group": "detectors",
      "tags": ["detector", "camera"],
      "footprint": {{"anchor_cell": [0, 0, 0], "cells": [[0, 0, 0]]}},
      "capabilities": {{"layout": "ready", "optical": "idealized", "cad": "absent", "control": "configured"}},
      "assets": {{"canvas_fragment": "assets/detector.ocanvas"}},
      "commercial": {{"vendor": "openUC2", "sku": "UC2-CAM-01", "price_eur": 120.0}}
    }},
    {{
      "id": "cube-empty-1x1",
      "uuid": "{CUBE_UUID}",
      "version": "1.0.0",
      "name": "Empty structural cube",
      "category": "mount",
      "group": "structure",
      "tags": ["cube", "structure"],
      "footprint": {{"anchor_cell": [0, 0, 0], "cells": [[0, 0, 0]]}},
      "capabilities": {{"layout": "ready", "optical": "absent", "cad": "manufacturing", "control": "absent"}},
      "assets": {{"icon": "assets/cube.svg"}},
      "commercial": {{"vendor": "openUC2", "sku": "UC2-CUBE-01", "price_eur": 6.0}}
    }}
  ]
}}"##
    )
}

/// A two-cell module, for rotation and collision tests.
pub fn two_cell_module_json(uuid: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "rail-2x1",
        "uuid": uuid,
        "version": "1.0.0",
        "name": "Two-cell rail",
        "category": "mount",
        "footprint": {"anchor_cell": [0, 0, 0], "cells": [[0, 0, 0], [1, 0, 0]]},
        "capabilities": {"layout": "ready", "optical": "absent", "cad": "absent", "control": "absent"},
        "assets": {},
        "commercial": {"vendor": "openUC2", "sku": "UC2-RAIL-2", "price_eur": 9.0}
    })
}
