//! Emit the committed `.ocanvas` v2 schema fixture.
//!
//! Generated rather than hand-written, with a test that re-derives it and
//! compares. A fixture nobody regenerates drifts away from the schema it exists
//! to pin, which is the pattern `emit_folded_singlet` already follows.
//!
//! `cargo run -p oc-core --features scene3 --example emit_v2_fixture`

fn main() {
    let v1 = oc_core::Scene::from_json(include_str!(
        "../../../web/public/catalog/openuc2/assets/mirror.ocanvas"
    ))
    .expect("v1 fixture parses");
    let (s3, _) = oc_core::migrate_v1::migrate_v1_to_v2(&v1).expect("migrates");
    print!("{}", s3.to_json());
}
