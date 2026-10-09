//! Reading a `.ocanvas` document of either version, and the three ways that
//! can fail.
//!
//! The v2 fixture is a file the browser saved (`web/tests/save.spec.ts` keeps
//! it current), not one written to look like it.
#![cfg(feature = "scene3")]

use oc_core::document::{read_document, Document, DocumentError, READABLE_VERSIONS};

const BROWSER_V2: &str = include_str!("../../oc-py/tests/data/browser-default.ocanvas");
const SHIPPED_V1: &str =
    include_str!("../../../web/public/catalog/openuc2/assets/detector.ocanvas");

#[test]
fn both_versions_read_and_keep_their_version() {
    let v2 = read_document(BROWSER_V2).expect("the browser's file reads");
    assert_eq!(v2.version(), 2);
    assert!(matches!(v2, Document::V2(_)));

    let v1 = read_document(SHIPPED_V1).expect("a shipped v1 file reads");
    assert_eq!(v1.version(), 1);
    // Normalizing never migrates.
    let again = read_document(&v1.to_json()).unwrap();
    assert_eq!(again.version(), 1);
    assert_eq!(READABLE_VERSIONS, [1, 2]);
}

#[test]
fn a_browser_file_normalizes_to_itself() {
    let doc = read_document(BROWSER_V2).unwrap();
    let a: serde_json::Value = serde_json::from_str(BROWSER_V2).unwrap();
    let b: serde_json::Value = serde_json::from_str(&doc.to_json()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn a_v1_file_reaches_the_3d_scene_through_the_migration() {
    let (scene, report) = read_document(SHIPPED_V1).unwrap().into_scene3().unwrap();
    assert!(report.is_some(), "a v1 file says it was migrated");
    assert_eq!(scene.detectors.len(), 1);
    let (_, report) = read_document(BROWSER_V2).unwrap().into_scene3().unwrap();
    assert!(report.is_none(), "a v2 file was not");
}

#[test]
fn a_document_with_no_header_is_version_1() {
    // Files written before the header existed.
    let mut v: serde_json::Value = serde_json::from_str(SHIPPED_V1).unwrap();
    let obj = v.as_object_mut().unwrap();
    obj.remove("format");
    obj.remove("version");
    assert_eq!(read_document(&v.to_string()).unwrap().version(), 1);
}

fn edited(f: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(BROWSER_V2).unwrap();
    f(&mut v);
    v.to_string()
}

#[test]
fn the_three_failures_stay_apart() {
    for text in [
        "not json",
        "[1]",
        r#"{"format":"svg","version":2}"#,
        r#"{"version":"2"}"#,
    ] {
        assert!(
            matches!(read_document(text), Err(DocumentError::NotADocument(_))),
            "{text}"
        );
    }
    assert_eq!(
        read_document(&edited(|v| v["version"] = 7.into())).unwrap_err(),
        DocumentError::UnsupportedVersion { found: 7 }
    );
    // The right kind of file at a readable version, with a scene that does
    // not hold up, is neither of the other two.
    for broken in [
        edited(|v| v["detectors"][0]["bins"] = serde_json::json!([0, 64])),
        edited(|v| v["detectors"][0]["bins"] = "many".into()),
    ] {
        assert!(
            matches!(read_document(&broken), Err(DocumentError::InvalidScene(_))),
            "{:?}",
            read_document(&broken)
        );
    }
}
