//! Records the C33 corpus's fingerprints as the kernel computes them now.
//!
//! Run it at the commit before the fidelity profile existed (212c747), so the
//! file holds what the pre-profile kernel computed:
//!
//! ```text
//! git worktree add ../oc-pre-profile 212c747
//! cp crates/oc-trace/tests/{record_corpus3.rs,corpus3} ../oc-pre-profile/crates/oc-trace/tests/ -r
//! cd ../oc-pre-profile
//! CORPUS3_OUT=<path>/pre_profile_corpus3.json \
//!   cargo test -p oc-trace --features trace3 --test record_corpus3 -- --ignored
//! ```
//!
//! and copy the result to `tests/data/pre_profile_corpus3.json`. Since CF-08
//! the corpus names the legacy broadband AR `DesignKind::BroadbandArLegacy`;
//! at 212c747 it is `DesignKind::BroadbandAr`, so rename it in the copy. Running it on
//! a later commit records that commit's numbers, which is not what C33 holds
//! the kernel to.
#![cfg(feature = "trace3")]

mod common;
mod corpus3;

#[test]
#[ignore = "records a golden file; see the module docs"]
fn record() {
    let path = std::env::var("CORPUS3_OUT").expect("CORPUS3_OUT names the output file");
    let mut doc = serde_json::Map::new();
    for case in corpus3::cases() {
        let mut per = serde_json::Map::new();
        for (name, cfg) in corpus3::configs() {
            let lines = corpus3::fingerprint(&case.trace(&cfg));
            per.insert(name.to_string(), serde_json::json!(lines));
        }
        doc.insert(case.name.to_string(), serde_json::Value::Object(per));
    }
    std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
}
