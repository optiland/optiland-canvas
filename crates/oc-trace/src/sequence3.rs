//! Sequence extraction: trace a source's **axial ray** through the scene and
//! read off the ordered surface chain that reaches a detector.
//!
//! This is the inversion that lets a sequential consumer (an Optiland emitter,
//! a paraxial ladder, a diagnostic view) stop *predicting* where the beam goes
//! and instead *ask*: the non-sequential reference tracer walks the actual
//! scene, the event log records what happened, and the prescription is the one
//! branch lineage that ends on the requested detector. Where a static
//! beam-path compiler needs alignment tolerances, fold idealizations, and
//! unsupported-case errors, the extractor needs none of them — a folded path,
//! a curved mirror in it, a splitter arm are all just what the ray did.
//!
//! Failure is physical and named, in the [`crate::result3::CapabilityReport`]
//! spirit: the axial ray either reached the detector (a prescription, with
//! anything it cannot express in `notes`), or it did not
//! ([`SequenceError::NotReached`] carries every branch's actual fate), or it
//! reached it more than one way ([`SequenceError::Ambiguous`] names the
//! chains, and the caller disambiguates by declaring a tighter intent).
//!
//! Extraction traces **one** launch ray with ghosts off: Fresnel
//! back-reflections are not part of a designed sequence, while deterministic
//! splitters ([`SurfaceResponse::ExplicitRt`](oc_core::scene3::SurfaceResponse))
//! follow both arms and are disambiguated by which arm reaches the detector.

use std::collections::HashMap;

use oc_core::id::{DetectorId, PathId, PathIntentId, SourceId};
use oc_core::scene3::{Scene3, Source3};
use oc_core::seq3::{
    SeqAction, SeqHit3, SeqHitKind, SeqLaunch3, SeqTerminal3, SequencePrescription3, SEQ3_FORMAT,
    SEQ3_VERSION,
};
use oc_core::vec3::{Ray3, Vec3};

use crate::result3::{ChildRole, LaunchRay3, TraceEvent3, TraceEventKind};
use crate::tracer3::{trace3, TraceConfig3};

/// Why a sequence could not be extracted. Every variant is a statement about
/// the scene, not about the extractor.
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceError {
    SourceNotFound(SourceId),
    DetectorNotFound(DetectorId),
    IntentNotFound(PathIntentId),
    /// The source cannot produce an axial ray (empty spectrum, degenerate
    /// frame).
    DegenerateSource { source: SourceId, reason: String },
    /// No branch of the axial ray reached the detector. `fates` lists where
    /// every branch actually ended, e.g. `"lens → fold_mirror → escaped"`.
    NotReached {
        source: SourceId,
        detector: DetectorId,
        fates: Vec<String>,
    },
    /// More than one distinct branch lineage reached the detector (splitter
    /// arms recombining). `chains` summarizes each; the caller picks by
    /// declaring a tighter intent or a different detector.
    Ambiguous {
        source: SourceId,
        detector: DetectorId,
        chains: Vec<String>,
    },
}

impl std::fmt::Display for SequenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceNotFound(s) => write!(f, "source {s} not in scene"),
            Self::DetectorNotFound(d) => write!(f, "detector {d} not in scene"),
            Self::IntentNotFound(i) => write!(f, "path intent {i} not in scene"),
            Self::DegenerateSource { source, reason } => {
                write!(f, "source {source} cannot launch an axial ray: {reason}")
            }
            Self::NotReached {
                source,
                detector,
                fates,
            } => {
                write!(
                    f,
                    "axial ray of {source} never reached {detector}; branch fates: [{}]",
                    fates.join("; ")
                )
            }
            Self::Ambiguous {
                source,
                detector,
                chains,
            } => {
                write!(
                    f,
                    "axial ray of {source} reached {detector} along {} distinct chains: [{}]",
                    chains.len(),
                    chains.join("; ")
                )
            }
        }
    }
}

impl std::error::Error for SequenceError {}

/// The axial launch ray of a source: its frame origin, aimed along its local
/// `+X` optical axis, at its highest-weight spectral line, with unit flux.
pub fn axial_launch(source: &Source3) -> Result<LaunchRay3, SequenceError> {
    let samples = source.spectrum.samples();
    let (lambda, _) = samples
        .iter()
        .copied()
        // Highest weight wins; ties resolve to the earliest authored line, so
        // the choice is deterministic.
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| SequenceError::DegenerateSource {
            source: source.id,
            reason: "empty spectrum".into(),
        })?;
    let origin = source.transform.transform_point(Vec3::ZERO);
    let dir = source.transform.rotate(Vec3::X);
    let ray = Ray3::new(origin, dir).map_err(|e| SequenceError::DegenerateSource {
        source: source.id,
        reason: format!("{e:?}"),
    })?;
    Ok(LaunchRay3 {
        ray,
        lambda_um: lambda,
        flux: 1.0,
        source: Some(source.id),
    })
}

/// The extraction trace configuration: primary path only, generous budgets.
///
/// `flux_eps` is far below the default because a legitimate chain through k
/// lossy splitters carries `~0.5^k` of the launch flux, and culling it would
/// turn a real sequence into a false [`SequenceError::NotReached`].
fn extraction_config(revision: u64) -> TraceConfig3 {
    TraceConfig3 {
        ghosts: false,
        record_events: true,
        record_detector_hits: false,
        max_depth: 64,
        max_branches: 4096,
        flux_eps: 1e-9,
        revision,
        ..TraceConfig3::default()
    }
}

/// Extract the sequence a named [`PrimaryPathIntent`](oc_core::scene3::PrimaryPathIntent)
/// describes.
pub fn extract_for_intent(
    scene: &Scene3,
    intent: PathIntentId,
    revision: u64,
) -> Result<SequencePrescription3, SequenceError> {
    let pi = scene
        .intent
        .primary_paths
        .iter()
        .find(|p| p.id == intent)
        .ok_or(SequenceError::IntentNotFound(intent))?;
    let mut p = extract_sequence3(scene, pi.source, pi.detector, revision)?;
    p.intent = Some(intent);
    Ok(p)
}

/// Trace `source`'s axial ray and extract the ordered surface chain that ends
/// on `detector`.
pub fn extract_sequence3(
    scene: &Scene3,
    source: SourceId,
    detector: DetectorId,
    revision: u64,
) -> Result<SequencePrescription3, SequenceError> {
    let src = scene
        .sources
        .iter()
        .find(|s| s.id == source)
        .ok_or(SequenceError::SourceNotFound(source))?;
    if !scene.detectors.iter().any(|d| d.id() == detector) {
        return Err(SequenceError::DetectorNotFound(detector));
    }

    let launch = axial_launch(src)?;
    let cfg = extraction_config(revision);
    let result = trace3(scene, &[launch], &cfg);
    let events = &result.events;

    // Where each path id was born: the interaction event that spawned it, and
    // the role it took there.
    let mut born_at: HashMap<PathId, (usize, ChildRole)> = HashMap::new();
    for (i, e) in events.iter().enumerate() {
        if let TraceEventKind::Interact { children, .. } = &e.kind {
            for (role, id) in children {
                born_at.insert(*id, (i, *role));
            }
        }
    }

    // Candidate terminals: the first hit of the target detector per path id.
    let mut first_hit_by_path: HashMap<PathId, usize> = HashMap::new();
    for (i, e) in events.iter().enumerate() {
        if let TraceEventKind::DetectorHit { detector: d, .. } = &e.kind {
            if *d == detector {
                first_hit_by_path.entry(e.path_id).or_insert(i);
            }
        }
    }

    // The ancestor line of a path id, root first.
    let lineage = |path: PathId| -> Vec<PathId> {
        let mut line = vec![path];
        let mut cur = path;
        while let Some((idx, _)) = born_at.get(&cur) {
            cur = events[*idx].path_id;
            line.push(cur);
        }
        line.reverse();
        line
    };

    // A probe target can be re-hit further down its own lineage; those later
    // hits are the same chain arriving again, not a second chain. Keep only
    // candidates with no earlier candidate on their own ancestor line.
    let mut candidates: Vec<(PathId, usize)> = first_hit_by_path.into_iter().collect();
    candidates.sort_by_key(|(_, i)| *i);
    let survivors: Vec<(PathId, usize)> = candidates
        .iter()
        .filter(|(path, i)| {
            let line = lineage(*path);
            !candidates
                .iter()
                .any(|(p2, i2)| i2 < i && *p2 != *path && line.contains(p2))
        })
        .copied()
        .collect();

    let chain_summary = |terminal_idx: usize| -> String {
        let line = lineage(events[terminal_idx].path_id);
        let mut parts: Vec<String> = Vec::new();
        for e in events[..=terminal_idx].iter() {
            if !line.contains(&e.path_id) {
                continue;
            }
            match &e.kind {
                TraceEventKind::Interact { surface, .. } => {
                    parts.push(object_label(scene, surface.object.raw()));
                }
                TraceEventKind::DetectorHit { detector: d, .. } => {
                    parts.push(format!("detector {}", d.raw()));
                }
                TraceEventKind::ApertureBlocked { aperture } => {
                    parts.push(format!("blocked by aperture {}", aperture.raw()));
                }
                TraceEventKind::Escaped => parts.push("escaped".into()),
            }
        }
        parts.join(" → ")
    };

    match survivors.len() {
        0 => {
            // Every branch fate, so the error is a diagnosis and not a shrug:
            // walk each terminal event (no children, or a non-interaction end).
            let mut fates: Vec<String> = Vec::new();
            for (i, e) in events.iter().enumerate() {
                let terminal = match &e.kind {
                    TraceEventKind::Interact { children, .. } => children.is_empty(),
                    TraceEventKind::DetectorHit { terminal, .. } => *terminal,
                    TraceEventKind::ApertureBlocked { .. } | TraceEventKind::Escaped => true,
                };
                if terminal {
                    fates.push(chain_summary(i));
                }
                if fates.len() >= 12 {
                    fates.push("…".into());
                    break;
                }
            }
            if fates.is_empty() {
                fates.push("no events: the axial ray left the scene untouched".into());
            }
            Err(SequenceError::NotReached {
                source,
                detector,
                fates,
            })
        }
        1 => {
            let (path, terminal_idx) = survivors[0];
            Ok(build_prescription(
                scene,
                &launch,
                source,
                detector,
                revision,
                events,
                &lineage(path),
                terminal_idx,
            ))
        }
        _ => Err(SequenceError::Ambiguous {
            source,
            detector,
            chains: survivors.iter().map(|(_, i)| chain_summary(*i)).collect(),
        }),
    }
}

/// The label of a scene object by raw id, or a positional fallback.
fn object_label(scene: &Scene3, raw: u64) -> String {
    let named = scene
        .bodies
        .iter()
        .map(|b| (&b.common, "body"))
        .chain(scene.surfaces.iter().map(|s| (&s.common, "surface")))
        .find(|(c, _)| c.id.raw() == raw);
    match named {
        Some((c, _)) if !c.label.is_empty() => c.label.clone(),
        Some((_, kind)) => format!("{kind} {raw}"),
        None => format!("object {raw}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_prescription(
    scene: &Scene3,
    launch: &LaunchRay3,
    source: SourceId,
    detector: DetectorId,
    revision: u64,
    events: &[TraceEvent3],
    line: &[PathId],
    terminal_idx: usize,
) -> SequencePrescription3 {
    let mut notes = vec![
        "extracted from the axial ray with ghosts off: Fresnel back-reflections are not part of \
         the sequence"
            .to_string(),
    ];
    let spectral_lines = scene
        .sources
        .iter()
        .find(|s| s.id == source)
        .map(|s| s.spectrum.samples().len())
        .unwrap_or(1);
    if spectral_lines > 1 {
        notes.push(format!(
            "source has {spectral_lines} spectral lines; extracted at its highest-weight line \
             λ = {} µm",
            launch.lambda_um
        ));
    }

    let mut hits: Vec<SeqHit3> = Vec::new();
    let mut prev_point = launch.ray.origin;
    let mut cum_arc = 0.0;

    for (i, e) in events.iter().enumerate() {
        if i > terminal_idx || !line.contains(&e.path_id) {
            continue;
        }
        match &e.kind {
            TraceEventKind::Interact {
                surface,
                body,
                normal,
                n1,
                n2,
                children,
            } => {
                // The chain continues via the one child on the ancestor line
                // (for the last interaction: the terminal's own path id).
                let followed = children.iter().find(|(_, id)| line.contains(id));
                let Some((role, _)) = followed else {
                    // Not on the followed line after all (an interaction event
                    // of an ancestor path whose followed child is elsewhere
                    // cannot happen — every ancestor's followed child is on
                    // the line by construction). Defensive skip.
                    continue;
                };
                let action = match role {
                    ChildRole::Transmit => SeqAction::Transmit,
                    ChildRole::Reflect => SeqAction::Reflect,
                    ChildRole::Tir => SeqAction::Tir,
                    ChildRole::ThinLens => SeqAction::ThinLens,
                };
                let split = children.len() > 1;
                let arc = (e.point - prev_point).length();
                cum_arc += arc;
                prev_point = e.point;
                let label = object_label(scene, surface.object.raw());
                if split {
                    let arms: Vec<String> = children
                        .iter()
                        .map(|(r, _)| format!("{r:?}").to_lowercase())
                        .collect();
                    notes.push(format!(
                        "split at '{label}' ({}/{}): followed {:?}, arms spawned: [{}]",
                        surface.object,
                        surface.patch,
                        action,
                        arms.join(", ")
                    ));
                }
                hits.push(SeqHit3 {
                    object: surface.object,
                    patch: surface.patch,
                    kind: if *body {
                        SeqHitKind::Body
                    } else {
                        SeqHitKind::Surface
                    },
                    label,
                    point: e.point,
                    normal: *normal,
                    action,
                    split,
                    n_before: *n1,
                    n_after: *n2,
                    arc_mm: arc,
                    cum_arc_mm: cum_arc,
                    opl_mm: e.opl_mm,
                });
            }
            TraceEventKind::DetectorHit {
                detector: d, uv, ..
            } => {
                if i == terminal_idx {
                    let arc = (e.point - prev_point).length();
                    cum_arc += arc;
                    return SequencePrescription3 {
                        format: SEQ3_FORMAT.into(),
                        version: SEQ3_VERSION,
                        revision,
                        intent: None,
                        source,
                        detector,
                        launch: SeqLaunch3 {
                            point: launch.ray.origin,
                            dir: launch.ray.dir,
                            lambda_um: launch.lambda_um,
                        },
                        hits,
                        terminal: SeqTerminal3 {
                            detector: *d,
                            uv_mm: *uv,
                            point: e.point,
                            arc_mm: arc,
                            cum_arc_mm: cum_arc,
                            opl_mm: e.opl_mm,
                        },
                        notes,
                    };
                }
                // A probe passage on the way: reported, never a hit.
                notes.push(format!(
                    "passed probe detector {} at cumulative arc {:.3} mm",
                    d.raw(),
                    cum_arc + (e.point - prev_point).length()
                ));
            }
            // Blocked/escaped events cannot lie on the followed line before the
            // terminal: the line ends at the terminal by construction.
            TraceEventKind::ApertureBlocked { .. } | TraceEventKind::Escaped => {}
        }
    }
    unreachable!("the terminal event is on the followed line and returns above")
}

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::build3::SingletGeom;
    use oc_core::scene::Spectrum;
    use oc_core::scene3::{Scene3, SpectralScalar, SurfaceResponse};
    use oc_core::seq3::{SeqAction, SeqHitKind, SequencePrescription3};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// A monochromatic collimated beam aimed along +X.
    fn beam(s: &mut Scene3, pos: Vec3) -> SourceId {
        s.add_collimated_beam(pos, 5.0, 0.25, 3, Spectrum::Monochromatic(0.55))
    }

    fn mirror(r: f64) -> SurfaceResponse {
        SurfaceResponse::Mirror {
            reflectivity: SpectralScalar::Constant(r),
        }
    }

    fn splitter() -> SurfaceResponse {
        SurfaceResponse::ExplicitRt {
            reflectance: SpectralScalar::Constant(0.5),
            transmittance: SpectralScalar::Constant(0.5),
        }
    }

    #[test]
    fn straight_chain_through_a_thin_lens() {
        let mut s = Scene3::new();
        let src = beam(&mut s, Vec3::new(-50.0, 0.0, 0.0));
        let lens = SurfaceResponse::ThinLens {
            focal_length_mm: 50.0,
        };
        s.add_planar_element(Vec3::ZERO, 10.0, 0.0, lens.clone(), lens);
        let det = s.add_area_detector(Vec3::new(50.0, 0.0, 0.0), -Vec3::X, 10.0, 10.0, [8, 8]);

        let p = extract_sequence3(&s, src, det, 7).unwrap();
        assert_eq!(p.revision, 7);
        assert_eq!(p.source, src);
        assert_eq!(p.detector, det);
        assert!(close(p.launch.lambda_um, 0.55));
        assert_eq!(p.hits.len(), 1);
        let h = &p.hits[0];
        assert_eq!(h.action, SeqAction::ThinLens);
        assert_eq!(h.kind, SeqHitKind::Surface);
        assert!(!h.split);
        assert!(close(h.n_before, 1.0) && close(h.n_after, 1.0));
        assert!(close(h.arc_mm, 50.0) && close(h.cum_arc_mm, 50.0));
        assert!(close(p.terminal.arc_mm, 50.0) && close(p.terminal.cum_arc_mm, 100.0));
        // The axial ray goes through the lens centre undeviated: centred hit.
        assert!(close(p.terminal.uv_mm.x, 0.0) && close(p.terminal.uv_mm.y, 0.0));

        // The wire form round-trips exactly.
        let back = SequencePrescription3::from_json(&p.to_json()).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn folded_chain_off_a_mirror() {
        let mut s = Scene3::new();
        let src = beam(&mut s, Vec3::new(-40.0, 0.0, 0.0));
        // Normal at 135°: (−1, 1, 0)/√2 folds +X into +Y.
        s.add_planar_element(Vec3::ZERO, 12.0, 135.0, mirror(1.0), mirror(1.0));
        let det = s.add_area_detector(Vec3::new(0.0, 60.0, 0.0), -Vec3::Y, 10.0, 10.0, [4, 4]);

        let p = extract_sequence3(&s, src, det, 0).unwrap();
        assert_eq!(p.hits.len(), 1);
        let h = &p.hits[0];
        assert_eq!(h.action, SeqAction::Reflect);
        assert!(h.point.length() < 1e-9, "fold at the origin, got {}", h.point);
        // Incident-opposing normal: runs against the +X ray.
        assert!(h.normal.dot(Vec3::X) < 0.0);
        assert!(close(h.arc_mm, 40.0));
        assert!(close(p.terminal.cum_arc_mm, 100.0));
        assert!((p.terminal.point - Vec3::new(0.0, 60.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn splitter_arms_resolve_by_detector() {
        let mut s = Scene3::new();
        let src = beam(&mut s, Vec3::new(-40.0, 0.0, 0.0));
        s.add_planar_element(Vec3::ZERO, 12.0, 135.0, splitter(), splitter());
        let det_r = s.add_area_detector(Vec3::new(0.0, 60.0, 0.0), -Vec3::Y, 10.0, 10.0, [4, 4]);
        let det_t = s.add_area_detector(Vec3::new(60.0, 0.0, 0.0), -Vec3::X, 10.0, 10.0, [4, 4]);

        let pr = extract_sequence3(&s, src, det_r, 0).unwrap();
        assert_eq!(pr.hits.len(), 1);
        assert_eq!(pr.hits[0].action, SeqAction::Reflect);
        assert!(pr.hits[0].split);
        assert!(pr.notes.iter().any(|n| n.contains("split at")));

        let pt = extract_sequence3(&s, src, det_t, 0).unwrap();
        assert_eq!(pt.hits.len(), 1);
        assert_eq!(pt.hits[0].action, SeqAction::Transmit);
        assert!(pt.hits[0].split);
    }

    #[test]
    fn unreached_detector_reports_branch_fates() {
        let mut s = Scene3::new();
        let src = beam(&mut s, Vec3::new(-40.0, 0.0, 0.0));
        s.add_planar_element(Vec3::ZERO, 12.0, 135.0, mirror(1.0), mirror(1.0));
        // The fold sends the beam to +Y; this detector sits where it never goes.
        let det = s.add_area_detector(Vec3::new(0.0, -60.0, 0.0), Vec3::Y, 10.0, 10.0, [4, 4]);

        let err = extract_sequence3(&s, src, det, 0).unwrap_err();
        match err {
            SequenceError::NotReached { fates, .. } => {
                assert!(!fates.is_empty());
                assert!(
                    fates.iter().any(|f| f.contains("escaped")),
                    "expected an escaped fate, got {fates:?}"
                );
            }
            other => panic!("expected NotReached, got {other:?}"),
        }
    }

    #[test]
    fn recombining_arms_are_ambiguous() {
        let mut s = Scene3::new();
        let src = beam(&mut s, Vec3::new(-40.0, 0.0, 0.0));
        // Splitter at the origin: reflect → +Y, transmit → +X.
        s.add_planar_element(Vec3::ZERO, 12.0, 135.0, splitter(), splitter());
        // Fold each arm toward the shared detector: (0,40) turns +Y into +X,
        // (40,0) turns +X into +Y.
        s.add_planar_element(Vec3::new(0.0, 40.0, 0.0), 12.0, -45.0, mirror(1.0), mirror(1.0));
        s.add_planar_element(Vec3::new(40.0, 0.0, 0.0), 12.0, 135.0, mirror(1.0), mirror(1.0));
        // A diagonal detector both arms strike.
        let n = Vec3::new(-1.0, -1.0, 0.0).normalize();
        let det = s.add_area_detector(Vec3::new(45.0, 45.0, 0.0), n, 20.0, 20.0, [4, 4]);

        let err = extract_sequence3(&s, src, det, 0).unwrap_err();
        match err {
            SequenceError::Ambiguous { chains, .. } => assert_eq!(chains.len(), 2),
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn glass_body_records_media_and_patches() {
        let mut s = Scene3::new();
        s.add_singlet(Vec3::ZERO, SingletGeom::biconvex(), 1.5168, 64.17);
        let src = beam(&mut s, Vec3::new(-70.0, 0.0, 0.0));
        let det = s.add_area_detector(Vec3::new(48.0, 0.0, 0.0), -Vec3::X, 12.0, 12.0, [8, 8]);

        let p = extract_sequence3(&s, src, det, 0).unwrap();
        assert_eq!(p.hits.len(), 2, "front and back of the singlet");
        let (front, back) = (&p.hits[0], &p.hits[1]);
        assert_eq!(front.kind, SeqHitKind::Body);
        assert_eq!(front.action, SeqAction::Transmit);
        assert!(!front.split, "ghosts are off during extraction");
        assert_ne!(front.patch, back.patch);
        assert_eq!(front.object, back.object);
        // Air → glass → air, at the d-line of an AbbePoint(nd=1.5168) glass.
        assert!(close(front.n_before, 1.0));
        assert!((front.n_after - 1.5168).abs() < 5e-3);
        assert!((back.n_before - 1.5168).abs() < 5e-3);
        assert!(close(back.n_after, 1.0));
        // OPL through the glass exceeds geometric arc.
        assert!(p.terminal.opl_mm > p.terminal.cum_arc_mm);
    }

    #[test]
    fn intent_extraction_resolves_and_stamps_the_intent() {
        let mut s = Scene3::new();
        let src = beam(&mut s, Vec3::new(-40.0, 0.0, 0.0));
        s.add_planar_element(Vec3::ZERO, 12.0, 135.0, mirror(1.0), mirror(1.0));
        let det = s.add_area_detector(Vec3::new(0.0, 60.0, 0.0), -Vec3::Y, 10.0, 10.0, [4, 4]);
        let intent = s.alloc_path_intent_id();
        s.intent.primary_paths.push(oc_core::scene3::PrimaryPathIntent {
            id: intent,
            source: src,
            detector: det,
            stop: None,
            label: "main".into(),
        });

        let p = extract_for_intent(&s, intent, 3).unwrap();
        assert_eq!(p.intent, Some(intent));
        assert_eq!(p.detector, det);

        let missing = extract_for_intent(&s, PathIntentId(9999), 3).unwrap_err();
        assert_eq!(missing, SequenceError::IntentNotFound(PathIntentId(9999)));

        // Unknown source/detector ids are typed errors too.
        assert_eq!(
            extract_sequence3(&s, SourceId(9999), det, 0).unwrap_err(),
            SequenceError::SourceNotFound(SourceId(9999))
        );
        assert_eq!(
            extract_sequence3(&s, src, DetectorId(9999), 0).unwrap_err(),
            SequenceError::DetectorNotFound(DetectorId(9999))
        );
    }
}
