//! The derived port graph (§14.6).
//!
//! Ports are **advisory**. Nothing here feeds the tracer: Canvas traces free
//! space, and two modules whose ports disagree still exchange light exactly as
//! the geometry says they do. What the graph is for is telling a user that the
//! layout they built is not the layout they meant — a cube one cell too far, or
//! mounted with its window facing away.
//!
//! The graph is derived on demand and never stored. §14.6 is explicit that
//! connections are not authoritative state, and a stored edge would be one more
//! thing to invalidate on every move.
//!
//! # What counts as connected
//!
//! Two optical ports connect when, after placement, their positions coincide,
//! their heights agree, and their outward normals oppose (§9.4). Adjacent openUC2
//! cubes satisfy this by construction: a port on a `+X` face at `x = +25` and its
//! neighbour's `-X` port at `x = -25` land on the same 50 mm cell boundary.
//!
//! # Why one layer at a time
//!
//! Only ports on the same layer are paired. Canvas traces one horizontal plane
//! (§5.6) and optical paths between layers are an explicit non-goal (§4), so a
//! cross-layer "connection" would describe light that this engine never carries.
//! A module stranded off the active layer is already reported as such, and
//! saying it a second time in port language would not tell anyone anything new.
//!
//! # What this deliberately does not judge
//!
//! **Roles.** An `output` meeting an `output` is reported as connected. `role` is
//! nominal: Canvas light is reversible, and a singlet run backwards is a singlet.
//! Refusing such a pair would flag every legitimately reversed optic, and the
//! geometry — which is what actually decides where light goes — really does mate.
//! The role is carried through for the UI to label, not to veto.
//!
//! **Fidelity.** Port status is not a [`ProjectWarning`](crate::ProjectWarning).
//! Those report what the *optics* cannot do — a layout-only module, an idealized
//! fragment — and travel with the flattened export (§12.4), which is an optical
//! artifact. A port that has nothing across from it says nothing about fidelity;
//! it is a fact about the layout, it is normal for the open end of every chain,
//! and mixing it into the export report would bury the fidelity caveats that
//! report exists to carry. The status rides on the port, where the UI draws it.

use crate::catalog::Port;
use crate::coordinates::{uc2_to_canvas, GridProfile, Quadrant, Uc2Point};

/// How close two ports must be to count as mated, in millimetres.
///
/// Configurable per §14.6; these are its stated defaults. They are deliberately
/// loose compared to the grid: a port is either on a cell boundary or it is
/// nowhere near one, so the tolerance never has to discriminate between two
/// plausible partners.
pub const POSITION_TOL_MM: f64 = 0.5;
/// How close two ports' heights must be to count as mated (§14.6).
pub const HEIGHT_TOL_MM: f64 = 0.5;
/// How anti-parallel two outward normals must be: `dot <= -0.999` (§14.6).
pub const DIRECTION_DOT_MAX: f64 = -0.999;

/// How near two ports must be before a mismatch is worth mentioning.
///
/// Half a cell. Beyond this the ports are not a failed connection, they are
/// simply two ports in different places, and warning about every such pair would
/// make the real near-misses impossible to see.
pub const NEAR_MM: f64 = 25.0;

/// A catalogue port placed into the scene's frame.
///
/// Positions are Canvas millimetres, which is where the footprint overlay and the
/// optical geometry already are, so drawing one needs no further mapping.
/// `height_mm` is `Z_uc2` and has no Canvas counterpart — the 2D trace does not
/// model it — but it decides whether two ports meet, so it is carried explicitly
/// rather than dropped (§6.4).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PlacedPort {
    pub instance: String,
    pub module_id: String,
    pub port_id: String,
    /// The descriptor's `type`: `optical`, `mechanical`, … Uninterpreted.
    pub kind: String,
    /// `input`, `output`, or `bidirectional`.
    pub role: String,
    /// Canvas position, millimetres.
    pub position_mm: [f64; 2],
    /// Height above the mounting plane (`Z_uc2`), millimetres.
    pub height_mm: f64,
    /// Outward connector normal in Canvas, unit length.
    pub direction: [f64; 2],
    pub layer: i32,
    pub status: PortStatus,
}

/// What a port found across from it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum PortStatus {
    /// Mated with an aligned partner.
    Connected { instance: String, port: String },
    /// A partner is near enough that this looks intended, but it does not mate.
    /// The most common real mistake, and the reason the graph exists.
    Misaligned {
        instance: String,
        port: String,
        why: Misalignment,
    },
    /// Nothing near. Normal for a beam crossing free space, and normal for the
    /// open end of any chain.
    Free,
}

/// Why a nearby pair does not mate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Misalignment {
    /// Close, but not on the same point: usually a module one cell out.
    Offset,
    /// Same point, different beam height.
    Height,
    /// In the same place, but facing the same way rather than at each other:
    /// usually a descriptor whose normal points into its own cube instead of out
    /// of it (§9.4).
    Direction,
    /// Aligned in every geometric sense, but the two declare no vocabulary in
    /// common (§9.4). Advisory only.
    Compatibility,
}

impl PortStatus {
    /// Whether this port is mated.
    pub fn is_connected(&self) -> bool {
        matches!(self, PortStatus::Connected { .. })
    }
}

/// One instance's ports, ready to be placed.
pub(crate) struct PortSource<'a> {
    pub instance: &'a str,
    pub module_id: &'a str,
    pub ports: &'a [Port],
    pub anchor: [i32; 3],
    pub rotation: Quadrant,
}

/// Place every port and pair them up.
///
/// `O(n²)` over ports, which is nothing at openUC2 scale: a large setup is tens
/// of modules with a handful of ports each. It runs on placement rather than on
/// every frame.
pub(crate) fn graph(grid: &GridProfile, sources: &[PortSource<'_>]) -> Vec<PlacedPort> {
    let mut placed: Vec<PlacedPort> = Vec::new();
    for s in sources {
        for p in s.ports {
            if let Some(pp) = place(grid, s, p) {
                placed.push(pp);
            }
        }
    }

    // Statuses are decided against the placed set, so this is a second pass over
    // an immutable snapshot of the positions rather than a fold that could see
    // half-updated neighbours.
    let fixed = placed.clone();
    for (i, port) in placed.iter_mut().enumerate() {
        port.status = classify(i, port, &fixed, sources);
    }
    placed
}

/// Put one port into the scene's frame.
///
/// Returns `None` for a port with no pose: `face` alone says which side of the
/// cube a connector is on but not where, and inventing a position from it would
/// put a port somewhere the descriptor never claimed. §9.4 keeps `face` for
/// compatibility and adds `pose` for exactly this reason; a descriptor that
/// carries only the legacy field is reported as having no placeable ports rather
/// than guessed at.
fn place(grid: &GridProfile, s: &PortSource<'_>, p: &Port) -> Option<PlacedPort> {
    let pose = p.pose.as_ref()?;
    if !pose.position_mm.iter().all(|v| v.is_finite())
        || !pose.direction.iter().all(|v| v.is_finite())
    {
        return None;
    }

    // The same rigid placement the module's optics get, so a port cannot drift
    // away from the geometry it describes.
    let t = grid.placement(s.anchor, s.rotation);
    let local = uc2_to_canvas(Uc2Point::from_mm(pose.position_mm));
    let world = t.transform_point(local);

    // The direction is a vector: rotated, never translated.
    let dir_local = uc2_to_canvas(Uc2Point::from_mm(pose.direction));
    let dir = t.rotate(dir_local);
    let len = (dir.x * dir.x + dir.y * dir.y).sqrt();
    // A port whose normal is vertical has no in-plane direction to draw or to
    // compare, so it cannot participate in a 2D graph.
    if len < 1e-9 {
        return None;
    }

    Some(PlacedPort {
        instance: s.instance.to_string(),
        module_id: s.module_id.to_string(),
        port_id: p.id.clone(),
        kind: p.kind.clone(),
        role: p.role.clone(),
        position_mm: [world.x, world.y],
        // Z is untouched by an in-plane placement: the grid's rotations are about
        // the vertical axis, and a module cannot be laid on its side.
        height_mm: pose.position_mm[2],
        direction: [dir.x / len, dir.y / len],
        layer: s.anchor[2],
        status: PortStatus::Free,
    })
}

/// Decide one port's status against all the others.
fn classify(
    self_idx: usize,
    port: &PlacedPort,
    all: &[PlacedPort],
    sources: &[PortSource<'_>],
) -> PortStatus {
    // Only optical ports form an optical graph. A mechanical dovetail meeting an
    // optical window is not a misalignment, it is two unrelated things.
    if !port.kind.eq_ignore_ascii_case("optical") {
        return PortStatus::Free;
    }

    let mut nearest: Option<(f64, &PlacedPort, Misalignment)> = None;
    for (i, other) in all.iter().enumerate() {
        if i == self_idx || other.instance == port.instance {
            continue;
        }
        if !other.kind.eq_ignore_ascii_case("optical") || other.layer != port.layer {
            continue;
        }

        let d = distance(port.position_mm, other.position_mm);
        let dh = (port.height_mm - other.height_mm).abs();
        let dot = port.direction[0] * other.direction[0] + port.direction[1] * other.direction[1];

        if d <= POSITION_TOL_MM && dh <= HEIGHT_TOL_MM && dot <= DIRECTION_DOT_MAX {
            if !compatible(port, other, sources) {
                return PortStatus::Misaligned {
                    instance: other.instance.clone(),
                    port: other.port_id.clone(),
                    why: Misalignment::Compatibility,
                };
            }
            return PortStatus::Connected {
                instance: other.instance.clone(),
                port: other.port_id.clone(),
            };
        }

        if d > NEAR_MM {
            continue;
        }
        // Report the most specific thing wrong: a port in the right place facing
        // the wrong way is a different mistake from one in the wrong place, and
        // "move it" and "turn it round" are different fixes.
        let why = if d > POSITION_TOL_MM {
            Misalignment::Offset
        } else if dh > HEIGHT_TOL_MM {
            Misalignment::Height
        } else {
            Misalignment::Direction
        };
        let closer = match nearest {
            Some((best, _, _)) => d < best,
            None => true,
        };
        if closer {
            nearest = Some((d, other, why));
        }
    }

    match nearest {
        Some((_, other, why)) => PortStatus::Misaligned {
            instance: other.instance.clone(),
            port: other.port_id.clone(),
            why,
        },
        None => PortStatus::Free,
    }
}

/// Whether two mated ports declare a vocabulary in common (§9.4).
///
/// An empty list on either side means "unspecified", which is treated as
/// compatible: most descriptors do not declare one, and refusing every such pair
/// would report the whole catalogue as incompatible with itself.
fn compatible(a: &PlacedPort, b: &PlacedPort, sources: &[PortSource<'_>]) -> bool {
    let vocab = |p: &PlacedPort| -> Vec<String> {
        sources
            .iter()
            .find(|s| s.instance == p.instance)
            .and_then(|s| s.ports.iter().find(|q| q.id == p.port_id))
            .map(|q| q.compatibility.clone())
            .unwrap_or_default()
    };
    let (va, vb) = (vocab(a), vocab(b));
    va.is_empty() || vb.is_empty() || va.iter().any(|x| vb.contains(x))
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Port, PortPose};

    fn optical(id: &str, role: &str, pos: [f64; 3], dir: [f64; 3]) -> Port {
        Port {
            id: id.to_string(),
            kind: "optical".to_string(),
            role: role.to_string(),
            face: None,
            pose: Some(PortPose {
                position_mm: pos,
                direction: dir,
            }),
            aperture_mm: None,
            compatibility: Vec::new(),
        }
    }

    /// A module with one output on its +X face and one input on its -X face: the
    /// shape of every in-line openUC2 optic.
    fn inline_ports() -> Vec<Port> {
        vec![
            optical("optical-in", "input", [-25.0, 0.0, 25.0], [-1.0, 0.0, 0.0]),
            optical("optical-out", "output", [25.0, 0.0, 25.0], [1.0, 0.0, 0.0]),
        ]
    }

    fn source<'a>(
        instance: &'a str,
        ports: &'a [Port],
        anchor: [i32; 3],
        rot: Quadrant,
    ) -> PortSource<'a> {
        PortSource {
            instance,
            module_id: "m",
            ports,
            anchor,
            rotation: rot,
        }
    }

    fn find<'a>(ports: &'a [PlacedPort], instance: &str, id: &str) -> &'a PlacedPort {
        ports
            .iter()
            .find(|p| p.instance == instance && p.port_id == id)
            .expect("port")
    }

    #[test]
    fn a_port_lands_on_its_modules_cell_boundary() {
        let grid = GridProfile::openuc2_core_50();
        let ports = inline_ports();
        let g = graph(&grid, &[source("a", &ports, [0, 0, 0], Quadrant::R0)]);

        let out = find(&g, "a", "optical-out");
        assert!((out.position_mm[0] - 25.0).abs() < 1e-12, "{out:?}");
        assert!((out.position_mm[1]).abs() < 1e-12);
        assert!((out.direction[0] - 1.0).abs() < 1e-12);
        assert_eq!(out.height_mm, 25.0);
    }

    #[test]
    fn adjacent_modules_connect() {
        // The whole point of a 50 mm grid: neighbours mate by construction.
        let grid = GridProfile::openuc2_core_50();
        let ports = inline_ports();
        let g = graph(
            &grid,
            &[
                source("a", &ports, [0, 0, 0], Quadrant::R0),
                source("b", &ports, [1, 0, 0], Quadrant::R0),
            ],
        );

        assert_eq!(
            find(&g, "a", "optical-out").status,
            PortStatus::Connected {
                instance: "b".to_string(),
                port: "optical-in".to_string(),
            }
        );
        // And the relation is symmetric.
        assert!(find(&g, "b", "optical-in").status.is_connected());
        // The far ends of the chain have nothing to meet.
        assert_eq!(find(&g, "a", "optical-in").status, PortStatus::Free);
        assert_eq!(find(&g, "b", "optical-out").status, PortStatus::Free);
    }

    #[test]
    fn a_rotated_module_carries_its_ports_round() {
        let grid = GridProfile::openuc2_core_50();
        let ports = inline_ports();
        // 90° CCW: the +X face now looks along +Y.
        let g = graph(&grid, &[source("a", &ports, [0, 0, 0], Quadrant::R90)]);

        let out = find(&g, "a", "optical-out");
        assert!(out.position_mm[0].abs() < 1e-12, "{out:?}");
        assert!((out.position_mm[1] - 25.0).abs() < 1e-12, "{out:?}");
        assert!((out.direction[1] - 1.0).abs() < 1e-12, "{out:?}");
    }

    #[test]
    fn an_inward_normal_is_reported_as_a_direction_mismatch() {
        // §9.4 defines `pose.direction` as the *outward* normal. b's input is
        // authored pointing into its own cube, so the two ports meet in space and
        // face the same way. Nothing about the footprint shows this.
        let grid = GridProfile::openuc2_core_50();
        let a = inline_ports();
        let b = vec![optical(
            "optical-in",
            "input",
            [-25.0, 0.0, 25.0],
            [1.0, 0.0, 0.0],
        )];
        let g = graph(
            &grid,
            &[
                source("a", &a, [0, 0, 0], Quadrant::R0),
                source("b", &b, [1, 0, 0], Quadrant::R0),
            ],
        );

        assert_eq!(
            find(&g, "a", "optical-out").status,
            PortStatus::Misaligned {
                instance: "b".to_string(),
                port: "optical-in".to_string(),
                why: Misalignment::Direction,
            }
        );
    }

    #[test]
    fn a_reversed_module_still_mates() {
        // b is turned 180°, so light enters through what its descriptor calls an
        // output. The graph reports the connection: the geometry mates, and role
        // is nominal — a singlet run backwards is still a singlet. Flagging this
        // would flag every deliberately reversed optic.
        let grid = GridProfile::openuc2_core_50();
        let ports = inline_ports();
        let g = graph(
            &grid,
            &[
                source("a", &ports, [0, 0, 0], Quadrant::R0),
                source("b", &ports, [1, 0, 0], Quadrant::R180),
            ],
        );

        assert_eq!(
            find(&g, "a", "optical-out").status,
            PortStatus::Connected {
                instance: "b".to_string(),
                port: "optical-out".to_string(),
            }
        );
    }

    #[test]
    fn a_module_one_cell_too_far_is_an_offset_not_a_connection() {
        let grid = GridProfile::openuc2_core_50();
        let ports = inline_ports();
        let g = graph(
            &grid,
            &[
                source("a", &ports, [0, 0, 0], Quadrant::R0),
                source("b", &ports, [2, 0, 0], Quadrant::R0),
            ],
        );
        // 50 mm apart: far beyond NEAR_MM, so this is not a near-miss at all.
        assert_eq!(find(&g, "a", "optical-out").status, PortStatus::Free);
    }

    #[test]
    fn a_beam_height_mismatch_is_named_as_one() {
        let grid = GridProfile::openuc2_core_50();
        let a = inline_ports();
        let b = vec![optical(
            "optical-in",
            "input",
            [-25.0, 0.0, 40.0],
            [-1.0, 0.0, 0.0],
        )];
        let g = graph(
            &grid,
            &[
                source("a", &a, [0, 0, 0], Quadrant::R0),
                source("b", &b, [1, 0, 0], Quadrant::R0),
            ],
        );

        assert_eq!(
            find(&g, "a", "optical-out").status,
            PortStatus::Misaligned {
                instance: "b".to_string(),
                port: "optical-in".to_string(),
                why: Misalignment::Height,
            },
            "same point in plan, 15 mm apart in height"
        );
    }

    #[test]
    fn ports_on_different_layers_do_not_pair() {
        // They would mate perfectly in plan. Canvas does not carry light between
        // layers, so calling this a connection would describe something that
        // cannot happen.
        let grid = GridProfile::openuc2_core_50();
        let ports = inline_ports();
        let g = graph(
            &grid,
            &[
                source("a", &ports, [0, 0, 0], Quadrant::R0),
                source("b", &ports, [1, 0, 1], Quadrant::R0),
            ],
        );
        assert_eq!(find(&g, "a", "optical-out").status, PortStatus::Free);
    }

    #[test]
    fn a_ports_own_module_is_not_its_partner() {
        // An in-line module's own two ports are 50 mm apart and face opposite
        // ways; nothing should let it satisfy itself.
        let grid = GridProfile::openuc2_core_50();
        let ports = inline_ports();
        let g = graph(&grid, &[source("a", &ports, [0, 0, 0], Quadrant::R0)]);
        assert!(g.iter().all(|p| p.status == PortStatus::Free));
    }

    #[test]
    fn a_non_optical_port_is_left_alone() {
        let grid = GridProfile::openuc2_core_50();
        let mut mech = inline_ports();
        for p in &mut mech {
            p.kind = "mechanical".to_string();
        }
        let g = graph(
            &grid,
            &[
                source("a", &mech, [0, 0, 0], Quadrant::R0),
                source("b", &mech, [1, 0, 0], Quadrant::R0),
            ],
        );
        assert!(
            g.iter().all(|p| p.status == PortStatus::Free),
            "mechanical ports form no optical graph"
        );
    }

    #[test]
    fn a_port_with_no_pose_is_dropped_rather_than_guessed_at() {
        let grid = GridProfile::openuc2_core_50();
        let legacy = vec![Port {
            id: "optical-in".to_string(),
            kind: "optical".to_string(),
            role: "input".to_string(),
            face: Some("-X".to_string()),
            pose: None,
            aperture_mm: None,
            compatibility: Vec::new(),
        }];
        let g = graph(&grid, &[source("a", &legacy, [0, 0, 0], Quadrant::R0)]);
        assert!(g.is_empty(), "a face is not a position");
    }

    #[test]
    fn mated_ports_with_no_vocabulary_in_common_are_flagged() {
        let grid = GridProfile::openuc2_core_50();
        let mut a = inline_ports();
        a[1].compatibility = vec!["openuc2-optical-free-space-v1".to_string()];
        let mut b = inline_ports();
        b[0].compatibility = vec!["openuc2-fibre-v1".to_string()];

        let g = graph(
            &grid,
            &[
                source("a", &a, [0, 0, 0], Quadrant::R0),
                source("b", &b, [1, 0, 0], Quadrant::R0),
            ],
        );
        assert_eq!(
            find(&g, "a", "optical-out").status,
            PortStatus::Misaligned {
                instance: "b".to_string(),
                port: "optical-in".to_string(),
                why: Misalignment::Compatibility,
            }
        );
    }

    #[test]
    fn an_undeclared_vocabulary_does_not_block_a_connection() {
        let grid = GridProfile::openuc2_core_50();
        let mut a = inline_ports();
        a[1].compatibility = vec!["openuc2-optical-free-space-v1".to_string()];
        let b = inline_ports(); // declares nothing

        let g = graph(
            &grid,
            &[
                source("a", &a, [0, 0, 0], Quadrant::R0),
                source("b", &b, [1, 0, 0], Quadrant::R0),
            ],
        );
        assert!(find(&g, "a", "optical-out").status.is_connected());
    }
}
