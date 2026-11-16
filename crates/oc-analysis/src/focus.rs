//! Focused sampling for `uniformity`: a pilot trace finds the part of each
//! source's sampling domain that sends light into the region, and the job
//! spends most of its rays there.
//!
//! **The domain.** A source draws its origins from a unit square and its
//! directions from another, each through a map that keeps the source's own
//! measure ([`SampleWindow`]). A rectangle `[[u0, u1], [v0, v1]]` of either
//! square holds exactly its area's share of the source's light.
//!
//! **The pilot.** For each source it can cut, rounds alternate between the
//! directions' square and the origins', two of each, directions first. A
//! round cuts the box so far into 8 × 8 windows, every window with the same
//! rays, and traces the scene with that source's 64 windows alone, one per
//! kernel block, so each block's readout is one window's light into the
//! region. It repeats, twice the rays each time, until the region has seen
//! enough hits ([`FIRST_ROUND_HITS`], [`ROUND_HITS`]); then the windows
//! holding all but a thousandth of the light, widened by one window on every
//! side, make the new box. The pilot stops where it is once its budget
//! cannot hold the next repeat. Every decision reads whole blocks, so the
//! pilot, like the rest of the job, does not depend on how it is stepped.
//! Its light only chooses the boxes: it is in no number of the result.
//!
//! **The trace.** Each cut source is traced as a partition of its domain:
//! the box `R` (origins' box × directions' box) and the rest cut into up to
//! eight rectangles. Every piece is a source with that window and its share
//! of the flux, `flux × measure`, which the kernel spreads evenly over the
//! piece's rays. `R` takes nine tenths of the rays; the rest, a tenth
//! between them by measure and at least [`MIN_PIECE_RAYS`] each. The pieces
//! cover the domain once, each sampled at its own density with the flux per
//! ray its own measure over its rays, so the estimate is the plain one's
//! expectation, unbiased, whatever the pilot found. A box that misses light
//! makes that light noisier, never lost.

use oc_core::scene3::{SampleWindow, Scene3, Source3};
use oc_trace::{TraceBlocks3, TraceConfig3, TracePart3, TRACE_BLOCKS};
use serde_json::{json, Value};

use crate::uniformity::Uniformity;

/// A rectangle of a unit square, `[[u low, u high], [v low, v high]]`.
type Rect = [[f64; 2]; 2];

const WHOLE: Rect = [[0.0, 1.0], [0.0, 1.0]];

/// The dimensions of a domain, by the names a result gives them.
const DIMS: [&str; 2] = ["spatial", "angular"];

/// Windows along each axis of a pilot round: 8 × 8 is the kernel's 64
/// blocks, one window to a block.
const GRID: usize = 8;

/// Rounds of the pilot along each cut dimension.
const ROUNDS: usize = 2;

/// The share of a round's light the windows kept hold at least.
const KEEP: f64 = 0.999;

/// The share of a cut source's rays that goes to the rest of its domain.
pub(crate) const DEFENSIVE_SHARE: f64 = 0.1;

/// The fewest rays a piece of the rest of the domain gets.
pub(crate) const MIN_PIECE_RAYS: u64 = 16;

/// Rays per window in a round's first repeat; each repeat doubles them, or
/// takes what is left of the budget.
const FIRST_WINDOW_RAYS: u64 = 64;

/// The hits into the region a round collects before it narrows a box: fewer
/// for a source's first round, which sees the least light per ray, and
/// whose windows are the coarsest.
const FIRST_ROUND_HITS: f64 = 32.0;
const ROUND_HITS: f64 = 128.0;

/// The fewest and the most rays a request may give the pilot; with none it
/// gets a quarter of the job's, within these.
pub(crate) const MIN_PILOT_RAYS: u64 = FIRST_WINDOW_RAYS * TRACE_BLOCKS as u64;
pub(crate) const MAX_PILOT_RAYS: u64 = 1 << 24;

fn area(r: &Rect) -> f64 {
    (r[0][1] - r[0][0]) * (r[1][1] - r[1][0])
}

/// A rectangle given in fractions of `outer`, in the square's own terms.
fn within(outer: &Rect, i: [usize; 2], j: [usize; 2]) -> Rect {
    let at = |k: usize, n: usize| {
        let [lo, hi] = outer[k];
        if n == GRID {
            hi
        } else {
            lo + (hi - lo) * n as f64 / GRID as f64
        }
    };
    [[at(0, i[0]), at(0, i[1])], [at(1, j[0]), at(1, j[1])]]
}

/// The rest of the unit square once `b` is taken: up to four rectangles, the
/// strips left and right of it full height, and below and above it between.
fn frame(b: &Rect) -> Vec<Rect> {
    let [[u0, u1], [v0, v1]] = *b;
    let mut out = Vec::new();
    if u0 > 0.0 {
        out.push([[0.0, u0], [0.0, 1.0]]);
    }
    if u1 < 1.0 {
        out.push([[u1, 1.0], [0.0, 1.0]]);
    }
    if v0 > 0.0 {
        out.push([[u0, u1], [0.0, v0]]);
    }
    if v1 < 1.0 {
        out.push([[u0, u1], [v1, 1.0]]);
    }
    out
}

/// Which of `[origins, directions]` focusing may cut on `src`: a dimension
/// a window cuts and the source spreads over more than one sample.
fn cut_dims(src: &Source3) -> [bool; 2] {
    let [ws, wa] = src.window_dims();
    let (s, a) = (src.samples.spatial_samples, src.samples.angular_samples);
    if src.pairs_rays() {
        // Every ray its own origin and direction: both spread over them all.
        [ws && s.max(1) * a.max(1) > 1, wa && s.max(1) * a.max(1) > 1]
    } else {
        [ws && s > 1, wa && a > 1]
    }
}

/// `src`'s sample counts for about `n` rays per line, sampled as `src` is:
/// paired if it pairs, the spread dimension scaled if not.
fn with_rays(src: &Source3, n: u64) -> (u32, u32) {
    let n = n.clamp(1, u32::MAX as u64 / 2) as u32;
    let (s, a) = (
        src.samples.spatial_samples.max(1),
        src.samples.angular_samples.max(1),
    );
    if src.pairs_rays() {
        if src.spectrum.is_continuum() && (s == 1 || a == 1) {
            // Paired by its continuum alone: keep it from spreading both.
            if s == 1 {
                (1, n)
            } else {
                (n, 1)
            }
        } else {
            (n.div_ceil(2).max(2), 2)
        }
    } else if cut_dims(src)[0] {
        ((n as f64 / a as f64).round().max(1.0) as u32, a)
    } else {
        (s, (n as f64 / s as f64).round().max(1.0) as u32)
    }
}

/// The launch rays one pass of `src` emits.
fn rays_of(src: &Source3) -> u64 {
    src.spectrum.ray_multiplicity() as u64
        * src.samples.spatial_samples.max(1) as u64
        * src.samples.angular_samples.max(1) as u64
}

/// A seed for piece `k` of `src` in stage `stage`, apart from every other.
fn seed_for(seed: u64, stage: u64, k: u64) -> u64 {
    let mut z = seed
        ^ stage.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (k + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `src` over the window `[origins, directions]`, with the flux of that part
/// of its domain and about `n` rays per line.
fn piece(src: &Source3, w: [Rect; 2], n: u64, seed: u64) -> Source3 {
    let mut out = src.clone();
    let [ws, wa] = src.window_dims();
    let some = |on: bool, r: Rect| (on && r != WHOLE).then_some(r);
    out.window = SampleWindow {
        spatial: some(ws, w[0]),
        angular: some(wa, w[1]),
    };
    out.flux = src.flux * area(&w[0]) * area(&w[1]);
    let (s, a) = with_rays(src, n);
    out.samples.spatial_samples = s;
    out.samples.angular_samples = a;
    out.samples.seed = seed;
    out
}

/// One round of the pilot: a source's dimension, cut 8 × 8.
#[derive(Debug, Clone, Copy)]
struct Stage {
    source: usize,
    dim: usize,
}

/// A round being traced: its windows traced `repeat + 1` times so far, each
/// time with twice the rays, or what is left of the budget.
struct Active {
    blocks: TraceBlocks3,
    part: TracePart3,
    repeat: u32,
    /// Launch rays per window in this repeat.
    each: u64,
    /// Each window's light into the region in this repeat, by block.
    flux: Vec<f64>,
    /// Each window's light over the repeats done, and the hits it stands
    /// for: light over the flux one ray carries.
    sum: Vec<f64>,
    hits: f64,
}

/// The pilot of a focused job, and once it is done, the scene it found.
pub(crate) struct Pilot {
    base: Scene3,
    cfg: TraceConfig3,
    stages: Vec<Stage>,
    /// The stage being traced; `stages.len()` once every one is done.
    stage: usize,
    active: Option<Active>,
    /// Each source's boxes, `[origins, directions]`, so far.
    boxes: Vec<[Rect; 2]>,
    /// Launch rays in the repeats done, and the most the pilot may trace.
    traced: u64,
    budget: u64,
    /// Whether the budget ran out before a round saw enough light.
    short: bool,
    /// What each round found: its source, dimension, rays and hits.
    rounds: Vec<Value>,
}

impl Pilot {
    /// A pilot of at most `budget` launch rays on `base`, the scene as the
    /// job traces it.
    pub(crate) fn new(base: &Scene3, cfg: TraceConfig3, budget: u64) -> Self {
        let mut stages = Vec::new();
        for (i, src) in base.sources.iter().enumerate() {
            // Directions first: the optics take in a narrow cone of a wide
            // source wherever the region is, so the first cut is often there,
            // and every round after it sees more light per ray.
            let cut = cut_dims(src);
            for _ in 0..ROUNDS {
                for dim in [1, 0] {
                    if cut[dim] {
                        stages.push(Stage { source: i, dim });
                    }
                }
            }
        }
        let mut out = Self {
            base: base.clone(),
            cfg,
            boxes: vec![[WHOLE; 2]; base.sources.len()],
            stages,
            stage: 0,
            active: None,
            traced: 0,
            budget,
            short: false,
            rounds: Vec::new(),
        };
        out.begin(0, FIRST_WINDOW_RAYS, Vec::new(), 0.0);
        out
    }

    /// Window `c` (row-major, `u` fastest) of the current stage's round, in
    /// repeat `repeat`, of about `each` launch rays.
    fn cell(&self, c: usize, repeat: u32, each: u64) -> Source3 {
        let Stage { source, dim } = self.stages[self.stage];
        let src = &self.base.sources[source];
        let b = self.boxes[source];
        let (i, j) = (c % GRID, c / GRID);
        let mut w = b;
        w[dim] = within(&b[dim], [i, i + 1], [j, j + 1]);
        let n = each / src.spectrum.ray_multiplicity() as u64;
        let stage = ((self.stage as u64) << 8) | repeat as u64;
        piece(
            src,
            w,
            n.max(1),
            seed_for(src.samples.seed, 1 + stage, c as u64),
        )
    }

    /// Begin repeat `repeat` of the current stage, of about `each` launch
    /// rays per window, carrying the sums of the repeats before, if the
    /// budget holds it; or end the pilot.
    fn begin(&mut self, repeat: u32, each: u64, sum: Vec<f64>, hits: f64) {
        if self.done() {
            return;
        }
        let left = self.budget.saturating_sub(self.traced) / TRACE_BLOCKS as u64;
        let each = each.min(left);
        let mut scene = self.base.clone();
        scene.sources = (0..TRACE_BLOCKS)
            .map(|c| self.cell(c, repeat, each))
            .collect();
        let rays = rays_of(&scene.sources[0]) * TRACE_BLOCKS as u64;
        if each < FIRST_WINDOW_RAYS || self.traced + rays > self.budget {
            // Out of rays: the boxes stay as the rounds done left them.
            self.short = true;
            self.stage = self.stages.len();
            return;
        }
        let blocks = TraceBlocks3::with_passes(scene, 1, None, self.cfg);
        // The kernel cuts the rays into 64 equal blocks: one window each,
        // unless a window dropped a ray. Then the pilot cannot read the
        // windows apart, and stops where it is.
        if blocks.blocks() != TRACE_BLOCKS || blocks.rays() as u64 != rays {
            self.stage = self.stages.len();
            return;
        }
        let part = blocks.part(0, false);
        self.active = Some(Active {
            blocks,
            part,
            repeat,
            each,
            flux: Vec::with_capacity(TRACE_BLOCKS),
            sum,
            hits,
        });
    }

    pub(crate) fn done(&self) -> bool {
        self.stage >= self.stages.len()
    }

    /// Launch rays traced so far.
    pub(crate) fn traced(&self) -> u64 {
        self.traced
            + self.active.as_ref().map_or(0, |a| {
                let (done, each) = a.part.progress();
                (a.flux.len() * each + done) as u64
            })
    }

    /// The most launch rays the pilot traces: its budget until it is done,
    /// and then the rays it traced.
    pub(crate) fn total(&self) -> u64 {
        if self.done() {
            self.traced
        } else {
            self.budget
        }
    }

    /// Trace up to `rays` more pilot rays; the rays traced.
    pub(crate) fn step(&mut self, rays: u64, reader: &Uniformity) -> u64 {
        let before = self.traced();
        let mut left = rays.max(1);
        while left > 0 {
            let Some(a) = self.active.as_mut() else { break };
            let (was, each) = a.part.progress();
            let room = (each - was) as u64;
            a.blocks.step(&mut a.part, left.min(room) as usize);
            left = left.saturating_sub((a.part.progress().0 - was) as u64);
            if !a.part.done() {
                continue;
            }
            a.flux.push(reader.region_flux(a.part.detectors()));
            if a.flux.len() < TRACE_BLOCKS {
                a.part = a.blocks.part(a.flux.len(), false);
                continue;
            }
            let a = self.active.take().expect("a round in hand");
            self.traced += a.blocks.rays() as u64;
            let mut sum = a.sum;
            sum.resize(TRACE_BLOCKS, 0.0);
            let mut hits = a.hits;
            for (c, f) in a.flux.iter().enumerate() {
                sum[c] += f;
                let cell = self.cell(c, a.repeat, a.each);
                hits += f / (cell.flux / rays_of(&cell) as f64);
            }
            let first = self.stage == 0
                || self.stages[self.stage - 1].source != self.stages[self.stage].source;
            if hits >= if first { FIRST_ROUND_HITS } else { ROUND_HITS } {
                let Stage { source, dim } = self.stages[self.stage];
                self.narrow(&sum);
                self.rounds.push(json!({
                    "source": self.base.sources[source].id.0,
                    "dimension": DIMS[dim],
                    "repeats": a.repeat + 1,
                    "hits": hits.round(),
                    "box": self.boxes[source][dim],
                }));
                self.stage += 1;
                self.begin(0, FIRST_WINDOW_RAYS, Vec::new(), 0.0);
            } else {
                self.begin(a.repeat + 1, 2 * a.each, sum, hits);
            }
        }
        self.traced() - before
    }

    /// The box of the windows holding all but a thousandth of a round's
    /// light, widened by a window each side.
    fn narrow(&mut self, flux: &[f64]) {
        let total: f64 = flux.iter().sum();
        let mut order: Vec<usize> = (0..flux.len()).collect();
        // Brightest first; ties in window order, so the pick is fixed.
        order.sort_by(|&a, &b| flux[b].total_cmp(&flux[a]).then(a.cmp(&b)));
        let (mut lo, mut hi) = ([GRID, GRID], [0, 0]);
        let mut kept = 0.0;
        for c in order {
            if kept >= KEEP * total || flux[c] <= 0.0 {
                break;
            }
            kept += flux[c];
            let ij = [c % GRID, c / GRID];
            for k in 0..2 {
                lo[k] = lo[k].min(ij[k]);
                hi[k] = hi[k].max(ij[k]);
            }
        }
        let Stage { source, dim } = self.stages[self.stage];
        let b = self.boxes[source][dim];
        let i = [lo[0].saturating_sub(1), (hi[0] + 2).min(GRID)];
        let j = [lo[1].saturating_sub(1), (hi[1] + 2).min(GRID)];
        self.boxes[source][dim] = within(&b, i, j);
    }

    /// What the pilot did, for the result's `sampling`.
    pub(crate) fn report(&self) -> Value {
        json!({
            "rays": self.traced,
            "budget": self.budget,
            "budgetSpent": self.short,
            "rounds": self.rounds,
        })
    }

    /// The scene to trace once the pilot is done: each cut source as the
    /// partition of its domain, and a report of what was cut.
    pub(crate) fn focused(&self) -> (Scene3, Value) {
        let mut scene = self.base.clone();
        scene.sources.clear();
        let mut report = Vec::new();
        for (i, src) in self.base.sources.iter().enumerate() {
            let [bs, ba] = self.boxes[i];
            let n = src.samples.spatial_samples.max(1) as u64
                * src.samples.angular_samples.max(1) as u64;
            let share = area(&bs) * area(&ba);
            let mut pieces: Vec<[Rect; 2]> = frame(&bs).into_iter().map(|c| [c, WHOLE]).collect();
            pieces.extend(frame(&ba).into_iter().map(|c| [bs, c]));
            let rest = 1.0 - share;
            let counts: Vec<u64> = pieces
                .iter()
                .map(|w| {
                    ((DEFENSIVE_SHARE * n as f64 * area(&w[0]) * area(&w[1]) / rest).round() as u64)
                        .max(MIN_PIECE_RAYS)
                })
                .collect();
            let in_box = n.saturating_sub(counts.iter().sum());
            let reason = if pieces.is_empty() {
                Some(if cut_dims(src) == [false, false] {
                    "nothing to cut"
                } else if self.short {
                    "the pilot ran out of rays before it saw enough light"
                } else {
                    "the whole domain reaches the region"
                })
            } else if in_box < n / 2 {
                Some("too few rays to cut")
            } else {
                None
            };
            if let Some(reason) = reason {
                scene.sources.push(src.clone());
                report.push(json!({ "source": src.id.0, "focused": false, "reason": reason }));
                continue;
            }
            let k0 = u64::MAX;
            scene.sources.push(piece(
                src,
                [bs, ba],
                in_box,
                seed_for(src.samples.seed, k0, 0),
            ));
            for (k, (w, c)) in pieces.iter().zip(&counts).enumerate() {
                scene.sources.push(piece(
                    src,
                    *w,
                    *c,
                    seed_for(src.samples.seed, k0, 1 + k as u64),
                ));
            }
            let shown = |on: bool, r: Rect| if on { json!(r) } else { Value::Null };
            let [ws, wa] = cut_dims(src);
            report.push(json!({
                "source": src.id.0,
                "focused": true,
                "spatial": shown(ws, bs),
                "angular": shown(wa, ba),
                "domainShare": share,
                "rayShare": in_box as f64 / n as f64,
                "pieces": 1 + pieces.len(),
            }));
        }
        (scene, Value::Array(report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oc_core::scene::Spectrum;
    use oc_core::scene3::SamplingSequence;
    use oc_core::vec3::Vec3;
    use oc_trace::TraceJob3;

    /// A box that misses the light loses none of it: a beam 20 mm square
    /// onto a detector 2 mm square in one corner, its box in the opposite
    /// corner. The rest of the domain still carries the corner's light, at a
    /// tenth of the rays: the detector reads its hundredth of the flux, and
    /// the pieces launch the beam's flux, to rounding.
    #[test]
    fn a_box_that_misses_the_light_loses_none_of_it() {
        let mut s = Scene3::new();
        s.add_collimated_beam(
            Vec3::ZERO,
            10.0,
            10.0,
            40_000,
            Spectrum::Monochromatic(0.55),
        );
        s.sources[0].samples.sequence = SamplingSequence::Stratified;
        s.add_area_detector(Vec3::new(50.0, 6.0, 6.0), -Vec3::X, 1.0, 1.0, [1, 1]);
        let cfg = TraceConfig3 {
            ghosts: false,
            ..TraceConfig3::default()
        };
        let mut pilot = Pilot::new(&s, cfg, MIN_PILOT_RAYS);
        pilot.stage = pilot.stages.len();
        pilot.boxes[0] = [[[0.0, 0.25], [0.0, 0.25]], WHOLE];
        let (focused, report) = pilot.focused();
        assert_eq!(report[0]["focused"], true, "{report}");
        assert_eq!(focused.sources.len(), 3);
        let launched: f64 = focused.sources.iter().map(|s| s.flux).sum();
        assert!((launched - 1.0).abs() < 1e-15, "{launched}");
        let read = |scene: Scene3| {
            let mut job = TraceJob3::with_passes(scene, 16, None, cfg);
            while !job.step(1 << 16) {}
            job.finish().detectors[0].total_incident
        };
        let plain = read(s);
        let cut = read(focused);
        assert!((plain - 0.01).abs() < 2e-4, "forward {plain}");
        assert!(
            (cut - 0.01).abs() < 1e-3,
            "focused, a box off the light: {cut}"
        );
    }

    #[test]
    fn a_box_and_its_frame_cover_the_square_once() {
        let b = within(&WHOLE, [2, 5], [0, 3]);
        let f = frame(&b);
        assert_eq!(f.len(), 3, "{f:?}");
        let total: f64 = area(&b) + f.iter().map(area).sum::<f64>();
        assert_eq!(total, 1.0);
        assert!(frame(&WHOLE).is_empty());
        let inner = within(&b, [1, 2], [7, 8]);
        assert_eq!(inner[0][1], b[0][0] + (b[0][1] - b[0][0]) * 2.0 / 8.0);
        assert_eq!(inner[1][1], b[1][1]);
    }
}
