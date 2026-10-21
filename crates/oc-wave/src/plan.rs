//! The choices a wave path makes from the field itself, recorded and
//! replayed (R09 and finding 3 of the expert review of the wave engine).
//!
//! A wave path is a linear operator on the launched field only while what it
//! decides is fixed: which propagator a step runs, the window a field is
//! carried onto, the sphere a field is read about, the orders a modulator's
//! field is cut into, the readout's points. Each of those is chosen from the
//! field it meets, so the path as it runs is not linear: twice the field
//! gives twice the result, but the sum of two fields of different widths can
//! be carried on windows neither would get alone. A [`PathPlan`] holds every
//! such choice, in the order the path made them; a job that replays it
//! ([`crate::path::WaveJob::replaying`]) makes none of its own, and is one
//! fixed linear operator on whatever field it launches
//! ([`crate::path::WaveJob::with_launch`]).
//!
//! Each choice is made through [`decide`], which a job's run routes to its
//! recorder: recording, it runs the choice and keeps its value; replaying, it
//! hands back the recorded value, in order, and refuses by name when the run
//! asks for a choice the plan does not hold there. Outside a job (a
//! propagator called on its own) `decide` simply runs the choice.

use std::any::Any;
use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
struct Entry {
    label: &'static str,
    /// The value as `Debug` writes it: what two plans are compared by.
    shown: String,
    value: Arc<dyn Any + Send + Sync>,
    /// How many choices were made while this one was (nested in it): a
    /// replay that returns this one skips them.
    span: usize,
}

/// Every choice a wave path made from the field, in the order made, and the
/// path it was made on (its spec and its chains' events), which a replay
/// must match.
#[derive(Clone, Default)]
pub struct PathPlan {
    entries: Vec<Entry>,
    context: String,
}

impl PathPlan {
    /// How many choices the plan holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The choices' names, in order.
    pub fn labels(&self) -> Vec<&'static str> {
        self.entries.iter().map(|e| e.label).collect()
    }

    /// How many choices of one name the plan holds.
    pub fn count(&self, label: &str) -> usize {
        self.entries.iter().filter(|e| e.label == label).count()
    }

    /// How many choices two plans make differently, the first `n` of each
    /// compared in turn, and how many one holds past the other.
    pub fn differences(&self, other: &PathPlan) -> usize {
        let n = self.len().min(other.len());
        let differ = (0..n).filter(|&i| self.entries[i].label != other.entries[i].label || self.entries[i].shown != other.entries[i].shown).count();
        differ + self.len().max(other.len()) - n
    }
}

impl PartialEq for PathPlan {
    fn eq(&self, other: &Self) -> bool {
        self.context == other.context
            && self.entries.len() == other.entries.len()
            && self.entries.iter().zip(&other.entries).all(|(a, b)| a.label == b.label && a.shown == b.shown && a.span == b.span)
    }
}

impl fmt::Debug for PathPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.entries.iter().map(|e| format!("{}: {}", e.label, e.shown))).finish()
    }
}

/// A job's recorder: the plan it records, or the plan it replays and where
/// it has got to in it.
#[derive(Clone, Default)]
pub(crate) struct Recorder {
    plan: PathPlan,
    cursor: Option<usize>,
    mismatch: Option<String>,
}

impl Recorder {
    /// A recorder of the choices made on the path `context` describes.
    pub(crate) fn recording(context: String) -> Self {
        Recorder { plan: PathPlan { entries: Vec::new(), context }, cursor: None, mismatch: None }
    }

    /// Whether `plan` was recorded on the path `context` describes.
    pub(crate) fn fits(plan: &PathPlan, context: &str) -> bool {
        plan.context == context
    }

    /// A recorder that replays `plan` from its first choice.
    pub(crate) fn replay(plan: &PathPlan) -> Self {
        Recorder { plan: plan.clone(), cursor: Some(0), mismatch: None }
    }

    /// The plan recorded, or the one replayed.
    pub(crate) fn plan(&self) -> &PathPlan {
        &self.plan
    }

    /// Why the replay could not go on, if it could not; and, once the run
    /// is over (`done`), the choices the plan holds that the run never
    /// reached.
    pub(crate) fn check(&self, done: bool) -> Result<(), String> {
        if let Some(m) = &self.mismatch {
            return Err(m.clone());
        }
        match self.cursor {
            Some(c) if done && c < self.plan.len() => Err(format!(
                "the plan holds {} choices and the run made {c}: the scene or the spec is not the one it was recorded on",
                self.plan.len()
            )),
            _ => Ok(()),
        }
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

/// Run `f` with `rec` as the recorder its choices go to, and put it back.
pub(crate) fn with<R>(rec: &mut Recorder, f: impl FnOnce() -> R) -> R {
    let prev = ACTIVE.with(|a| a.replace(Some(std::mem::take(rec))));
    let out = f();
    *rec = ACTIVE.with(|a| a.replace(prev)).unwrap_or_default();
    out
}

/// A choice made from the field: `f` computes it. Recording, its value is
/// kept under `label`; replaying, the recorded value is returned instead and
/// `f` is not run.
pub(crate) fn decide<T: Clone + fmt::Debug + Send + Sync + 'static>(label: &'static str, f: impl FnOnce() -> T) -> T {
    enum Step<T> {
        Free,
        Record(usize),
        Replayed(T),
    }
    // What to do, decided without holding the recorder while `f` runs: it may
    // make choices of its own.
    let step = ACTIVE.with(|a| {
        let mut a = a.borrow_mut();
        let Some(r) = a.as_mut() else { return Step::Free };
        match r.cursor {
            None => {
                r.plan.entries.push(Entry { label, shown: String::new(), value: Arc::new(()), span: 0 });
                Step::Record(r.plan.entries.len() - 1)
            }
            Some(_) if r.mismatch.is_some() => Step::Free,
            Some(c) => match r.plan.entries.get(c) {
                Some(e) if e.label == label => match e.value.downcast_ref::<T>() {
                    Some(v) => {
                        let v = v.clone();
                        r.cursor = Some(c + 1 + e.span);
                        Step::Replayed(v)
                    }
                    None => {
                        r.mismatch = Some(format!("choice {c} of the plan ({label}) holds a value of another kind"));
                        Step::Free
                    }
                },
                Some(e) => {
                    r.mismatch = Some(format!("choice {c} of the plan is the {}, where the run makes the {label}", e.label));
                    Step::Free
                }
                None => {
                    r.mismatch = Some(format!("the plan holds {c} choices, and the run makes more (the {label})"));
                    Step::Free
                }
            },
        }
    });
    match step {
        Step::Free => f(),
        Step::Replayed(v) => v,
        Step::Record(i) => {
            let v = f();
            ACTIVE.with(|a| {
                if let Some(r) = a.borrow_mut().as_mut() {
                    let span = r.plan.entries.len() - 1 - i;
                    r.plan.entries[i] = Entry { label, shown: format!("{v:?}"), value: Arc::new(v.clone()), span };
                }
            });
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_replay_returns_the_recorded_choices_in_order_and_skips_what_they_nested() {
        let mut rec = Recorder::default();
        let run = |x: f64| {
            let a = decide("outer", || decide("inner", || x) + 1.0);
            let b = decide("second", || x * 3.0);
            (a, b)
        };
        let first = with(&mut rec, || run(2.0));
        assert_eq!(first, (3.0, 6.0));
        assert_eq!(rec.plan().labels(), vec!["outer", "inner", "second"]);
        let mut again = Recorder::replay(rec.plan());
        let replayed = with(&mut again, || run(10.0));
        assert_eq!(replayed, (3.0, 6.0));
        assert!(again.check(true).is_ok());
        // A run that asks for another choice is refused by name.
        let mut wrong = Recorder::replay(rec.plan());
        with(&mut wrong, || decide("second", || 1.0));
        assert!(wrong.check(false).unwrap_err().contains("outer"));
        // Outside a job a choice is simply made.
        assert_eq!(decide("free", || 7), 7);
    }
}
