//! Project-level undo (§14.8).
//!
//! A snapshot is the whole [`OpenUc2Setup`]: the instances, the custom Canvas
//! content, and the view state the scene is materialized against. That is the
//! complete input to materialization, so restoring one and regenerating gives
//! back exactly the project that was there — the scene is derived and is never
//! snapshotted itself (§5.1).
//!
//! This is why history lives here rather than in the core's object history. Undo
//! that reached the optical objects but not the placements would leave a module
//! whose setup says one cell and whose geometry sits in another, which is the
//! divergence §14.8 exists to prevent.
//!
//! # Gestures
//!
//! One *settled* operation is one undo step (§15.6). A drag is not one call: it
//! is a `move_module` per pointer move, and recording each would bury the user's
//! previous edit under fifty identical-looking steps. So a caller wraps a drag in
//! a gesture, and everything inside it collapses into the single snapshot taken
//! at the start. A gesture that ends where it began records nothing at all.
//!
//! Calls made outside a gesture — a click to place, a rotate button — record
//! immediately, so a caller that never opens one still gets correct history.

use crate::setup::OpenUc2Setup;

/// How many undo steps to keep.
///
/// A snapshot is a setup, not a scene: a few hundred bytes per placed module. The
/// cap is here so a long session cannot grow without bound, not because the
/// snapshots are expensive.
const LIMIT: usize = 128;

/// The undo/redo stacks.
///
/// Holds no scene and no catalogue, only setups, so it cannot disagree with the
/// project about anything.
#[derive(Debug, Default)]
pub(crate) struct History {
    undo: Vec<OpenUc2Setup>,
    redo: Vec<OpenUc2Setup>,
    /// True between `begin_gesture` and `end_gesture`.
    gesture: bool,
    /// The state as it was when the open gesture's first mutation ran. Held back
    /// rather than pushed, so a gesture that changes nothing can be dropped
    /// without having already cleared the redo stack.
    pending: Option<OpenUc2Setup>,
}

impl History {
    /// Whether [`record`](Self::record) would use a snapshot right now.
    ///
    /// Lets the caller skip cloning the setup on every move of a drag, where the
    /// gesture already holds the only snapshot that will be kept.
    pub(crate) fn wants_before(&self) -> bool {
        !self.gesture || self.pending.is_none()
    }

    /// Record the state a successful mutation just changed.
    ///
    /// `before` must be the setup as it was *before* the mutation, and this must
    /// run only once the mutation has actually succeeded: a refused placement is
    /// not an edit and must not become an undo step.
    pub(crate) fn record(&mut self, before: OpenUc2Setup) {
        if self.gesture {
            if self.pending.is_none() {
                self.pending = Some(before);
            }
            return;
        }
        self.push_undo(before);
    }

    fn push_undo(&mut self, s: OpenUc2Setup) {
        self.undo.push(s);
        // A new edit invalidates any future that was undone away.
        self.redo.clear();
        if self.undo.len() > LIMIT {
            self.undo.remove(0);
        }
    }

    /// Begin coalescing mutations into one undo step.
    ///
    /// Idempotent: a second call while a gesture is open continues it rather than
    /// splitting the drag in two.
    pub(crate) fn begin_gesture(&mut self) {
        self.gesture = true;
    }

    /// Settle the open gesture, given the setup it ended at.
    ///
    /// The comparison is what makes a drag that returns to its starting cell a
    /// no-op: it leaves both stacks untouched, so the user's redo survives having
    /// picked a module up and put it back.
    pub(crate) fn end_gesture(&mut self, current: &OpenUc2Setup) {
        self.gesture = false;
        if let Some(before) = self.pending.take() {
            if before != *current {
                self.push_undo(before);
            }
        }
    }

    pub(crate) fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub(crate) fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// The state to go back to, without committing to it.
    ///
    /// Peek-then-pop: restoring is fallible, and a failed undo must leave the
    /// stacks exactly as it found them.
    pub(crate) fn peek_undo(&self) -> Option<&OpenUc2Setup> {
        self.undo.last()
    }

    pub(crate) fn peek_redo(&self) -> Option<&OpenUc2Setup> {
        self.redo.last()
    }

    /// Commit an undo: drop the restored state and remember where we came from.
    pub(crate) fn commit_undo(&mut self, came_from: OpenUc2Setup) {
        self.undo.pop();
        self.redo.push(came_from);
    }

    /// Commit a redo. The mirror of [`commit_undo`](Self::commit_undo): the undo
    /// stack grows by exactly the step the redo stack loses, so walking back and
    /// forth over the same edit is stable.
    pub(crate) fn commit_redo(&mut self, came_from: OpenUc2Setup) {
        self.redo.pop();
        self.undo.push(came_from);
    }

    /// The two stacks' sizes. There is no reset: opening a different document
    /// builds a different project, and a fresh project starts with fresh history.
    #[cfg(test)]
    pub(crate) fn depth(&self) -> (usize, usize) {
        (self.undo.len(), self.redo.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(layer: i32) -> OpenUc2Setup {
        let mut s = OpenUc2Setup::new("openuc2-core-50");
        // Any field that distinguishes one snapshot from another will do.
        s.view.active_layer = layer;
        s
    }

    #[test]
    fn an_edit_outside_a_gesture_records_immediately() {
        let mut h = History::default();
        assert!(!h.can_undo());
        h.record(setup(0));
        assert!(h.can_undo());
        assert_eq!(h.peek_undo(), Some(&setup(0)));
    }

    #[test]
    fn a_gesture_records_one_step_however_many_mutations_it_makes() {
        let mut h = History::default();
        h.begin_gesture();
        // A drag: the first move captures the pre-drag state, the rest do not.
        h.record(setup(0));
        assert!(!h.wants_before(), "the gesture already has its snapshot");
        h.record(setup(1));
        h.record(setup(2));
        h.end_gesture(&setup(9));

        assert_eq!(h.depth(), (1, 0));
        assert_eq!(h.peek_undo(), Some(&setup(0)), "the state before the drag");
    }

    #[test]
    fn a_gesture_that_changes_nothing_records_nothing() {
        // Picking a module up and putting it back is not an edit, and must not
        // cost the user their undo step or their redo stack.
        let mut h = History::default();
        h.record(setup(5));
        h.commit_undo(setup(6));
        assert_eq!(h.depth(), (0, 1));

        h.begin_gesture();
        h.record(setup(7));
        h.end_gesture(&setup(7));

        assert_eq!(h.depth(), (0, 1), "both stacks untouched");
    }

    #[test]
    fn a_new_edit_drops_the_redone_future() {
        let mut h = History::default();
        h.record(setup(0));
        h.commit_undo(setup(1));
        assert!(h.can_redo());

        h.record(setup(2));
        assert!(!h.can_redo(), "the old future is unreachable now");
    }

    #[test]
    fn undo_and_redo_move_one_step_each_way() {
        let mut h = History::default();
        h.record(setup(0));
        assert_eq!(h.depth(), (1, 0));

        h.commit_undo(setup(1));
        assert_eq!(h.depth(), (0, 1));
        assert_eq!(h.peek_redo(), Some(&setup(1)));

        h.commit_redo(setup(0));
        assert_eq!(h.depth(), (1, 0));
        assert_eq!(h.peek_undo(), Some(&setup(0)));
    }

    #[test]
    fn the_stack_is_capped_and_drops_the_oldest() {
        let mut h = History::default();
        for i in 0..(LIMIT as i32 + 10) {
            h.record(setup(i));
        }
        assert_eq!(h.depth(), (LIMIT, 0));
        // The most recent steps are the ones worth keeping.
        assert_eq!(h.peek_undo(), Some(&setup(LIMIT as i32 + 9)));
    }

    #[test]
    fn a_gesture_that_never_mutates_is_harmless() {
        let mut h = History::default();
        h.begin_gesture();
        h.end_gesture(&setup(0));
        assert_eq!(h.depth(), (0, 0));
    }

}
