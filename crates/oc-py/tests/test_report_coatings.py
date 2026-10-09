"""The coating validation report (coating-fidelity spec CF-12, C32) and its
refusal to mix profiles in one column (C34)."""

from __future__ import annotations

import pytest

import report_coatings as rep


def test_c32_the_report_lists_every_case_within_tolerance():
    rows = rep.rows()
    cases = {p.stem for p in rep.FIXTURES.glob("*.json")} - {"tolerances"}
    assert {r.case for r in rows} == cases
    assert {r.oracle for r in rows} == {"tmm", "optiland"}
    assert all(r.passes for r in rows), [r for r in rows if not r.passes]
    text = rep.render(rows, "abc123")
    assert "at commit `abc123`" in text
    assert f"{len(rows)} comparisons, 0 over tolerance." in text


def test_c32_the_committed_report_is_current():
    committed = rep.DEFAULT_OUT.read_text(encoding="utf-8")
    assert rep.comparable(committed) == rep.comparable(rep.render(rep.rows(), "any"))


def test_c34_a_table_without_a_profile_column_refuses_two_profiles():
    rows = rep.rows()[:2]
    mixed = [rows[0], rep.Row(**{**rows[1].__dict__, "profile": "Reference"})]
    with pytest.raises(ValueError, match="cannot share a table without a profile column"):
        rep.render(mixed, "x", profile_column=False)
    # One profile is fine without the column, and two are fine with it.
    rep.render(rows, "x", profile_column=False)
    assert "Reference" in rep.render(mixed, "x")
