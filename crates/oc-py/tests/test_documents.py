"""The file contract: every document the canvas writes is one the bridge reads.

The browser saves version 2 and the bridge used to read version 1 only, so
every file saved from the app failed here with a JSON type error about detector
bins. These tests hold the bridge to files the app actually saved (the
fixtures come from ``web/tests/save.spec.ts``, which re-saves the same scenes
and fails when they drift), to the v1 documents the project ships, and to the
three ways a document can fail to read, which raise three different errors.

None of this needs Optiland.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import optiland_canvas as oc

HERE = Path(__file__).parent
FIXTURES = HERE / "data"
BROWSER_FILES = sorted(FIXTURES.glob("browser-*.ocanvas"))
# The v1 documents the web app ships with its openUC2 catalogue.
V1_FILES = sorted((HERE / "../../../web/public/catalog/openuc2/assets").resolve().glob("*.ocanvas"))


def _doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_fixtures_are_there():
    names = {p.name for p in BROWSER_FILES}
    assert {"browser-default.ocanvas", "browser-splitter.ocanvas"} <= names
    assert V1_FILES, "the v1 catalogue documents moved"


@pytest.mark.parametrize("path", BROWSER_FILES, ids=lambda p: p.name)
def test_a_file_the_browser_saved_reads_as_version_2(path: Path):
    assert oc.validate(str(path)) == 2
    assert _doc(path)["version"] == 2
    # `load` validates and hands back the text as it was.
    assert oc.load(str(path)) == path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", BROWSER_FILES, ids=lambda p: p.name)
def test_saving_a_browser_file_keeps_it_version_2_and_changes_nothing(path: Path, tmp_path: Path):
    out = tmp_path / path.name
    oc.save(str(path), str(out))
    again = _doc(out)
    assert again["version"] == 2
    # The browser writes the same normal form the bridge does, so a save is
    # the identity on what it saved.
    assert again == _doc(path)


@pytest.mark.parametrize("path", V1_FILES, ids=lambda p: p.name)
def test_a_version_1_document_still_reads_and_stays_version_1(path: Path, tmp_path: Path):
    assert oc.validate(str(path)) == 1
    out = tmp_path / path.name
    oc.save(str(path), str(out))
    assert _doc(out)["version"] == 1, "normalizing never migrates"


def test_the_readable_versions_are_named():
    assert oc.READABLE_VERSIONS == (1, 2)


# --------------------------------------------------------------------------- #
# Three ways to fail, three errors                                            #
# --------------------------------------------------------------------------- #
def _default() -> dict:
    return _doc(FIXTURES / "browser-default.ocanvas")


@pytest.mark.parametrize(
    "text",
    [
        "this is not json",
        "[1, 2, 3]",
        json.dumps({"format": "some-other-tool", "version": 2}),
        json.dumps({"format": "ocanvas", "version": "two"}),
    ],
    ids=["not json", "not an object", "another format", "a version that is not a number"],
)
def test_the_wrong_file_is_not_a_document(text: str):
    with pytest.raises(oc.NotADocumentError):
        oc.validate(text)


def test_a_document_from_the_future_names_its_version():
    doc = _default()
    doc["version"] = 3
    with pytest.raises(oc.UnsupportedVersionError, match="version 3"):
        oc.validate(doc)


def test_a_broken_scene_is_invalid_not_unreadable():
    # Readable version, right format, and a detector with no pixels.
    doc = _default()
    doc["detectors"][0]["bins"] = [0, 64]
    with pytest.raises(oc.InvalidSceneError, match="bins"):
        oc.validate(doc)
    # A field of the wrong type is the scene's problem too, not the file's.
    doc = _default()
    doc["detectors"][0]["bins"] = "sixty-four"
    with pytest.raises(oc.InvalidSceneError):
        oc.validate(doc)


def test_every_document_error_is_still_a_value_error():
    for cls in (
        oc.NotADocumentError,
        oc.UnsupportedVersionError,
        oc.InvalidSceneError,
        oc.NotExportableError,
    ):
        assert issubclass(cls, oc.OcanvasError)
        assert issubclass(cls, ValueError)


def test_every_entry_point_reads_the_same_way():
    doc = _default()
    doc["version"] = 3
    for call in (oc.validate, oc.scene_to_seq, lambda d: oc.scene3_sequence(d, 5, 6)):
        with pytest.raises(oc.UnsupportedVersionError):
            call(doc)


# --------------------------------------------------------------------------- #
# The traced chain of a browser file                                          #
# --------------------------------------------------------------------------- #
def test_the_default_scene_s_axial_ray_threads_the_singlet():
    doc = _default()
    source = doc["sources"][0]["id"]
    detector = doc["detectors"][0]["common"]["id"]
    seq = oc.scene3_sequence(doc, source, detector)
    lens = doc["bodies"][0]["common"]["id"]
    assert [h["object"] for h in seq["hits"]] == [lens, lens]
    assert seq["terminal"]["detector"] == detector
