# The module catalogue

The browser loads a **generated, pinned, static** catalogue. It never talks to
GitHub.

That is a deliberate constraint, not a limitation to work around. A UI that fetched
modules live would be subject to CORS and availability, and — worse — a setup you
saved today would silently mean something else tomorrow when a module changed
under its pin.

## What ships

```text
dist/canvas/
  catalog.json         normalized modules, grid profiles, provenance
  catalog-report.json  coverage, machine-readable
  catalog-report.md    coverage, for humans
  assets/              content-addressed icons and .ocanvas fragments
```

`catalog.json` records where it came from and at what revision:

```json
{
  "format": "openuc2-canvas-catalog",
  "version": "1.0.0",
  "source": { "repository": "openUC2/openUC2-optikit-modules", "commit": "<sha>" },
  "descriptor_schema_version": "2.0.0",
  "grid_profiles": [ { "id": "openuc2-core-50", "pitch_mm": [50, 50, 55], "...": "" } ],
  "modules": []
}
```

Each module entry carries enough to browse and place it — name, category,
footprint, ports, capabilities, assets, commercial fields, hashes, source path —
without fetching `module.yaml`.

The **UI must not silently update to a newer module revision.** A `.uc2canvas` pins
the catalogue commit; see [setup-format.md](setup-format.md).

## Building it

The compiler lives in the module repository (`scripts/build_canvas_catalog.py`,
PR M1). It discovers `modules/**/module.yaml` **recursively** — the repository is
organized into nested category directories, and a scan of `modules/`' immediate
children misses most of it.

```sh
python ../openUC2-optikit-modules/scripts/build_canvas_catalog.py \
  --output web/public/catalog/openuc2 \
  --source-commit "$(git -C ../openUC2-optikit-modules rev-parse HEAD)"
```

It is **deterministic**: same source tree, byte-identical output. Modules sort by
`(category, group, name, id, version)`, assets are content-addressed, and no
wall-clock time enters the output unless explicitly asked for. `--strict` exits
non-zero on invalid descriptors, missing assets, duplicate identities, invalid
fragments, or undeclared unsupported optical content.

## The development fixture

**The module repository is not checked out yet, so what currently ships is a
fixture**, generated from
[`test_support.rs`](../../crates/oc-openuc2/src/test_support.rs):

```sh
cargo run -p oc-openuc2 --example emit_dev_catalog --features test-support \
  -- web/public/catalog/openuc2
```

Everything in it is invented: the UUIDs, `Example Vendor`, the `EXAMPLE-*` SKUs,
the prices, and an all-zero source commit that reads as **unpinned** everywhere a
real catalogue carries a sha. The UI says so, in red, permanently. Edit
`test_support.rs` and regenerate — never hand-edit `web/public/catalog/openuc2/`.

It exists so the loading path is exercised end to end rather than mocked, and so
the fixture the browser loads and the fixture the Rust tests assert against cannot
drift apart. It drops out when M1 lands, into the same directory with the same
shape.

## Coverage is the deliverable

The catalogue ingests **every valid descriptor**, not only the ready ones, and each
module advertises its four readiness axes independently
([fidelity.md](fidelity.md)). A module with an icon and a footprint is useful for
layout even with no optical model at all.

The coverage report — total, valid/invalid, layout-ready, the optical/CAD/control
breakdowns, missing icons, missing fragments, duplicate ids, unsupported
categories, and a per-module issue list — is a standalone milestone. It is meant
to be useful in an openUC2 status meeting before the editor integration is
finished, because "which of our modules are actually ready?" is a question worth
answering on its own.
