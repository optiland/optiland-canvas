# openUC2 examples

## `folded-singlet-demo.uc2canvas`

The first integrated demonstration (spec §16.4): a source-to-detector path using
only physics Canvas already validates.

```text
  grid    [0,0,0]      [1,0,0]       [2,0,0]
          source  -->  singlet  -->  45° mirror
                                         |
                                         v  (folded onto -y)
                                     detector  [2,-1,0], rotated 270°
```

Three-line collimated source → N-BK7 singlet → 45° fold mirror → line detector.
Four modules, six ports, three connections, and light that refracts, disperses,
reflects, and lands somewhere you can read a number off.

### Opening it

In the browser, go to `#openuc2` and press **open**, or just press **build the
folded-singlet demo** to assemble the same layout from the catalogue.

Headlessly:

```rust
let project = OpenUc2Project::load(catalog, &std::fs::read_to_string(path)?)?;
let out = oc_trace::trace(project.scene(), &TraceConfig::default());
```

### It is pinned to the development fixture

This setup resolves against the **development fixture** catalogue — invented
UUIDs, `Example Vendor`, `EXAMPLE-*` SKUs, invented prices, and an all-zero
commit. It is not openUC2 module data, and a setup for the real catalogue (PR M1)
will be a different file. See [../../docs/openuc2/catalogue.md](../../docs/openuc2/catalogue.md).

### It is generated, not hand-written

The demo is defined once, in `test_support::folded_singlet_demo`, and emitted:

```sh
cargo run -p oc-openuc2 --example emit_folded_singlet --features test-support \
  -- examples/openuc2/folded-singlet-demo.uc2canvas
```

`the_committed_demo_still_matches_the_code` re-reads this file and checks it
against that definition, so if you edit it by hand the tests will tell you. Output
is deterministic: re-running on an unchanged tree produces an unchanged file.

### Why the camera is rotated 270° and not 90°

Because of where its window is, and it is the one thing here worth a paragraph.

A bare line detector has no sides. At 90° and at 270° the same sensor lies across
the same beam, and the trace is *identical* — same hits, same flux to the last
digit. But the module takes light in through its `-X` face, and only 270° turns
that face toward the mirror. At 90° the camera is mounted backwards: the picture
looks right, the numbers look right, and the only thing that can tell you is the
port graph, which reports its input unconnected.

Turn on the **ports** overlay to see it. That mistake — invisible in the geometry,
invisible in the numbers — is why ports exist.

### What it demonstrates

- catalogue loading, and placement on the 50 mm grid;
- spectral rays (F, d, and C lines) through a real Sellmeier N-BK7;
- refraction, dispersion, and reflection;
- detector hits, reported from the f64 reference tracer;
- a BOM over four distinct products;
- `.uc2canvas` round-trip: the reopened scene is the same scene, byte for byte;
- flat `.ocanvas` export that traces to the same numbers standing alone.

No filter, dichroic, or fluorescence claim is made anywhere in it, and none would
be honest yet — see [../../docs/openuc2/fidelity.md](../../docs/openuc2/fidelity.md).
