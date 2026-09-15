# The oc-py kernel wheel, built reproducibly in a container.
#
# One recipe, two callers: the `kernel-wheel` CI job and a maintainer building
# a wheel for a platform GitHub has no runner for (a Jetson, say). Build it
# from the REPO ROOT — `oc-py` is a workspace member and cargo has to see the
# whole workspace to parse it:
#
#   docker build -f crates/oc-py/Dockerfile --build-arg STAMP=$(git rev-parse --short=9 HEAD) -o type=local,dest=dist .
#
# The wheel lands in `dist/`, tagged for the BUILD platform's architecture.
#
# ubuntu:22.04 is a deliberate pin, not "some linux". maturin tags the wheel
# with the glibc it linked against, and 22.04 is glibc 2.35 — so this produces
# `manylinux_2_35_<arch>`, the same floor as the x86_64 wheel already vendored
# downstream, and installable on anything newer (the optikit runtime image is
# Debian trixie / glibc 2.41). Building on a newer base would raise that floor
# and silently narrow where the wheel installs.
FROM ubuntu:22.04 AS build

# `rustup` rather than apt: jammy ships rustc 1.75, and the floor here is far
# above it. pyo3 0.29 asks for 1.83 (crates/oc-py/Cargo.toml), but Cargo.lock
# resolves transitive deps that declare `edition = "2024"`, which needs 1.85 —
# a 1.83 toolchain fails at `cargo metadata`, before a line is compiled.
# Pinned rather than `stable` so a wheel rebuilt next year is the same wheel.
ARG RUST_VERSION=1.98.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
ENV PATH=/root/.cargo/bin:$PATH

RUN pip3 install --no-cache-dir 'maturin>=1.5,<2.0'

WORKDIR /src
COPY . .

# The kernel commit, as `+<sha9>` build metadata on the wheel version. It is
# passed in rather than read from git because the build context need not be a
# repository — and because the stamp IS the contract: consumers vendor the
# wasm tarball and this wheel together and pin both to one canvas commit, so a
# wheel that quietly built unstamped would install fine and break that pin.
# Refusing to build without it is the whole point.
ARG STAMP
RUN test -n "${STAMP}" || { echo "error: --build-arg STAMP=<sha9> is required" >&2; exit 1; }

RUN set -eu; \
    # `-i.bak`, not a bare `-i`: BSD sed reads the next argument as the backup
    # suffix, so `-i -E` swallows the `-E`. The suffixed form is the one
    # spelling GNU and BSD both accept, and keeping it identical to the CI job
    # means the two cannot drift.
    sed -i.bak -E "s/^version = \"([0-9][^\"+]*)\"/version = \"\1+${STAMP}\"/" crates/oc-py/pyproject.toml; \
    rm -f crates/oc-py/pyproject.toml.bak; \
    grep -q "^version = \".*+${STAMP}\"$" crates/oc-py/pyproject.toml; \
    grep '^version' crates/oc-py/pyproject.toml

RUN cd crates/oc-py && maturin build --release --out /dist

# `-o type=local,dest=...` exports THIS stage's filesystem, so the last stage
# holds the wheel and nothing else — no 2 GB of toolchain to filter through.
FROM scratch AS export
COPY --from=build /dist/ /
