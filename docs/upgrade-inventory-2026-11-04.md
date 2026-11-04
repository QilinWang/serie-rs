# SERIE-RS direct dependency version inventory

Date: 2026-11-04 (America/New_York). Scope: read-only inspection of
`serie-rs/Cargo.toml` and its `Cargo.lock`, plus the
official crates.io API. The latest column uses crates.io `max_stable_version`
and was checked as not yanked; MSRV is crates.io's `rust_version` field, with
“not declared” meaning the field is absent, not that any Rust version works.
No builds, tests, upgrades, generated lockfiles or numerical runs were done.

| Direct crate | Manifest constraint |  Locked | Latest stable |  Latest MSRV | Status                                            |
| ------------ | ------------------- | ------: | ------------: | -----------: | ------------------------------------------------- |
| ndarray      | ^0.17.2             |  0.17.2 |        0.17.2 |         1.64 | already latest                                    |
| nalgebra     | ^0.34               |  0.34.2 |        0.35.0 |         1.89 | incompatible version requirement                  |
| safetensors  | ^0.8.0              |   0.8.0 |         0.8.0 |         1.80 | already latest                                    |
| serde_json   | ^1                  | 1.0.149 |       1.0.154 |         1.71 | compatible update                                 |
| arrow-array  | ^56                 |  56.2.1 |        60.1.0 |         1.88 | incompatible major; pair with arrow-ipc           |
| arrow-ipc    | ^56                 |  56.2.1 |        60.1.0 |         1.88 | incompatible major; pair with arrow-array         |
| sha2         | ^0.10               |  0.10.9 |        0.11.1 |         1.88 | incompatible version requirement                  |
| csv          | ^1                  |   1.4.0 |         1.4.0 |         1.73 | already latest                                    |
| rand         | ^0.10.0             |  0.10.0 |        0.10.3 |         1.85 | compatible update; RNG behavior needs measurement |
| rand_distr   | ^0.6.0              |   0.6.0 |         0.6.0 |         1.85 | already latest; rand dependency can still change  |
| rand_pcg     | ^0.10.1             |  0.10.1 |        0.10.2 |         1.85 | compatible update; RNG behavior needs measurement |
| rayon        | ^1.11.0             |  1.11.0 |        1.12.0 |         1.80 | compatible update                                 |
| colored      | ^3.1.1              |   3.1.1 |         3.1.1 |         1.80 | already latest                                    |
| tabled       | ^0.20.0             |  0.20.0 |        0.22.0 | not declared | incompatible version requirement                  |
| clap         | ^4.5                |  4.5.60 |         4.6.7 |         1.85 | compatible update                                 |

Caret syntax above is Cargo's normal interpretation of bare version strings.
The highest non-yanked releases still allowed by the old series are
`nalgebra 0.34.2`, `arrow-array 56.2.1`, `arrow-ipc 56.2.1`, `sha2 0.10.9`,
and `tabled 0.20.0`. Thus normal `cargo update` cannot reach their newer
incompatible lines without changing `Cargo.toml` requirements.

## Couplings and caveats

- `arrow-ipc 60.1.0` declares `arrow-array`, `arrow-buffer`, `arrow-data`, and
  `arrow-schema` `^60.1.0`; `arrow-array 60.1.0` declares matching buffer,
  data, and schema dependencies. Upgrade the direct Arrow pair together, then
  verify actual IPC input/output and any Arrow API changes. Tests that lack
  external data must not be counted as exercising those files.
- `nalgebra 0.35.0` declares `simba ^0.10`; under its optional random features it
  declares `rand ^0.10` and `rand_distr ^0.6`. The project currently disables
  nalgebra defaults and enables `std`; check the resolved feature graph, and
  review numerical/decomposition behavior rather than assuming identical output.
- `sha2 0.11.1` declares `digest ^0.11`; the current `sha2 0.10` line reaches
  `0.10.9`. Preserve the actual digest assertion and check the new trait API.
- `tabled 0.20.0` declares optional `tabled_derive ^0.11`; that derive version
  directly declares `proc-macro-error2 ^2.0.1`. `tabled 0.22.0` declares
  `tabled_derive ^0.12`; the crates.io dependency list for
  `tabled_derive 0.12.0` does not include `proc-macro-error2`. This is a reason
  to check whether the earlier future-compatibility warning disappears, not a
  guarantee: the final cargo feature/dependency tree and compiler report decide.
- The existing lock contains `chacha20 0.10.0`, which official crates.io
  metadata marks yanked. The latest stable `chacha20 0.10.2` is not yanked.
  A lock update can change it via the `rand` chain; this removes that known
  locked version, but does not establish bit-identical noise or trajectory data.
- Rust/Cargo 1.99.1 (official latest stable at this observation) exceeds all
  _declared_ MSRVs in this table. `tabled 0.22.0` declares none; its successful
  actual build must be checked. The compiler and toolchain installation should
  be project-scoped or side-by-side if the shared default is in use elsewhere.
- Before and after dependency changes, verify the same previously named CLI
  inputs/seed and complete clean/noise/noisy tensor bytes. File hashes alone
  are insufficient because earlier headers had semantically equivalent
  metadata in different byte orders. If outputs differ, report which tensors,
  how and the scope, and do not silently change experimental settings or old
  scientific results.

## Primary sources

Each version/MSRV/yanked field was read from
`https://crates.io/api/v1/crates/<crate>` and each coupling from
`https://crates.io/api/v1/crates/<crate>/<version>/dependencies`; these are the
registry's official metadata, not an assertion that the final dependency
resolver will necessarily select the same transitive versions. The observed
older lockfile and manifest are in the standalone repository at main
`2851274247a5ef14058828519e9a43bff6a6d655`.

- Cargo update semantics: https://doc.rust-lang.org/cargo/commands/cargo-update.html
- Rust official releases: https://blog.rust-lang.org/releases/
