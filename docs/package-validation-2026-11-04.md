# Cargo package verification

> Historical result under the pre-upgrade lock. See the [subsequent Rust and dependency upgrade](upgrade-2026-11-04.md) for current versions and verification.

**A standalone `.crate` was built and tested locally.** This is a package
verification, not a crates.io release or a claim about every feature and
external dataset.

On 2026-11-04, Rust/Cargo 1.97.1 and the checked-in historical lockfile were
used offline. Cargo packaged 26 files and verified the package build. The
resulting `serie-rs-0.1.0.crate` had SHA-256
`7dd8776bb0674cb82d69da502ed47ed70f4266af39aa9a1301005afd41fc6782`.
It was extracted into a separate directory outside the repository; commands
below operated on that extracted package, not on the parent workspace:

- `cargo test --offline --locked --lib -j 1 -- --test-threads=1`: **121 passed**.
  Both tests that include the historical declarations ran and passed.
- `cargo test --offline --locked --lib --features cli -j 1
stage_noise::tests::common_random_numbers_two_sigmas_differ_only_in_amplitude
-- --exact --test-threads=1`: **1 passed, 211 filtered out**.
- `cargo run --offline --locked --features cli -j 1 -- list-systems`: passed
  and listed nine synthetic systems.

The packaged declarations and `LICENSE` have the same SHA-256 as their
historical/root counterparts. `SOURCE_PROVENANCE.json` in the repository
records the original declaration paths, new package paths and test-only source
changes. The package includes its own README and license. The package tests
above do not establish every optional-feature test, external-data branch,
cross-language conversion, platform, or end-to-end training.

## Dependency resolution and warnings

The locked offline build passed but Cargo warned that `chacha20 0.10.0`, in the
`rand 0.10.0 → chacha20 0.10.0` dependency chain, is yanked. A separate local
inspection extracted the same crate without its lockfile and generated a new
lockfile **offline from the packages available in that machine's cache**. It
selected `rand 0.10.3`, `rand_core 0.10.1`, and `chacha20 0.10.2` instead of
`0.10.0` for each in the historical lock. This was dependency resolution only:
the new lock was not applied to this project and no data was generated with it.
Other caches, online registries and downstream applications can resolve
differently. RNG-related dependency changes need numerical and provenance
review before any claim that data are reproducible across the two environments.

The CLI build also reported that historical transitive dependency
`proc-macro-error2 2.0.1` may be rejected by a future Rust version. No dependency
was changed in this packaging pass.

## Reproduce

From a checkout with the locked dependencies available:

```sh
cargo package --offline --locked -p serie-rs
mkdir -p /tmp/serie-package-check
tar -xzf target/package/serie-rs-0.1.0.crate -C /tmp/serie-package-check
cd /tmp/serie-package-check/serie-rs-0.1.0
cargo test --offline --locked --lib -j 1 -- --test-threads=1
cargo test --offline --locked --lib --features cli -j 1 \
  stage_noise::tests::common_random_numbers_two_sigmas_differ_only_in_amplitude \
  -- --exact --test-threads=1
cargo run --offline --locked --features cli -j 1 -- list-systems
```

The recorded build used `--allow-dirty` because the audited packaging changes
had not yet been committed, one build job and one test/Rayon/OpenMP/BLAS
thread. A clean committed checkout can omit `--allow-dirty` as shown. The
package archive itself was a generated validation artifact and is not a
published release.
