# Independent design review: SERIE-RS Rust and dependency upgrades

Date: 2026-11-04. Scope: source, tests and prior public validation notes in the
standalone Rust workspace. No builds, tests, experiments, GPU use or repository
edits were performed for this review.

## Main risk

The owner explicitly requested upgrading Rust, Cargo and all upgradeable
dependencies. A successful compile or passing suite does not establish that a
seed produces the same data after an RNG or numerical dependency changes.
`serie-rs/src/dgp_bank.rs:545-605` creates one `rand_pcg::Pcg64` stream per
spawned noise seed; Gaussian samples `rand_distr::StandardNormal` as f64 and
casts to f32, while Laplace, Student-t3 and contaminated noise have their own
sampling paths. `serie-rs/src/stage_generate.rs:626-630,875-879` also draws
stochastic trajectory innovations and regime uniforms. API-compatible updates
can therefore still change data bytes or numerical behavior. The default tests
check many semantics and distributional properties but generally do not prove
full old/new output equivalence.

## Measurement design

1. Record pinned source revision, lockfile, exact Rust/Cargo versions, manifest
   and feature selection. Keep the old and candidate sources recoverable with
   scoped commits or isolated workspaces. Do not change the global toolchain
   for other working sessions; install the newer stable toolchain alongside it
   and select it explicitly or via a project pin.
2. If practical, measure three separately identified cases: old toolchain and
   old locked dependencies; new toolchain with old locked dependencies; new
   toolchain and upgraded dependencies. This distinguishes compiler effects
   from dependency effects. Build and run serially with capped jobs/threads,
   no training and no GPU. The user's existing same-preset Lorenz control and
   `param` shock with sigma 0.1, trajectory seed 42, noise seed 1000 and eight
   draws provide authorized existing settings. Include a stochastic CLI system
   such as the already-existing OU preset with the same existing CLI defaults,
   and, if using library-level checks, existing Gaussian, Laplace, Student-t3
   and contaminated test inputs rather than inventing scientific settings.
3. Record actual clean/noise/noisy arrays, manifest, metadata, shapes, dtype,
   precision, selected feature flags and seed semantics. Compare per-tensor
   bytes and bitwise mismatch counts first. Whole `.safetensors` file hashes
   are insufficient because serialized metadata map order can vary while
   parsed metadata and tensor bytes remain identical. If a tensor differs,
   report mismatch count and numerical magnitude without calling it equivalent
   based on an unapproved tolerance; identify which case and axis changed.
   Confirm seed spawning and metadata values separately. These CLI output
   arrays are f32, while Lorenz integration uses f64 before export: equal
   exports alone do not prove bit identity of internal f64 trajectories.
4. Preserve the existing CLI direct trajectory-seed semantics versus the full
   bank's spawned seed semantics. Upgrading `rand` or related packages does
   not authorize silently changing the input seed, sample distribution, dtype,
   floating-point equations, intervention, generator or defaults. If a newer
   upstream version changes fixed-seed output, report the actual result and
   limitation. Do not rewrite historical assertions to make a changed result
   look equivalent; distinguish explicitly approved dependency upgrade from
   any additional proposed scientific change.
5. Exercise `cargo check` / `cargo test` for default and relevant feature paths;
   specifically include `draw_protocol::tests::spawned_draw_seeds_are_stable_and_not_consecutive`,
   `stage_generate::tests::test_discrete_seeded_determinism_and_seed_sensitivity`,
   `dgp_bank::tests::extending_noise_bank_preserves_the_existing_draw_prefix_bitwise`,
   the real paired-noise test, and the declaration tests. Honor existing
   expected outputs and tolerances without altering them. A single green focused
   test is not coverage of optional known-data branches; report skipped data.
6. Build a real `.crate`, independently extract it, repeat the package library
   tests and relevant CLI smoke, and verify the expected fixture/license files.
   Distinguish a package build, actual extracted-package tests and publication;
   no crates.io publication follows automatically. Recheck Rust/Python artifact
   loading and metadata only to the extent needed; do not claim cross-language
   random byte identity or full-bank reconstruction from CLI's incomplete
   header.
7. Inspect all direct and transitive upgrades, explicitly distinguishing
   semver-compatible lock refresh from changes to declared version ranges.
   Cross-major Arrow/`sha2` updates may require API fixes and feature/format
   checks; do not claim all dependencies upgraded if yanked, unavailable,
   MSRV-incompatible or untested versions remain. Cargo resolution is not
   proof of a successful build. Compare yanked warnings, fresh consumer lock
   behavior and optional `cli`/known-data features as separate evidence.

## Verdict

The owner's instruction authorizes the dependency upgrade; a reversible,
isolated upgrade plus the above audit is the correct next step. It does not
justify unannounced changes to scientific settings or treating statistical
similarity as exact reproducibility. Report positive and negative results and
leave any truly incompatible semantic decision visible rather than concealing
it with a new default.
