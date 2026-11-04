# Independent design review: Rust shock example and Python interchange

Date: 2026-11-04. Scope: source inspection and read-only examination of the
retained OU test artifacts; no simulation, build, training or benchmark run.

## Findings

1. The CLI `generate` and `generate_dgp_bank` are distinct producer paths.
   `serie-rs/src/main.rs` passes only `system`, `seed` and `noise_seed` to
   `write_dgp_bank_safetensors`; that writer adds `schema` and `feature_names`.
   The Python `load_dgp_bank_safetensors` can read clean/noise tensors, but
   `dgp_bank_from_tensors` requires `source_name`, `noise_kind`, `is_stochastic`,
   trajectory/noise seed bases and spawned seed lists. It will refuse the
   CLI-generated artifact for missing metadata. Do not claim the CLI output
   rebuilds a full Python DGPBank, and do not invent the missing metadata.
   The CLI's shock selection is recorded in its separate manifest summary, not
   in the safetensors metadata.
2. The existing OU Student-t files were found under names produced by the Rust
   test `student_t3_bank_records_its_noise_law_in_metadata`, and archived test
   output records that test as passing. Python has read and rebuilt the three
   retained files. Their path names, metadata and time support a likely source,
   but do not by themselves cryptographically bind each file to the precise
   binary or commit: describe them as retained test-named files and include
   exact hashes, or generate and bind a fresh sample before stronger claims.
   Python reading Rust output does not prove Python/Rust RNG bit identity.
3. Three retained OU files have the same parsed header and payload bytes but
   different serialized header bytes and full-file hashes. Inspecting the JSON
   shows the `__metadata__` key order varies. The Rust writer collects metadata
   in a `HashMap`; this explains why file-hash equivalence is not the same as
   array-byte equivalence for these artifacts. Do not generalize beyond the
   checked files or claim the whole safetensors file is bitwise reproducible.

## Suggested demonstration checks

The unchanged `lorenz_param_shock` source has original parameters `(10, 28,
8/3)`, post-shock `(10.1, 28.1, 8.1/3)`, 35% of 35,999 integration steps,
RK4 with `dt=0.01`, no burn-in, and initial condition `(1, 0.98, 1.1)`.
The shock is dispatched at integration step 12,599; the returned trajectory
includes the initial condition. Generate a matched no-shock example with the
same CLI seeds and Gaussian noise dose, then compare the full pre-shock clean
prefix, noise draw bytes, and post-shock differences. Both implementations have
the same RK4 arithmetic in source but use separate code paths, so any bitwise
pre-shock equality is an empirical check, not a presumed guarantee. Check
`noisy = clean + float32(sigma) * noise[0]` and inspect the stored manifest.
The `float64` integration values are converted to `float32` for the artifact;
make no claim that artifact precision represents internal integration precision.

Label a visualization as trajectories, not forecasting outputs or a model
performance result. Verification of one specific preset and producer path does
not establish all systems, shocks, noise kinds, reverse interchange, Cargo
packaging or model-training behavior.
