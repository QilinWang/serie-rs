# CLI artifacts and full data banks

**The CLI exports usable tensors, but its trajectory seed is not a full-bank
trajectory seed base.** The two producer paths currently carry different
provenance. This note records what they do; it does not change their inputs,
random-number generators, precision, or output values.

|                    | CLI `generate`                                                               | Full Rust `generate_dgp_bank` / Python `build_dgp_bank`                                         |
| ------------------ | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Clean trajectories | One `[1, T, D]` trajectory                                                   | `[N_trajectory, T, D]`                                                                          |
| Trajectory seed    | `--seed` is assigned directly to the trajectory spec                         | Child seeds are derived from a trajectory seed base; each child is used for a draw              |
| Noise draws        | `--noise-seed` is a base; child seeds select separate Rust noise streams     | Both full-bank implementations also derive noise child seeds from a base                        |
| Noise distribution | CLI uses Gaussian                                                            | Full bank takes a noise process                                                                 |
| Intervention       | Preset appears in CLI `manifest.json`; the tensor header does not contain it | A full bank needs truthful source/intervention provenance; do not infer it from the system name |

For a concrete seed identity, when the base is `42`, the first derived
trajectory seed is `803958421`. The CLI instead passes `42` directly. These
numbers describe the existing code; they are not a new experiment setting.
Setting `trajectory_seed_base = 42` and recording `803958421` as the CLI's
actual trajectory seed would misstate provenance. Switching the CLI to that
child seed can change stochastic trajectories and needs an explicit scientific
decision.

Both implementations use the same child-seed derivation, but **equal child
seeds do not imply equal Rust and Python noise arrays**. The Rust path uses its
`Pcg64` and samples f64 normal values before converting to f32; the Python
Gaussian path uses NumPy's generator and samples f32. These are different RNG
and sampling paths.

## What Python can read

The CLI safetensors header contains `schema`, `feature_names`, `system`, `seed`,
and `noise_seed`. Its separate manifest includes the shock preset, sigma and
shape summary. Python's `load_dgp_bank_safetensors` loads the named arrays and
feature names. It does not prove full-bank provenance.

`dgp_bank_from_tensors` requires `source_name`, `noise_kind`, `is_stochastic`,
`trajectory_seed_base`, `noise_seed_base`, `trajectory_draw_seeds`, and
`noise_draw_seeds`. The Python `DGPBank` validates shapes and verifies that both
seed lists equal the sequences derived from their respective bases. This checks
the metadata contract; it does not regenerate the tensors to prove their origin.
The current CLI artifact fails reconstruction at the missing `source_name`.

The [Lorenz example](../examples/lorenz-shock/) checks two CLI artifacts and
records this boundary. Three retained test-named OU/Student-t artifacts from a
separate Rust full-bank path were also loaded and reconstructed by Python; their
exact producing build was not established. Neither observation establishes
reverse interchange, all presets, or same-seed byte identity across languages.

## Options for a future compatible contract

1. Describe the CLI's actual direct trajectory seed, preset and noise child
   seeds as such. This can improve provenance without changing the generated
   data, but does not satisfy the existing full-bank seed contract by itself.
2. Introduce an explicitly selected full-bank generation route, with its
   derived-seed behavior clearly visible to the caller. That would be a
   different data-generation choice, not a silent replacement for the CLI.
3. Design a representation that can truthfully carry direct trajectory seeds
   as well as base-and-child seeds. This requires a versioned interoperability
   design and validation in both producers and consumers.

No option here is implemented or scientifically approved by this audit. See
the [independent source audit](review-metadata-2026-11-04.md) for field and
source-line details. Python files inspected were `src/serie/api.py`,
`src/serie/data.py` and `src/serie/specs.py` in Fern main `da2a41ce`; their
SHA-256 values were respectively `38df04f42c026974c60be99acd29848e9d7bac9dfebb44cd72359fe4e0fb3605`,
`e730a42795bfbe9fc68878d8b355586f74e2b1bb6f2f4db53c774c34226635fd`, and
`ad55b27ada38f62de45112a077d6ceaab106fc67ae93604b85ac7f1bee5a2230`.
