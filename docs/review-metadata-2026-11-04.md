# Independent audit: CLI versus Python DGP bank

Date: 2026-11-04. Scope: read-only inspection of the current standalone Rust
source and the local Fern Python source. No build, training, data generation,
benchmark, or repository modification was performed in this audit.

## Verdict

The CLI tensor artifact and the full DGPBank generator use the same named
safetensors schema but represent different provenance contracts. Python can
load the CLI tensors, but its DGPBank constructor cannot truthfully reconstruct
a full bank from the current metadata. Adding the full-bank seed fields from
the CLI's `--seed` as if it were a trajectory seed base would misstate which
seed generated a stochastic trajectory. Changing the CLI to spawn a child seed
would alter stochastic trajectories and requires the owner's scientific
approval, not a silent compatibility fix. No existing numbers need change to
record the direct seed and actual noise seed derivation accurately.

## Exact data and seed flow

| Aspect           | CLI `generate`                                                                                                                                                                                                                                                  | `generate_dgp_bank` / Python `build_dgp_bank`                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trajectory seed  | `main.rs:180-183`: preset selected by system/shock, `spec.seed = args.seed`, generate one clean trajectory.                                                                                                                                                     | `dgp_bank.rs:772-783`: spawn each child seed from `trajectory_seed_base`, put that child in the preset, generate each trajectory. Python `src/serie/api.py:630-635` likewise calls `spawn_draw_seeds(seed, n_trajectory_draws)` then `load_data(..., seed=s)`.                                                                                                                                                                                            |
| Noise            | `main.rs:194-202`: Gaussian, `args.noise_seed` as the base and `args.n_noise_draws`; `dgp_bank.rs:543-564` derives child seeds via `spawn_draw_seeds`, then one Rust `Pcg64::seed_from_u64` stream per child, samples f64 `StandardNormal` and converts to f32. | Rust `dgp_bank.rs:804-809` uses the same `unit_noise_draws` function plus the chosen temporal process; Python `api.py:501-511,655-664` spawns noise child seeds and uses `np.random.default_rng(seed)`, with Gaussian sampled directly as f32 (`api.py:449-451`). Rust and NumPy have different RNG stacks/sampling routes; the Rust source explicitly warns that same child seeds do not make the generated banks bit-identical (`dgp_bank.rs:691-697`). |
| Child derivation | Rust `draw_protocol.rs:41-57`: deterministic SplitMix64-style u64 wrapping then low 32 bits; the Python implementation has matching operations in `src/serie/data.py:44-70`.                                                                                    | For example the existing base 42 has ordinal-0 child `803958421`, not 42; base 1000 has first children `1305264456`, `1264479158`, `1560021823`. These are mathematical identity examples, not proposed seed settings.                                                                                                                                                                                                                                    |
| Shapes and noise | CLI `main.rs:187-212`: clean `[1,T,D]`, noise `[N,T,D]`; when sigma positive, materializes one `[T,D]` noisy tensor as `clean[0] + f32(sigma) * noise[0]`.                                                                                                      | Rust bank `dgp_bank.rs:774-809` may contain multiple clean draws and noise draws. Both representations store the arrays; a named `noisy` tensor does not alter the stored frozen clean/noise draws.                                                                                                                                                                                                                                                       |
| Shock identity   | `main.rs:215-247`: shock selection occurs in `spec_for` and is reported in manifest summary but not safetensors extra metadata. `system=lorenz` alone does not identify `none` vs `param` vs `state` vs `switch`.                                               | Source identity and, when applicable, a meaningful intervention/source declaration must be retained. Python `DGPBank` has optional `source_declaration` (`specs.py:767-778`, `api.py:1027-1030`); do not assert an existing full bank automatically reconstructs a CLI shock from `system`.                                                                                                                                                               |

## Header fields and consumer checks

- CLI writer adds `system`, `seed`, `noise_seed` (`main.rs:227-241`); the
  shared serializer adds `schema` and `feature_names` (`dgp_bank.rs:186-193`).
  CLI manifest separately carries `system`, `shock`, `seed`, `noise_seed`,
  `sigma`, timestep/channel counts and number of draws (`main.rs:215-247`).
- Full Rust bank metadata includes `source_name`, `noise_kind`,
  `is_stochastic`, `draw_design`, `data_engine`, `axis_order`, `shape`,
  `trajectory_seed_base`, `noise_seed_base`, `trajectory_draw_seeds`, and
  `noise_draw_seeds` (`dgp_bank.rs:1096-1147`). Temporal AR1 fields are
  conditional. Python exporter includes the corresponding fields and can add
  source declaration (`src/serie/api.py:893-925`).
- Python `load_dgp_bank_safetensors` reads arrays, requires names `clean` and
  `noise`, and parses/checks feature names (`src/serie/api.py:932-976`). It does
  not establish full bank provenance merely by loading them.
- `dgp_bank_from_tensors` explicitly requires source name, noise kind,
  stochastic flag, the two seed bases and their two seed lists
  (`src/serie/api.py:1004-1026`). It casts clean and noise to f32 as existing
  behavior; changing that precision is a scientific change requiring approval.
  `DGPBank.__post_init__` checks rank, shared `[T,D]`, list lengths, and that
  **both** stored child-seed lists equal `spawn_draw_seeds(base,n)`
  (`src/serie/specs.py:789-829`). It checks the declared recipe; it does not
  recompute the trajectories or noises to prove they used that recipe.

## Safe boundary and open decisions

1. A metadata-only description of what CLI really did can state direct
   trajectory seed, system and shock preset, Gaussian innovation/noise base,
   derived noise child-seed list, actual array shape and dtype, and output
   engine. It must preserve existing generated numbers, seed interpretation,
   precision and command defaults; validate against actual outputs. These
   truthful fields alone do not satisfy the current Python full-bank contract.
2. Do not fill `trajectory_seed_base=args.seed` plus a synthetic spawned
   `trajectory_draw_seeds` list as if it generated the CLI trajectory. Even
   if a deterministic Lorenz trajectory happens to ignore seed, this would
   obscure actual provenance and generalizes incorrectly to OU and other
   stochastic systems (`stage_synthetic.rs:438-449,881-886`).
3. Do not silently modify CLI to use the bank's derived trajectory child, nor
   alter its RNG, normal sampling precision, number of draws, Gaussian choice,
   time step or intervention. Those decisions can change scientific data.
   A fully compatible route needs an explicitly agreed semantic design,
   for example a separate conversion/representation that honestly carries
   direct seeds or a separately selected bank-generation entry; each has
   compatibility implications and needs explicit scientific review.
4. `dgp_bank.rs:186-193` collects metadata in a HashMap. The previous
   independently checked three retained OU files had equal parsed headers and
   payload bytes but different serialized metadata order and whole-file hashes.
   Compare tensor bytes and parsed metadata separately; do not present whole-file
   hash equality as a general reproducibility contract.
5. The prior Lorenz and retained OU observations cover specific paths and
   artifacts. They do not prove reverse exchange, all systems or noise kinds,
   exact same-seed numerical equality between Rust and Python, or restored
   information absent from the CLI artifact.
