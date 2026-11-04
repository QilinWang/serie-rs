# Independent review: CLI and full-bank seed semantics

Date: 2026-11-04. Scope: `docs/cli-data-bank.md` in the candidate standalone
workspace, checked read-only against the Rust CLI/generator and the Python
sources. No tests, examples, experiments or builds were run.

**Verdict: no blocking issue.** The explanation accurately distinguishes the
CLI's direct trajectory seed from the full-bank spawned child, the noise-base
spawning used by both, and truthful tensor loading from full-bank reconstruction.
It does not assert equal same-seed generated arrays across Rust and NumPy.

- The cited `803958421` ordinal-0 child of base 42 matches the shared
  SplitMix64-style derivation in Rust `draw_protocol.rs:41-57` and Python
  `src/serie/data.py:44-70`. The CLI assigns `args.seed` directly in
  `main.rs:180-183`; the full-bank path assigns a derived child to the spec
  in `dgp_bank.rs:772-783`. Replacing the former with the latter can change
  stochastic trajectories; the document does not propose doing so silently.
- Rust `unit_noise_draws` seeds `rand_pcg::Pcg64`, samples f64
  `StandardNormal`, then casts to f32 (`dgp_bank.rs:543-565`). Python uses
  `np.random.default_rng` and its f32 Gaussian sampler
  (`src/serie/api.py:449-451,501-511`). Equal spawned seed identities do not
  establish identical streams or noise arrays.
- Python's tensor loader and DGPBank reconstruction checks are described
  accurately (`src/serie/api.py:932-976,1004-1030`,
  `src/serie/specs.py:789-829`). The document correctly notes that a
  self-consistent metadata seed recipe does not prove tensor origin, and that
  intervention identity is carried by the CLI manifest, not tensor header.
- All three recorded Python source hashes match both the local files and
  `jj file show -r da2a41ce` for `src/serie/api.py`, `data.py` and `specs.py`.
- The three future approaches are framed as options, not implemented changes;
  the note explicitly preserves existing seeds, noise, precision and values and
  does not claim scientific approval.
- Relative links to the Lorenz case and the source audit resolve in the
  candidate workspace.

The document does not establish reverse interoperability, array equivalence
across engines, compatibility for all presets, or approval for a new numerical
or seed policy. Its coverage statement appropriately leaves those open.
