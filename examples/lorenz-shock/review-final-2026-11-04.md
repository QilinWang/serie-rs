# Independent review: Lorenz shock case and Python interchange

Date: 2026-11-04. Scope: the candidate root README, example explanation and
`compare.py`, recorded `observed.json`, checked-in SVG, and the two retained
CLI output directories. Reviewed source behavior and performed read-only
Python/SVG checks; no simulation, build, model training or benchmark was run.

## Verdict

The checked example supports publication as a demonstration of this specific
Lorenz parameter shock and its CLI artifacts. It does not support claims of
forecast accuracy, a full cross-language `DGPBank` reconstruction from these
CLI files, or coverage of every intervention and preset. Those boundaries are
correctly stated in the candidate pages.

## Independent checks

- The CLI artifact file hashes match `observed.json`: control
  `eafe04892aff541aabbb3f56ca1d0332965c92c453eb663bd0a44f854bce9ce7`,
  shock `2b6db6abb1f9ff23cd88af5c75aefee9356036a9b39a41fbf59756e4efcf1c04`.
  Both manifests report the same Lorenz system, seed 42, noise seed 1000,
  sigma 0.1 and eight draws; the shock labels are `none` and `param`.
- Current Python `serie.api.load_dgp_bank_safetensors` reads three arrays from
  each: clean `[1, 36000, 3]`, noise `[8, 36000, 3]`, noisy `[36000, 3]`, all
  float32. `dgp_bank_from_tensors` refuses each with missing `source_name`.
  The observed header keys are exactly `feature_names`, `noise_seed`,
  `schema`, `seed`, and `system`, matching the explanation. Shock selection is
  in the manifest, not the tensor header.
- The clean prefix before index 12,599 and all noise draw bytes match across
  runs; first float32 clean divergence is index 12,599. In each output the
  full saved noisy array matches `clean[0] + float32(0.1) * noise[0]`.
- The plot uses channel 0, sample indices `[12300, 14100)` at stride two.
  Independently reconstructing the 900-point control and shock SVG path
  strings from the saved arrays produced exact matches. The event marker's
  computed x coordinate is `266.61`, matching the SVG. The SVG labels these
  as clean generated trajectories rather than forecasts.
- The two existing Lorenz presets, event step and f64 integration followed by
  CLI f32 export agree with source. The OU Student-t legacy artifact discussion
  correctly limits provenance to the retained test-named files and does not
  assert same-seed Python/Rust generator identity.

## Optional hardening

`compare.py` explicitly asserts the clean dtype but not the noise and noisy
array dtypes. Both real examples were independently observed to be float32;
adding dtype assertions for those arrays would make the reproducer refuse a
future artifact with unexpected element types. This is not a finding against
the actual recorded results.

## Resolution

The optional dtype and shape assertions were added; validation was rerun and
the recorded results remained identical. The retained OU file hashes are
recorded separately with the source-binding limitation.
