# Independent review: Rust, Cargo and dependency upgrade

Date: 2026-11-04. Scope: the candidate source and dependency diff, the upgrade
inventory and report, the saved array receipt, four f64 fixture outputs, the
actual `.crate` and the existing build/test logs. Review actions were read-only;
no build, test, training, benchmark or data generation was run in this review.

## Verdict

**No blocking issue for the stated upgrade and validation scope.** The source
patch does not alter the trajectory seed logic, numerical precision, model or
noise equations. The resulting data identity is supported only for the stated
six exported f32 cases and three existing f64 fixture families, not for every
system, input, feature, machine, or scientific experiment. The report makes
those limits explicit. The prior two review observations about a misplaced
Rust doc comment and a rewritten historical manifest comment were corrected.

## Source and dependency checks

- The 15 direct dependencies in the manifest and current Cargo.lock agree with
  the candidate's registry inventory. Major-series changes include Arrow
  56→60, nalgebra 0.34→0.35, sha2 0.10→0.11 and tabled 0.20→0.22. RNG-related
  updates include rand 0.10.0→0.10.3 and rand_pcg 0.10.1→0.10.2; the lock
  contains `chacha20 0.10.2` instead of the yanked 0.10.0. The saved later
  `cargo update --locked --dry-run` selected zero further compatible updates;
  this does not claim all transitive versions are unconditionally latest.
- `dgp_bank.rs` changes the digest-to-hex representation required by sha2
  0.11: each digest byte is written as two lowercase hex digits, and the
  existing first-16-character fingerprint remains unchanged. The added test
  exercises a zero prefix and byte order. Available known-data checks passed,
  including the HuGaDB prefix/full pin; ETTm1 and ETTm2 data were absent and
  their subbranches were skipped. The old `read_known_ipc` documentation is
  attached to the same function again; the original nalgebra historical
  comment is preserved with an added explanation of the upgrade.
- `SOURCE_PROVENANCE.json` keeps the historical lock description and lock
  digest, records the previous `dgp_bank.rs` candidate hash, and gives current
  SHA-256 values for Cargo.lock, the crate manifest and `dgp_bank.rs`; all
  three match the candidate files. The previous package validation remains a
  historical record rather than being rewritten as an upgraded test.
- Rust 1.99.1 is installed alongside stable 1.97.1 and selected by this
  workspace's `rust-toolchain.toml`; the global stable default was not switched.
  The install log separately records rustup itself auto-updating 1.29.0→1.29.3.
  The unpacked package was tested with `cargo +1.99.1`, since it does not
  inherit the workspace's toolchain pin.

## Numerical and artifact evidence

- The saved f32 receipt contains six cases, 15 tensors and 2,785,025 scalar
  values per build. For every tensor, all three stage hashes match and both
  recorded comparisons have zero changed values: Lorenz control, Lorenz
  parameter shock, and OU CLI examples; OU full-bank Student-t3, Laplace and
  contaminated-noise examples. Existing test inputs/seeds and their respective
  output shapes are documented. The three CLI manifest objects and parsed
  metadata, and the three non-Gaussian banks' parsed metadata, were independently
  compared and matched by case. File-header byte order was not mistaken for a
  tensor difference. These samples do not prove all random distributions or
  future seeds produce the same values.
- Four full f64 fixture JSON files (old toolchain/old lock, new toolchain/old
  lock, new toolchain/new lock, and unpacked updated crate) have SHA-256
  `758bcef654bad419bd1341b82201c676796156a623222d1e7e98d95b8882e467`.
  The retained reference matches that hash. The example takes existing linear
  canary, controlled Lorenz and pendulum test inputs; it records f64 `to_bits`,
  matrix dimensions/column-major order and action-Jacobian shapes. The JSON was
  compared as integers, without JavaScript number rounding. The example covers
  these existing fixtures, not all nalgebra operations or internal f64 states
  of every CLI trajectory.
- Cargo successfully packaged and verified a 27-file crate whose SHA-256 is
  `236e349a466765392a95205a3caa4b83fd3e929b8b04f60588210f8d73f126d1`.
  The extracted crate's source, original manifest and f64 example match the
  corresponding candidate bytes. Saved extracted-crate logs report 121/121
  default-feature and 213/213 all-feature library tests, with the two absent
  ETT asset branches noted above. CLI smoke listed nine systems, and the
  extracted f64 example produced the reference hash. No crates.io release was
  performed or implied.

The local validation and the observational registry inventory support the
specific conclusions in `docs/upgrade-2026-11-04.md`; future availability and
other downstream dependency resolutions remain outside this review.
