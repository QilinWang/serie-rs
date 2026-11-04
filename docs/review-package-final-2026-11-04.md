# Independent review: standalone Cargo package

Date: 2026-11-04. Scope: candidate packaging changes, provenance, package
validation report, generated `.crate`, extracted package and saved test logs.
I inspected the files and outputs without running a build, test, benchmark,
training or data generation. The separate metadata audit is reviewed in its
own record.

## Verdict

No blocking issue found for the stated local package-validation scope. The
changes preserve the existing scientific implementation; the package has not
been published and the checks do not cover every feature or external dataset.

## Evidence

- `controlled_linear.rs` changes only the two test-only `include_str!` paths.
  The two historical declarations moved into the crate retain SHA-256
  `ccbbb5d43de63c6d1d03a5aca6cdeaf683c79f0086055f8de1dd1c51eba283c8` and
  `26585cb14c1728d91c433a9622c305c353703c321428a769c7db8ca42b29ffd8`.
  The crate license and workspace license have identical SHA-256
  `c5be3b4a3520fb884af1ac30393df5ed0bbc38afa730199a57be17847b447389`.
  The crate README uses repository links instead of a missing parent README.
- The actual `.crate` archive has SHA-256
  `7dd8776bb0674cb82d69da502ed47ed70f4266af39aa9a1301005afd41fc6782` and
  contains both declarations, LICENSE and README. Extracted source and the
  declarations/license match the candidate. The provenance record preserves
  historic source hashes and records the new test-only include-path edits.
- The Cargo package log records 26 files and successful verification. The
  separately extracted crate's default-feature library test log reports
  121 passed, including both declaration tests. With `cli` enabled, the
  focused paired-noise test reports one passed and 211 filtered; the CLI
  `list-systems` smoke log lists nine systems. These numbers do not prove the
  remaining optional-feature or external-data branches ran.
- The locked package build warned that `chacha20 0.10.0` was yanked; the
  historical lock links it from `rand 0.10.0`. The separate no-lock offline
  resolution log and generated temporary lock select `rand 0.10.3`,
  `rand_core 0.10.1`, and `chacha20 0.10.2`, versus 0.10.0 in the historical
  lock. This is local dependency resolution only: the fresh lock was not used
  to build or generate data and proves no numerical equivalence. The existing
  `proc-macro-error2 2.0.1` future-Rust warning is separate. No dependencies or
  scientific settings were changed.

The package validation page and README accurately limit the result to a local
build/test rather than a crates.io release, and state the dependency caveats.
