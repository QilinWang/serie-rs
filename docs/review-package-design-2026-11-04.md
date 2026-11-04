# Cargo package layout: independent design review

Date: 2026-11-04. Scope: static inspection of public SERIE-RS workspace source,
Cargo metadata, test dependencies and source-provenance record. No build, test,
benchmark, training or scientific computation was run for this review.

The two `include_str!` declarations in `controlled_linear.rs` are the only
compile-time paths escaping the crate (`../../analysis_declarations/...`); both
occur in unit tests. Move the two unchanged declaration byte streams into
`serie-rs/analysis_declarations/` and change only these references to
`../analysis_declarations/...`. Preserve and verify the original SHA-256s.
There are no other references to the workspace-root declaration directory.

A standalone crate should contain the MIT LICENSE and a useful README whose
links do not refer to a parent not in the package. Copying the root license
byte-for-byte and replacing the crate README's relative workspace link with an
absolute repository URL preserves the existing license and disclosure; a package
listing by itself is not proof the packaged crate builds or tests.

`SOURCE_PROVENANCE.json` records archived source hashes, not the live tree's
current hashes. Preserve those original observations, document the relocated
declarations, and add an explicit test-only `controlled_linear.rs` entry to
`modified_files` with archived/current SHA-256s and the two path edits. Retain
the previously documented privacy edit and other archival caveats.

Verify an actual `.crate` using the checked-in lockfile, then inspect and test
its extracted contents in an isolated directory which has no source workspace
as parent. Run the default library tests and, subject to resources, the CLI
feature build or focused tests; document the exact features, toolchain and
counts. Tests for known data depend on a user-provisioned `SERIE_KNOWN_ROOT` or
cache and can return early on absent assets; a passing test count does not prove
those branches ran. External dataset licensing remains separate.
