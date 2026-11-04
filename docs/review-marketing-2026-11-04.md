# Review: SERIE-RS marketing README

Scope: reviewed the candidate README, conceptual SVG, and relocated recovery and validation document against the existing public source and prior README. I did not run builds, tests, commands, or benchmarks.

## Findings

1. **Clarify the parallel-windowing claim.** README line 17 says “optional Rayon-parallel windowing,” which can suggest the documented staged pipeline selects parallel execution through a configuration option. `stage_ready.rs:76` exposes `sliding_window_parallel` as a separate public function; `ready()` itself calls serial `sliding_window` for all three splits (`stage_ready.rs:338-341`). Recommend “preallocated buffers and a separate Rayon-parallel windowing function” (or equivalent). Do not imply a speedup without measurements.
2. **Make the output command directly usable.** README line 25 says “Add `--out`” but clap defines an output directory value (`main.rs:97-100`). Say “Add `--out <directory>` to write the bank and manifest.” The default invocation really prints a JSON summary; writing is conditional on that option (`main.rs:217-257`).
3. **Locate the paired-noise evidence more precisely.** README line 13 labels a link “See the test” but links to an entire source file. Link the `common_random_numbers_two_sigmas_differ_only_in_amplitude` test anchor instead, or say “See the implementation and test.” This is a usability improvement, not a claim failure.
4. **Feature gates deserve one concise pointer for Rust users.** The new README advertises stages but does not mention that `pipeline` gates noise, partition and ready and `known-data` gates load (`lib.rs:14-32`; `Cargo.toml:51-60`). Add a short link to the feature table or one concise sentence. Existing validation doc currently says only that modules have gates. The `cli` example correctly enables the complete set. No crate publication is implied.

## Claims checked

- The nine named CLI systems are declared in `main.rs:31-41`. Parameter, state, and switch presets exist only for selected system/shock combinations (`main.rs:435-461`); the selected-systems qualifier is necessary and present. The underlying shock variants are parameter replacement, additive state kick and same-system trajectory splice (`stage_synthetic.rs:1579-1599`).
- Stage order is correctly shown as synthetic generation or load/clean, then noise, partition and ready windows (`stage_noise.rs:66-80,111-126`; `stage_partition.rs:330`; `stage_ready.rs:311`).
- Fixed-noise scaling is supported by the injected Gaussian draw path (`stage_noise.rs:130-161`) and the paired test recorded by the previous README. It does not establish a paired comparison for all shock conditions, nor a forecasting result, and the candidate does not claim those.
- The recorded 121/121 figure, Lorenz 36,000×3 and eight default draws are inherited from the preceding README's recorded validation, not re-executed in this review. The CLI source independently shows eight Gaussian draws by default. The detail document keeps the date, toolchain, limits and warning.
- The SVG explicitly marks its curve as conceptual, and its staged branches match the source. Python demo, retired snapshot status, collaboration, provenance and external-data caveats remain disclosed. The revised prose is short, visually scannable, and fits the requested emoji-led marketing style.

Verdict: **Accept after findings 1–2 are corrected.** Findings 3–4 are recommended usability improvements. No unmeasured Rust-versus-Python speed or memory claim should be introduced.

## Resolution

All four findings were addressed in the README before publication. The parallel
function is described separately, the output directory argument is explicit,
the test link points to the test, and the feature-gate declaration is linked.
