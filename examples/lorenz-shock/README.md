# A real parameter shock

**Same system. Same eight noise draws. One intervention.** The image on the
project homepage is drawn from two generated Lorenz trajectories, not from a
forecasting model. The control and shocked runs have the same `float32` clean
values before step **12,599** and the same noise-draw bytes throughout. Their
clean trajectories first differ at step **12,599** in this run.

The source's existing `lorenz` and `lorenz_param_shock` presets use RK4,
`dt = 0.01`, 35,999 integration steps and initial state `(1, 0.98, 1.1)`.
At 35% of the steps, the parameter-shock preset replaces `(sigma, rho, beta)`
`(10, 28, 8/3)` with `(10.1, 28.1, 8.1/3)`. The CLI defaults used here are
trajectory seed `42`, noise seed `1000` and eight Gaussian draws; both commands
use observation noise scale `0.1`. These are the existing demonstration presets,
not a newly chosen research protocol. Integration uses `float64`; the CLI
converts the saved arrays to `float32`.

In each artifact, `clean` has shape `[1, 36000, 3]`, `noise` has shape
`[8, 36000, 3]`, and `noisy_sigma_0.1` has shape `[36000, 3]`. The check
compares the whole pre-shock clean prefix and all noise bytes, verifies a
post-shock difference, and checks the saved noisy values against
`clean[0] + float32(0.1) * noise[0]`. The image plots channel 0, steps
`[12300, 14100)`, with every second timestep shown; it displays the clean
trajectories. The selected plot range does not alter generated data.

## Reproduce

From a checkout of the repository, with Rust and a Python environment that has
NumPy and `safetensors` installed:

```sh
cargo run --locked -p serie-rs --features cli -- generate lorenz --shock none --sigma 0.1 --out /tmp/serie-lorenz-control
cargo run --locked -p serie-rs --features cli -- generate lorenz --shock param --sigma 0.1 --out /tmp/serie-lorenz-shock
python examples/lorenz-shock/compare.py /tmp/serie-lorenz-control /tmp/serie-lorenz-shock lorenz-shock.svg
```

The final command prints a JSON receipt and writes the plot. These directories
are demonstration outputs; use fresh directories to avoid mixing runs. The
checked-in SVG was formatted after generation, so formatting bytes may differ
from the script output. The [recorded receipt](observed.json) contains the
exact run's manifest summaries and hashes. Whole-file hashes can vary when
metadata keys are serialized in a different order, even when tensor bytes are
the same; compare the tensor hashes separately.

## Python interoperability: what was checked

The Python `serie.api.load_dgp_bank_safetensors` successfully loaded all three
arrays from each of these two CLI artifacts. However, the CLI artifact header
contains only `schema`, `feature_names`, `system`, `seed`, and `noise_seed`.
The richer `serie.api.dgp_bank_from_tensors` refuses it because required bank
metadata, starting with `source_name`, is absent. The intervention selection is
in the accompanying `manifest.json`, not the tensor header. Do not treat the
CLI artifact as a fully reconstructible Python `DGPBank`.

A separate observation used three retained, test-named OU + Student-t Rust
artifacts from an earlier run of `generate_dgp_bank`. Python loaded each and
rebuilt a `DGPBank` with clean shape `[2, 25001, 1]` and noise shape
`[3, 25001, 1]`. Their parsed headers and payload bytes agreed, but their
whole-file hashes differed as metadata key order differed. The test name,
metadata and earlier passing test log support their likely origin; no receipt
binds each retained file to a particular build. [Retained-file hashes](retained-ou-2026-11-04.json). This is not evidence of
same-seed identity between Python and Rust generators, reverse interchange, or
support for all presets.

## Run provenance

The Lorenz example was generated on 2026-11-04 from project commit `a92733348bfc`
using Rust 1.97.1, an offline locked Cargo development build and one build job.
The binary SHA-256 was
`db04b8a13ab78bb553f0a425e8047c49abf8da107d64ea0236865f847d238269`.
Inspection used Python 3.15.2, NumPy 2.5.2 and safetensors 0.8.0. The Python
SERIE loader was the current `/src/serie/api.py` from the local Fern checkout;
its SHA-256 was `38df04f42c026974c60be99acd29848e9d7bac9dfebb44cd72359fe4e0fb3605`.
No model was trained or evaluated. These checks cover these two CLI outputs and
the specifically described retained OU files, not every system, shock or
noise process.
