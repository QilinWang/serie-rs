# Sliding-window microbenchmark

**Status: exploratory; high system load and large timing variation.** In the
run with lower Rust variation, the materialized-window medians were **18.47 ms**
for the extracted Rust function and **20.40 ms** for Numba. That is not a
reliable speed ranking: in the other run, Rust's samples ranged from 21.81 ms
to 2,293.28 ms. The current Fern training path can use zero-copy views, so
these numbers are not an end-to-end comparison with that pipeline.

## What was measured

The actual `sliding_window` function and `WindowedSplit` type were extracted
verbatim at runtime from `serie-rs/src/stage_ready.rs` at commit `86611202`.
The small harness was compiled separately with `rustc -C opt-level=3`. This
is **not a Cargo release build of the library**; its compilation and inlining
context can differ. NumPy and Numba implementations construct the same two
C-contiguous arrays. The input is synthetic, not a Lorenz trajectory.

- Input: `float32 [36000, 3]`, deterministic unique values.
- Output X: `[35841, 96, 3]`; output Y: `[35841, 64, 3]`.
- Both arrays are materialized: 68,814,720 bytes total.
- Input generation, output checks, process startup and output destruction are
  outside the timed interval; allocation and copying are inside it.
- Numba's first call includes JIT compilation and is recorded separately.
- A separate validation invocation of Rust and Numba checked X and Y against
  NumPy for numerical and byte equality; the timed outputs were not checked
  individually. The JSON records source, input and output hashes.
- Two warm-ups and 15 timed repetitions per implementation, in opposite
  implementation orders. The benchmark uses one pinned CPU and limits library
  threading to one. Times are wall-clock durations.

## Recorded results (2026-11-04)

| Implementation                                         | Rust first: median (range), ms | Rust last: median (range), ms |
| ------------------------------------------------------ | -----------------------------: | ----------------------------: |
| Rust source-extracted function                         |         76.93 (21.81–2,293.28) |           18.47 (16.64–26.69) |
| NumPy materialized copy                                |     1,337.50 (634.95–5,793.75) |          49.44 (17.84–880.71) |
| Numba materialized copy                                |           18.88 (16.06–410.51) |           20.40 (16.71–28.98) |
| NumPy zero-copy views, different output representation |            0.047 (0.010–0.358) |           0.036 (0.014–1.554) |

The view measurement only creates views; it does **not** read or materialize
all window values. It answers a different question from the three copy timings.
The active Fern training implementation uses strided Torch views and consumes
windows later; the complete cost depends on how those windows are used.

Host: WSL, Intel Core i7-14700HX (10 logical CPUs visible), pinned to logical
CPU 4; Python 3.12.12, NumPy 2.2.6, Numba 0.63.1 and Rust 1.97.1. System load
averages recorded at the end of the runs were about 27.1 and 20.1 respectively
(1-minute averages), much higher than the 10 visible logical CPUs. CPU
contention and different run ordering may both influence the spread. The first
Numba calls took 17.59 s and 16.89 s including JIT and copying. The measured
Rust compilation took 31.42 s and 6.02 s. Neither is part of the steady-state
medians. The Python process peak RSS was about 381 MiB in each run; that is not
a measurement of total simultaneous process memory or allocation traffic.

The [first](run-a.json) and [second](run-b.json) records contain every sample,
versions, hashes, compiler flags and measured resource metadata. Remote hosts
were checked but became unreachable during the attempt. A low-contention run on
a specified machine is needed before using any speedup as a headline.

## Reproduce

From the repository root, using a Python environment with NumPy and Numba and
an installed Rust compiler. Each output path must not already exist. Choose an
available CPU for your machine; `4` is the CPU used in the recorded run.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  taskset -c 4 python -B benchmarks/window/benchmark.py /tmp/window-run-a rust-first > run-a.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  taskset -c 4 python -B benchmarks/window/benchmark.py /tmp/window-run-b rust-last > run-b.json
```

The work directories contain generated source, compiled binary and binary
input/output for inspection. They can be removed using the operator's normal
cleanup process after reviewing the records.
