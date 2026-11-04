"""Compare materialized f32 windows and report NumPy's distinct zero-copy view.

Run from the repository root with the Python environment and work directory
shown in the accompanying report. This is a kernel benchmark, not training.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "serie-rs/src/stage_ready.rs"
RUNNER = ROOT / "benchmarks/window/runner.rs"
N, D, SEQ, PRED, REPEATS = 36_000, 3, 96, 64, 15


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def numpy_view(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windows = np.lib.stride_tricks.sliding_window_view(data, SEQ + PRED, axis=0)
    windows = np.moveaxis(windows, -1, 1)
    return windows[:, :SEQ, :], windows[:, SEQ:, :]


def numpy_copy(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = numpy_view(data)
    return x.copy(order="C"), y.copy(order="C")


@numba.njit
def numba_copy(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = data.shape[0] - SEQ - PRED + 1
    x = np.empty((count, SEQ, D), dtype=np.float32)
    y = np.empty((count, PRED, D), dtype=np.float32)
    for i in range(count):
        for j in range(SEQ):
            for k in range(D):
                x[i, j, k] = data[i + j, k]
        for j in range(PRED):
            for k in range(D):
                y[i, j, k] = data[i + SEQ + j, k]
    return x, y


def times(operation, data: np.ndarray, repeat: int) -> list[int]:
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        result = operation(data)
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed)
        del result  # release outside timed interval
    return samples


def main() -> None:
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=False)
    order = sys.argv[2] if len(sys.argv) > 2 else "rust-first"
    assert order in ("rust-first", "rust-last")
    original = SOURCE.read_bytes()
    text = original.decode("utf-8")
    function = (
        "pub fn sliding_window("
        + text.split("pub fn sliding_window(", 1)[1].split(
            "\nuse rayon::prelude::*;", 1
        )[0]
    )
    structure = (
        "/// One split's windowed data."
        + text.split("/// One split's windowed data.", 1)[1].split(
            "\n// ready stage fn", 1
        )[0]
    )
    extracted = (structure + "\n" + function).encode()
    (work / "extracted.rs").write_bytes(extracted)
    environment = os.environ.copy()
    environment["SERIE_WINDOW_SOURCE"] = str(work / "extracted.rs")
    flags = [
        "rustc",
        "--edition=2024",
        "-C",
        "opt-level=3",
        str(RUNNER),
        "-o",
        str(work / "runner"),
    ]
    compile_start = time.perf_counter_ns()
    subprocess.run(flags, env=environment, check=True)
    rust_compile_ns = time.perf_counter_ns() - compile_start

    data = (np.arange(N * D, dtype=np.float32) / np.float32(32)).reshape(N, D)
    assert (
        data.dtype == np.float32
        and data.flags.c_contiguous
        and sys.byteorder == "little"
    )
    (work / "input.bin").write_bytes(data.tobytes())
    # The reference is explicitly materialized; checking happens outside timing.
    expected_x, expected_y = numpy_copy(data)
    view_x, view_y = numpy_view(data)
    assert np.shares_memory(view_x, data) and np.shares_memory(view_y, data)
    assert np.array_equal(view_x, expected_x) and np.array_equal(view_y, expected_y)
    cold_start = time.perf_counter_ns()
    first_x, first_y = numba_copy(data)
    numba_first_call_ns = time.perf_counter_ns() - cold_start
    assert first_x.dtype == np.float32 and first_y.dtype == np.float32
    assert first_x.flags.c_contiguous and first_y.flags.c_contiguous
    assert np.array_equal(first_x, expected_x) and np.array_equal(first_y, expected_y)
    assert sha(first_x.tobytes()) == sha(expected_x.tobytes())
    assert sha(first_y.tobytes()) == sha(expected_y.tobytes())
    del first_x, first_y

    rust_args = [
        str(work / "runner"),
        str(work / "input.bin"),
        str(work / "output.bin"),
        str(N),
        str(D),
        str(SEQ),
        str(PRED),
        str(REPEATS),
    ]

    def run_rust() -> tuple[list[int], int]:
        rust_start = time.perf_counter_ns()
        process = subprocess.run(rust_args, check=True, text=True, capture_output=True)
        process_ns = time.perf_counter_ns() - rust_start
        samples = [int(value) for value in process.stdout.splitlines()]
        assert len(samples) == REPEATS
        rust = np.fromfile(work / "output.bin", dtype="<f4")
        assert rust.size == expected_x.size + expected_y.size
        rx = rust[: expected_x.size].reshape(expected_x.shape)
        ry = rust[expected_x.size :].reshape(expected_y.shape)
        assert np.array_equal(rx, expected_x) and np.array_equal(ry, expected_y)
        assert sha(rx.tobytes()) == sha(expected_x.tobytes())
        assert sha(ry.tobytes()) == sha(expected_y.tobytes())
        return samples, process_ns

    if order == "rust-first":
        rust_ns, rust_process_ns = run_rust()
    # Warm each Python implementation without including startup in steady state.
    for fn in (numpy_copy, numba_copy, numpy_view):
        for _ in range(2):
            value = fn(data)
            del value
    if order == "rust-first":
        numpy_ns = times(numpy_copy, data, REPEATS)
        numba_ns = times(numba_copy, data, REPEATS)
    else:
        numba_ns = times(numba_copy, data, REPEATS)
        numpy_ns = times(numpy_copy, data, REPEATS)
        rust_ns, rust_process_ns = run_rust()
    view_ns = times(numpy_view, data, REPEATS)
    assert SOURCE.read_bytes() == original, "source changed during measurement"
    result = {
        "description": "f32 contiguous materialized sliding windows; NumPy view is a different output representation",
        "source_file_sha256": sha(original),
        "extracted_function_sha256": sha(function.encode()),
        "extracted_structure_sha256": sha(structure.encode()),
        "extracted_combined_sha256": sha(extracted),
        "runner_sha256": sha(RUNNER.read_bytes()),
        "benchmark_sha256": sha(Path(__file__).read_bytes()),
        "input_sha256": sha(data.tobytes()),
        "output_x_sha256": sha(expected_x.tobytes()),
        "output_y_sha256": sha(expected_y.tobytes()),
        "all_outputs_equal": True,
        "shape": {
            "input": list(data.shape),
            "x": list(expected_x.shape),
            "y": list(expected_y.shape),
        },
        "bytes": {"input": data.nbytes, "x": expected_x.nbytes, "y": expected_y.nbytes},
        "order": order,
        "repeats": REPEATS,
        "warmups_each": 2,
        "timing_ns": {
            "rust_extracted_copy": rust_ns,
            "numpy_copy": numpy_ns,
            "numba_copy": numba_ns,
            "numpy_view_different_representation": view_ns,
        },
        "median_ns": {
            name: statistics.median(values)
            for name, values in {
                "rust_extracted_copy": rust_ns,
                "numpy_copy": numpy_ns,
                "numba_copy": numba_ns,
                "numpy_view_different_representation": view_ns,
            }.items()
        },
        "rust_compile_ns": rust_compile_ns,
        "rust_process_ns_including_startup_warmup_validation_io": rust_process_ns,
        "numba_first_call_ns_including_compilation": numba_first_call_ns,
        "rustc_command": flags[:-1] + ["<work-dir>/runner"],
        "rustc_version_verbose": subprocess.check_output(
            ["rustc", "-vV"], text=True
        ).strip(),
        "python": sys.version,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "max_rss_kib_python": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "max_rss_kib_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_average": os.getloadavg(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
