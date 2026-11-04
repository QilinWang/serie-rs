"""Inspect two SERIE-RS CLI banks and draw their actual clean trajectories.

Usage: python compare.py CONTROL_DIR SHOCK_DIR OUTPUT_SVG
The existing Lorenz presets and CLI defaults are used unchanged. This script
checks saved float32 tensors; it does not run a forecast or train a model.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

EVENT_STEP = 12_599  # trunc(0.35 * 35_999), per the archived Rust preset
START, STOP = 12_300, 14_100  # Display-only crop; no generated values change.


def read(directory: Path) -> tuple[dict, dict[str, np.ndarray], str]:
    manifest = json.loads((directory / "manifest.json").read_text())
    data_path = directory / "dgp_bank.safetensors"
    return (
        manifest["summary"],
        load_file(str(data_path)),
        hashlib.sha256(data_path.read_bytes()).hexdigest(),
    )


def path(values: np.ndarray, lo: float, hi: float) -> str:
    width, height, left, top = 1050.0, 225.0, 92.0, 116.0
    return " ".join(
        ("M" if i == 0 else "L")
        + f"{left + width * i / (len(values) - 1):.2f},{top + height * (hi - float(value)) / (hi - lo):.2f}"
        for i, value in enumerate(values)
    )


def main() -> None:
    control_dir, shock_dir, output = map(Path, sys.argv[1:4])
    control, a, control_hash = read(control_dir)
    shock, b, shock_hash = read(shock_dir)
    for name in (
        "system",
        "seed",
        "noise_seed",
        "sigma",
        "n_noise_draws",
        "timesteps",
        "channels",
    ):
        assert control[name] == shock[name], name
    assert (control["system"], control["shock"], shock["shock"]) == (
        "lorenz",
        "none",
        "param",
    )
    assert (
        control["sigma"] == 0.1
        and control["timesteps"] == 36_000
        and control["channels"] == 3
    )
    assert a["clean"].shape == b["clean"].shape == (1, 36_000, 3)
    assert a["noise"].shape == b["noise"].shape == (8, 36_000, 3)
    assert a["clean"].dtype == b["clean"].dtype == np.float32
    assert a["noise"].dtype == b["noise"].dtype == np.float32
    assert a["noisy_sigma_0.1"].dtype == b["noisy_sigma_0.1"].dtype == np.float32
    assert a["noisy_sigma_0.1"].shape == b["noisy_sigma_0.1"].shape == (36_000, 3)
    assert a["noise"].tobytes() == b["noise"].tobytes(), "noise draws differ"
    assert (
        a["clean"][:, :EVENT_STEP].tobytes() == b["clean"][:, :EVENT_STEP].tobytes()
    ), "clean prefix differs"
    assert np.any(a["clean"][:, EVENT_STEP:] != b["clean"][:, EVENT_STEP:]), (
        "no divergence after intervention"
    )
    for arrays in (a, b):
        expected = arrays["clean"][0] + np.float32(0.1) * arrays["noise"][0]
        assert expected.tobytes() == arrays["noisy_sigma_0.1"].tobytes(), (
            "saved noisy values differ"
        )

    x_control = a["clean"][0, START:STOP:2, 0]
    x_shock = b["clean"][0, START:STOP:2, 0]
    lo = float(min(x_control.min(), x_shock.min())) - 1.0
    hi = float(max(x_control.max(), x_shock.max())) + 1.0
    marker = 92 + 1050 * (EVENT_STEP - START) / (len(x_control) - 1) / 2
    first_diff = int(np.flatnonzero(np.any(a["clean"][0] != b["clean"][0], axis=1))[0])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="435" viewBox="0 0 1200 435" role="img" aria-labelledby="title desc">
<title id="title">A parameter shock applied to a real generated Lorenz trajectory</title>
<desc id="desc">Two clean x-coordinate trajectories share the same preset until a parameter replacement. These are generated data, not forecasts.</desc>
<rect width="1200" height="435" rx="20" fill="#0e1927"/>
<g font-family="Inter,Segoe UI,Arial,sans-serif">
<text x="58" y="48" fill="#eaf6f1" font-size="23" font-weight="700">SAME SYSTEM. SAME NOISE. ONE SHOCK.</text>
<text x="58" y="78" fill="#9bb4c0" font-size="15">Real SERIE-RS output · Lorenz x coordinate · clean trajectories, not forecasts</text>
<text x="50" y="120" fill="#9bb4c0" font-size="13">{hi:.1f}</text><text x="50" y="341" fill="#9bb4c0" font-size="13">{lo:.1f}</text>
<path d="M92 116V341H1142" stroke="#72909c" stroke-opacity=".65" fill="none"/>
<path d="M92 172H1142M92 228H1142M92 284H1142" stroke="#607e87" stroke-opacity=".25"/>
<path d="M{marker:.2f} 105V341" stroke="#ffcf7a" stroke-width="2" stroke-dasharray="7 6"/>
<text x="{marker + 8:.2f}" y="120" fill="#ffcf7a" font-size="14" font-weight="700">PARAMETER SHOCK · STEP {EVENT_STEP:,}</text>
<path d="{path(x_control, lo, hi)}" fill="none" stroke="#809ab5" stroke-width="2.1"/>
<path d="{path(x_shock, lo, hi)}" fill="none" stroke="#6debc2" stroke-width="2.1"/>
<text x="92" y="377" fill="#9bb4c0" font-size="14">step {START:,}</text><text x="1065" y="377" fill="#9bb4c0" font-size="14">step {STOP - 2:,}</text>
<line x1="92" y1="405" x2="120" y2="405" stroke="#809ab5" stroke-width="3"/><text x="128" y="410" fill="#c4d1dc" font-size="15">control</text>
<line x1="227" y1="405" x2="255" y2="405" stroke="#6debc2" stroke-width="3"/><text x="263" y="410" fill="#c4d1dc" font-size="15">parameter shock</text>
</g></svg>\n'''
    output.write_text(svg)
    print(
        json.dumps(
            {
                "control_file_sha256": control_hash,
                "shock_file_sha256": shock_hash,
                "control_clean_sha256": hashlib.sha256(
                    a["clean"].tobytes()
                ).hexdigest(),
                "shock_clean_sha256": hashlib.sha256(b["clean"].tobytes()).hexdigest(),
                "shared_noise_sha256": hashlib.sha256(a["noise"].tobytes()).hexdigest(),
                "first_different_timestep_float32": first_diff,
                "event_step": EVENT_STEP,
                "prefix_bitwise_equal": True,
                "noise_draws_bitwise_equal": True,
                "noisy_values_match_float32_formula": True,
                "clean_shape": list(a["clean"].shape),
                "noise_shape": list(a["noise"].shape),
                "control": control,
                "shock": shock,
                "plot": {
                    "channel": 0,
                    "start_inclusive": START,
                    "stop_exclusive": STOP,
                    "stride": 2,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
