"""Rust output tries a switch's default arm after its cases, wherever the switch declares ``default``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import crosstl
from tests.test_translator.test_codegen.test_rust_codegen import (
    assert_generated_rust_smoke_compiles,
)

SWITCH = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int k = iin[i] - 3;
            float acc = 0.0;
            switch (k) {
                %s
            }
            outp[i] = acc;
        }
    }
}
"""


def rust_of(body: str) -> str:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "probe.cgl")
        path.write_text(SWITCH % body, encoding="utf-8")
        return crosstl.translate(str(path), backend="rust", format_output=False)


def arm_order(rust: str) -> list[str]:
    return [
        line.strip().split(" =>")[0]
        for line in rust.splitlines()
        if line.strip().endswith("=> {")
    ]


def test_leading_default_arm_is_written_last(tmp_path):
    rust = rust_of("""default:
                    acc = 1.0;
                    break;
                case -1:
                    acc = 2.0;
                    break;
                case 2:
                case 5:
                    acc = 4.0;
                    break;""")
    assert arm_order(rust) == ["(-1)", "2 | 5", "_"]
    assert_generated_rust_smoke_compiles(rust, tmp_path)


def test_default_that_falls_through_keeps_the_bodies_it_reaches(tmp_path):
    rust = rust_of("""case 0:
                    acc = 8.0;
                default:
                    acc += 1.0;
                case 4:
                    acc += 2.0;
                    break;""")
    assert arm_order(rust) == ["0", "4", "_"]
    arms = rust.split("=> {")
    # case 0 falls through default into case 4; default falls into case 4.
    assert (
        "acc = 8.0;" in arms[1]
        and "acc += 1.0;" in arms[1]
        and "acc += 2.0;" in arms[1]
    )
    assert "acc += 1.0;" in arms[3] and "acc += 2.0;" in arms[3]
    assert "acc = 8.0;" not in arms[3]
    assert_generated_rust_smoke_compiles(rust, tmp_path)
