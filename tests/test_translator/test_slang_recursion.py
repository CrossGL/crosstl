"""Slang output refuses recursive call cycles, which slangc rejects for its GPU targets, and lowers self tail calls.

slangc refuses a recursive call for SPIR-V, DXIL, Metal and WGSL (error E55201, 'recursion not allowed'), so the
Slang generator refuses a cycle in the static call graph with the same :class:`RecursionUnsupportedError` as SPIR-V,
WGSL and Metal output (#4747). A function whose self calls are all tail calls is a loop first, which Slang writes
as CrossGL's unconditional ``loop`` statement, ``for (;;)``. When slangc and spirv-val are installed, the Slang
output is compiled and the SPIR-V validated.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.call_graph import RecursionUnsupportedError

DIRECT = """
shader Rec {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* o;
        int fact(int n) {
            if (n <= 1) { return 1; }
            return n * fact(n - 1);
        }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            o[i] = fact(o[i]);
        }
    }
}
"""

MUTUAL = """
shader Rec {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* o;
        int even(int n);
        int odd(int n) {
            if (n == 0) { return 0; }
            return even(n - 1);
        }
        int even(int n) {
            if (n == 0) { return 1; }
            return odd(n - 1);
        }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            o[i] = even(o[i]);
        }
    }
}
"""

TAIL = """
shader Rec {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* o;
        int sum(int n, int acc) {
            if (n <= 0) { return acc; }
            return sum(n - 1, acc + n);
        }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            o[i] = sum(o[i], 0);
        }
    }
}
"""

LOOP = """
shader Rec {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* o;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int n = o[i];
            int steps = 0;
            loop {
                if (n <= 1) { break; }
                n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
                steps += 1;
            }
            o[i] = steps;
        }
    }
}
"""


def translate(tmp_path, source, backend="slang"):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def compile_slang(tmp_path, slang):
    """Compile ``slang`` for SPIR-V with slangc and validate it with spirv-val, when both are installed."""
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source = tmp_path / "probe.slang"
    source.write_text(slang, encoding="utf-8")
    binary = tmp_path / "probe.spv"
    completed = subprocess.run(
        [slangc, str(source), "-target", "spirv", "-entry", "main", "-stage"]
        + ["compute", "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    spirv_val = shutil.which("spirv-val")
    if spirv_val is not None:
        validated = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", str(binary)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert validated.returncode == 0, validated.stderr


@pytest.mark.parametrize(
    "source,cycle",
    [(DIRECT, "fact -> fact"), (MUTUAL, "odd -> even -> odd")],
    ids=["direct", "mutual"],
)
def test_slang_refuses_a_recursive_call_cycle(tmp_path, source, cycle):
    with pytest.raises(RecursionUnsupportedError) as raised:
        translate(tmp_path, source)

    assert raised.value.code.endswith("slang-recursion-unsupported")
    assert f"recursive call cycle '{cycle}'" in str(raised.value)
    assert "E55201" in str(raised.value)


def test_slang_writes_a_self_tail_call_as_a_loop(tmp_path):
    slang = translate(tmp_path, TAIL)

    assert "for (;;)" in slang
    assert "sum(n - 1" not in slang
    compile_slang(tmp_path, slang)


def test_slang_writes_the_crossgl_loop_statement(tmp_path):
    slang = translate(tmp_path, LOOP)

    assert "for (;;)" in slang
    assert "LoopNode" not in slang
    compile_slang(tmp_path, slang)
