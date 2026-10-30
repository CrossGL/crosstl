"""Metal value variable templates read as their initializer at each concrete use.

CrossGL has no variable templates. ``packing_size<uchar>`` used to reach the targets as the
undeclared identifier ``packing_size_u3cuchar_u3e``; it now folds to a literal of the declared type,
an array bound folds to its plain value, and a non-integral initializer is converted to the declared
type.
"""

import shutil
import subprocess

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

HEADER = """#include <metal_stdlib>
using namespace metal;

template <typename T>
constexpr constant uint packing_size = sizeof(uint) / sizeof(T);

template <int N>
constexpr constant int negated = -N;

template <typename T>
constexpr constant float scaled = float(sizeof(T)) * 0.5f;
"""

KERNEL = """
kernel void lanes(device uint* out [[buffer(0)]], device int* signed_out [[buffer(1)]],
                  device float* scale_out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = gid / packing_size<uchar> + gid % packing_size<half>;
  signed_out[gid] = int(gid) + negated<3>;
  scale_out[gid] = scaled<ushort>;
}
"""


def test_value_variable_templates_fold_at_concrete_uses():
    processed = MetalPreprocessor().preprocess(HEADER + KERNEL, "lanes.metal")
    body = processed[processed.index("kernel void lanes") :]
    assert "gid / 4u + gid % 2u" in body
    assert "int(gid) + (-3)" in body
    assert "scaled<" not in body and "float(float(sizeof(ushort)) * 0.5f)" in body


def test_value_variable_template_array_bound_folds_to_its_value():
    source = HEADER + (
        "template <typename T> struct Lanes { T val[packing_size<T>]; };\n"
        "kernel void k(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "  Lanes<ushort> lanes;\n  lanes.val[0] = ushort(gid);\n  lanes.val[1] = ushort(1);\n"
        "  out[gid] = uint(lanes.val[0]) + uint(lanes.val[1]);\n}\n"
    )
    processed = MetalPreprocessor().preprocess(source, "lanes.metal")
    assert "ushort val[2];" in processed


def test_uses_inside_unmaterialized_templates_are_left_for_materialization():
    source = HEADER + (
        "template <typename T> uint words(uint n) { return n / packing_size<T>; }\n"
    )
    processed = MetalPreprocessor().preprocess(source, "lanes.metal")
    assert "n / packing_size<T>" in processed


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_value_variable_templates_translate(tmp_path, backend):
    path = tmp_path / "lanes.metal"
    path.write_text(HEADER + KERNEL)
    code = translate(str(path), backend=backend, format_output=False)
    assert "packing_size" not in code and "negated" not in code and "scaled" not in code
    suffix = "comp" if backend == "opengl" else "hlsl"
    artifact = tmp_path / f"lanes.{suffix}"
    artifact.write_text(code)
    if backend == "opengl" and shutil.which("glslangValidator"):
        completed = subprocess.run(
            ["glslangValidator", "-S", "comp", str(artifact)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout[-2000:]
    if backend == "directx" and shutil.which("dxc"):
        completed = subprocess.run(
            ["dxc", "-T", "cs_6_0", "-E", "CSMain", str(artifact)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
