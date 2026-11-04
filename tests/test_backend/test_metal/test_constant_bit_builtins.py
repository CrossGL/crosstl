"""Clang bit-counting builtins fold in Metal constant expressions that size template arguments.

``constexpr short logN = __builtin_ctz(N);`` is a core constant expression once ``N`` is concrete. Template
materialization used to leave every local constant derived from it unresolved and refuse the specialization.
"""

import shutil
import subprocess

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.constant_bit_builtins import fold_bit_builtin
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)

SOURCE = """#include <metal_stdlib>
using namespace metal;
template <short R>
void scale_first(thread float* x) { x[0] = x[0] * R; }
template <int N, int R>
[[kernel]] void stage(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  constexpr short logN = __builtin_ctz(N);
  constexpr short logR = LOG_R;
  constexpr short logFinal = logN % logR;
  constexpr short final_radix = 1 << (logFinal);
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  scale_first<final_radix>(x);
  out[i] = x[0];
}
template [[host_name("stage64")]] [[kernel]] decltype(stage<64, 16>) stage<64, 16>;
"""


@pytest.mark.parametrize(
    "name, operand, expected",
    [
        ("__builtin_ctz", 64, 6),
        ("__builtin_ctz", 1, 0),
        ("__builtin_ctz", -8, 3),
        ("__builtin_ctz", 1 << 31, 31),
        ("__builtin_ctzll", 1 << 40, 40),
        ("__builtin_clz", 1, 31),
        ("__builtin_clz", -1, 0),
        ("__builtin_clzl", 1, 63),
        ("__builtin_popcount", 0, 0),
        ("__builtin_popcount", 255, 8),
        ("__builtin_popcount", -1, 32),
        ("__builtin_popcountll", -1, 64),
    ],
)
def test_bit_builtins_fold_with_their_operand_width(name, operand, expected):
    assert fold_bit_builtin(name, operand) == expected


@pytest.mark.parametrize(
    "name, operand",
    [
        # Undefined for zero; Clang rejects these calls in constant expressions.
        ("__builtin_ctz", 0),
        ("__builtin_clzll", 0),
        # Neither a signed nor an unsigned 32-bit value.
        ("__builtin_ctz", 1 << 32),
        ("__builtin_popcount", -(1 << 31) - 1),
        ("__builtin_parity", 3),
        ("ctz", 4),
    ],
)
def test_bit_builtins_without_a_defined_constant_value_do_not_fold(name, operand):
    assert fold_bit_builtin(name, operand) is None


@pytest.mark.parametrize(
    "log_r, specialization",
    [
        ("__builtin_ctz(R)", "scale_first_4"),
        ("__builtin_popcount(R - 1)", "scale_first_4"),
    ],
)
def test_template_arguments_derived_from_bit_builtins_materialize(
    log_r, specialization
):
    processed = MetalPreprocessor().preprocess(
        SOURCE.replace("LOG_R", log_r), "stage.metal"
    )
    assert f"{specialization}(x);" in processed
    assert "scale_first<" not in processed


@pytest.mark.parametrize(
    "log_r", ["__builtin_ctz(R - R)", "__builtin_clz(0)", "__builtin_parity(R)"]
)
def test_bit_builtins_without_a_constant_value_keep_the_specialization_refused(log_r):
    with pytest.raises(MetalTemplateSpecializationError) as caught:
        MetalPreprocessor().preprocess(SOURCE.replace("LOG_R", log_r), "stage.metal")
    assert "final_radix" in caught.value.unresolved_local_constants


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_bit_builtin_template_arguments_translate(tmp_path, backend):
    path = tmp_path / "stage.metal"
    path.write_text(SOURCE.replace("LOG_R", "__builtin_ctz(R)"))
    code = translate(str(path), backend=backend, format_output=False)
    assert "scale_first_4(" in code
    suffix = "comp" if backend == "opengl" else "hlsl"
    artifact = tmp_path / f"stage.{suffix}"
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
            [
                "dxc",
                "-T",
                "cs_6_2",
                "-enable-16bit-types",
                "-E",
                "CSMain",
                str(artifact),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
