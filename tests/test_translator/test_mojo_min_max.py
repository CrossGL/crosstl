"""Mojo output for CrossGL ``min``, ``max`` and ``clamp`` on every numeric scalar and vector type.

They become Mojo's ``std.math.min``/``max`` with each operand converted to the result type, and ``clamp(x, lo, hi)``
becomes ``min(max(x, lo), hi)``. The generator declares no module-level ``min``/``max``/``clamp``, which would hide
the standard ones for integer, ``Float16`` and ``Float64`` operands.
"""

from __future__ import annotations

import subprocess

import pytest

import crosstl
from tests.test_translator.test_codegen.test_mojo_codegen import find_mojo_compiler

KERNEL = """shader Probe {{
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer ivec2* iv;
        buffer uvec3* uv;
        buffer dvec2* dv;
        buffer double* ds;
        buffer int* is;
        buffer uint* us;
        buffer half* hs;
        buffer half2* hv;
        buffer vec3* fv;
        buffer float* fs;
        float twice(float x) {{ return x * 2.0; }}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""

CASES = {
    "integer-vectors": (
        "iv[i] = min(iv[i], iv[i + 1u]); iv[i + 1u] = max(iv[i], 3);"
        " iv[i + 2u] = clamp(iv[i], 1, 9); iv[i + 3u] = clamp(iv[i], iv[i + 1u], ivec2(7, 8));"
        " uv[i] = max(uv[i], uv[i + 1u]); uv[i + 1u] = min(uv[i], 4u);",
        [
            "iv[Int(i)] = min(iv[Int(i)], iv[Int((i + 1))])",
            "iv[Int((i + 1))] = max(iv[Int(i)], SIMD[DType.int32, 2](3))",
            "min(max(iv[Int(i)], SIMD[DType.int32, 2](1)), SIMD[DType.int32, 2](9))",
            "uv[Int((i + 1))] = min(uv[Int(i)], SIMD[DType.uint32, 4](4, 4, 4, 0))",
        ],
    ),
    "doubles": (
        "dv[i] = min(dv[i], dv[i + 1u]); dv[i + 1u] = clamp(dv[i], 0.0, 1.0);"
        " ds[i] = max(ds[i], ds[i + 1u]); ds[i + 1u] = clamp(ds[i], ds[i + 2u], ds[i + 3u]);",
        [
            "ds[Int(i)] = max(ds[Int(i)], ds[Int((i + 1))])",
            "min(max(dv[Int(i)], SIMD[DType.float64, 2](0.0)), SIMD[DType.float64, 2](1.0))",
        ],
    ),
    "halves": (
        "hs[i] = max(hs[i], hs[i + 1u]); hv[i] = clamp(hv[i], hs[i], hs[i + 1u]);",
        [
            "hs[Int(i)] = max(hs[Int(i)], hs[Int((i + 1))])",
            "min(max(hv[Int(i)], SIMD[DType.float16, 2](hs[Int(i)])), "
            "SIMD[DType.float16, 2](hs[Int((i + 1))]))",
        ],
    ),
    "integer-scalars": (
        "us[i] = min(us[i], 4u); us[i + 1u] = max(3u, us[i]);"
        " is[i] = min(is[i], int(us[i])); is[i + 1u] = clamp(is[i], 0, 5); is[i + 2u] = max(3, 4);",
        [
            "us[Int(i)] = min(us[Int(i)], UInt32(4))",
            "us[Int((i + 1))] = max(UInt32(3), us[Int(i)])",
            "is_[Int((i + 1))] = min(max(is_[Int(i)], Int32(0)), Int32(5))",
            "is_[Int((i + 2))] = max(Int32(3), Int32(4))",
        ],
    ),
    "float-vectors": (
        "fv[i] = max(fv[i], twice(fs[i])); fv[i + 1u] = clamp(fv[i], 0.0, 1.0);"
        " fs[i] = clamp(fs[i], 0.0, fs[i + 1u]); fs[i + 1u] = min(1.0, 2.0);",
        [
            # A call is evaluated once and splatted with the vec3 padding lane kept at zero.
            "fv[Int(i)] = max(fv[Int(i)], _crossgl_vec3_splat_f32(twice(fs[Int(i)])))",
            "min(max(fv[Int(i)], SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)), "
            "SIMD[DType.float32, 4](1.0, 1.0, 1.0, 0.0))",
            "fs[Int(i)] = min(max(fs[Int(i)], Float32(0.0)), fs[Int((i + 1))])",
            "fs[Int((i + 1))] = min(Float32(1.0), Float32(2.0))",
        ],
    ),
}


def translate(tmp_path, name):
    source = tmp_path / f"{name}.cgl"
    source.write_text(KERNEL.format(body=CASES[name][0]))
    return crosstl.translate(str(source), backend="mojo")


@pytest.mark.parametrize("name", sorted(CASES))
def test_min_max_clamp_use_std_min_max_on_the_result_type(tmp_path, name):
    code = translate(tmp_path, name)

    for line in CASES[name][1]:
        assert line in code
    for helper in ("def min(", "def max(", "def clamp("):
        assert helper not in code


@pytest.mark.parametrize("name", sorted(CASES))
def test_min_max_clamp_build_with_mojo(tmp_path, name):
    mojo = find_mojo_compiler()
    source = tmp_path / f"{name}.mojo"
    source.write_text(translate(tmp_path, name))

    result = subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "llvm",
            "--disable-warnings",
            str(source),
            "-o",
            str(tmp_path / f"{name}.ll"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
