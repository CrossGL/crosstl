"""MSL's ``isordered``, ``isunordered`` and ``isnormal`` on targets without them.

They reached GLSL, HLSL, WGSL and the C-family targets verbatim, as calls to undeclared functions, and SPIR-V refused
them. See :mod:`crosstl.translator.semantics.float_relations`.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.float_relations import (
    FloatRelationUnsupportedError,
)

RELATIONS = """
#include <metal_stdlib>
using namespace metal;

kernel void relations(device const float4 *values [[buffer(0)]],
                      device uint *flags [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
    float4 a = values[2 * tid];
    float4 b = values[2 * tid + 1];
    bool4 normal = isnormal(a);
    bool2 ordered = isordered(a.xy, b.xy);
    uint bits = uint(isunordered(a.x, b.x)) | (uint(isordered(a.y, b.y)) << 1)
        | (uint(isnormal(b.z)) << 2) | (uint(normal.w) << 3) | (uint(ordered.y) << 4)
        | (uint(isunordered(values[tid].x, values[tid + 1].y)) << 5);
    flags[tid] = bits;
}
"""

SIDE_EFFECTS = """
shader Relations {
    float bump(inout float v) {
        v = v + 1.0;
        return v;
    }

    compute {
        layout(local_size_x = 1) in;
        buffer float data[];
        void main() {
            float x = data[0];
            float y = data[1];
            vec2 p = vec2(data[3], data[4]);
            bool both = isunordered(bump(x), bump(y));
            bool one = isordered(x, bump(y));
            bvec2 lanes = isunordered(bump(x) * p, p.yx);
            data[2] = (both ? 1.0 : 0.0) + (one ? 2.0 : 0.0) + (lanes.y ? 4.0 : 0.0);
        }
    }
}
"""

NORMAL_HALF = """
shader Relations {
    compute {
        layout(local_size_x = 1) in;
        buffer float data[];
        void main() {
            half h = half(data[0]);
            data[1] = isnormal(h) ? 1.0 : 0.0;
        }
    }
}
"""

NORMAL_IMPURE = """
shader Relations {
    float bump(inout float v) {
        v = v + 1.0;
        return v;
    }

    compute {
        layout(local_size_x = 1) in;
        buffer float data[];
        void main() {
            float x = data[0];
            data[1] = isnormal(bump(x)) ? 1.0 : 0.0;
        }
    }
}
"""

NORMAL_DOUBLE = """
shader Relations {
    compute {
        layout(local_size_x = 1) in;
        buffer float data[];
        void main() {
            double d = double(data[0]);
            data[1] = isnormal(d) ? 1.0 : 0.0;
        }
    }
}
"""

USER_ISNORMAL = """
shader Relations {
    bool isnormal(float v) {
        return v > 0.0;
    }

    compute {
        layout(local_size_x = 1) in;
        buffer float data[];
        void main() {
            data[1] = isnormal(data[0]) ? 1.0 : 0.0;
        }
    }
}
"""

NAMES = ("isordered", "isunordered", "isnormal")


def _translate(tmp_path, source, backend, suffix=".metal"):
    path = tmp_path / f"relations{suffix}"
    path.write_text(source, encoding="utf-8")
    options = {"threadgroup_size": (64, 1, 1)} if suffix == ".metal" else None
    # Unformatted, so the expected text does not depend on the local clang-format.
    return crosstl.translate(
        str(path), backend=backend, source_options=options, format_output=False
    )


def _run(command, **kwargs):
    completed = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-2000:]


def _validate(tmp_path, backend, code):
    if backend == "directx" and shutil.which("dxc"):
        path = tmp_path / "relations.hlsl"
        path.write_text(code, encoding="utf-8")
        _run(
            [shutil.which("dxc"), "-T", "cs_6_2", "-E", "CSMain", "-HV", "2021"]
            + ["-enable-16bit-types", str(path)],
            timeout=180,
        )
    elif backend == "opengl" and shutil.which("glslangValidator"):
        path = tmp_path / "relations.comp"
        path.write_text(code, encoding="utf-8")
        _run([shutil.which("glslangValidator"), str(path)])
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly, module = tmp_path / "relations.spvasm", tmp_path / "relations.spv"
        assembly.write_text(code, encoding="utf-8")
        _run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(module)]
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", str(module)])


def _assert_lowered(code):
    for name in NAMES:
        assert f"{name}(" not in code


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_metal_relations_lower_to_valid_target_code(tmp_path, backend):
    code = _translate(tmp_path, RELATIONS, backend)
    _assert_lowered(code)
    _validate(tmp_path, backend, code)


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("directx", ["(isnan(a.x) || isnan(b.x))", "!or(isnan(a.xy), isnan(b.xy))"]),
        ("opengl", ["(isnan(a.x) || isnan(b.x))", "not(bvec2(uvec2(isnan(a.xy))"]),
    ],
)
def test_unordered_tests_are_nan_tests_of_both_operands(tmp_path, backend, expected):
    code = _translate(tmp_path, RELATIONS, backend)
    _assert_lowered(code)
    for text in expected:
        assert text in code


def test_isnormal_compares_the_magnitude_with_the_smallest_normal_value(tmp_path):
    code = _translate(tmp_path, RELATIONS, "directx")
    # 2^-126, the smallest normal binary32 value, applied to each component of a vector.
    assert "(isfinite(b.z) && (abs(b.z) >= 1.1754943508222875e-38))" in code
    assert "bool4((isfinite(a.x) && (abs(a.x) >= 1.1754943508222875e-38))" in code
    assert "(abs(a.w) >= 1.1754943508222875e-38)" in code


def test_isnormal_of_half_uses_the_binary16_bound(tmp_path):
    code = _translate(tmp_path, NORMAL_HALF, "directx", suffix=".cgl")
    _assert_lowered(code)
    assert ">= 6.103515625e-05" in code


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_operands_with_side_effects_are_evaluated_once_each(tmp_path, backend):
    code = _translate(tmp_path, SIDE_EFFECTS, backend, suffix=".cgl")
    _assert_lowered(code)
    _validate(tmp_path, backend, code)
    if backend == "vulkan":
        return
    # Both operands have effects: no short circuit, and the source order is kept.
    assert "bool((uint(isnan(bump(x))) | uint(isnan(bump(y)))))" in code
    # One operand has effects: it goes first, where the short circuit always evaluates it.
    assert "(isnan(bump(y)) || isnan(x))" in code


def test_vector_operands_with_side_effects_keep_the_componentwise_or(tmp_path):
    code = _translate(tmp_path, SIDE_EFFECTS, "directx", suffix=".cgl")
    assert "or(isnan((bump(x) * p)), isnan(p.yx))" in code


@pytest.mark.parametrize("source", [NORMAL_IMPURE, NORMAL_DOUBLE])
def test_isnormal_without_an_exact_lowering_is_refused(tmp_path, source):
    with pytest.raises(FloatRelationUnsupportedError) as caught:
        _translate(tmp_path, source, "opengl", suffix=".cgl")
    assert caught.value.code == (
        "project.translate.opengl-metal-relational-function-unsupported"
    )
    assert caught.value.missing_capabilities == ("opengl.metal-relational-isnormal",)


def test_a_source_defined_isnormal_keeps_its_calls(tmp_path):
    code = _translate(tmp_path, USER_ISNORMAL, "directx", suffix=".cgl")
    assert "isnormal(data[0])" in code or "isnormal(data.Load(0))" in code
    assert "isfinite" not in code


def test_metal_keeps_its_own_relational_functions(tmp_path):
    code = _translate(tmp_path, RELATIONS, "metal")
    assert "isunordered(a.x, b.x)" in code
    assert "isordered(a.xy, b.xy)" in code
    assert "isnormal(a)" in code
