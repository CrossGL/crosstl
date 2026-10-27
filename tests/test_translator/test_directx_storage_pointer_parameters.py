"""Storage pointer helper parameters reach HLSL as structured-buffer views.

CrossGL spells a storage pointer parameter ``buffer T* p``; MSL spells it ``device T* p``. Both lower to a
``(RW)StructuredBuffer<T>`` plus an element offset. A read-only parameter that a call passes a writable buffer to
gets an ``RWStructuredBuffer<T>`` overload, because HLSL cannot convert a writable buffer to a read-only one.
"""

import shutil
import subprocess

import pytest

from crosstl import translate
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXResourcePointerParameterError,
    HLSLCodeGen,
)

BUFFER_HELPERS = """
shader BufferHelpers {
    struct Item { float a; vec4 b; }
    float readAt(buffer float* values, int index) { return values[index]; }
    float readAlias(buffer float* values, int index) { return readAt(values, index); }
    void writeAt(buffer float* values, int index, float value) { values[index] = value; }
    float readItem(buffer Item* q, int i) { return q[i].a + q[i].b.x; }
    void bumpNext(buffer float* p, int i) { p[i + 1] = p[i] + 1.0; }
    compute {
        buffer float* inputValues;
        buffer float* outputValues;
        buffer Item* items;
        void main(uint3 gid @ gl_GlobalInvocationID) {
            float value = readAlias(inputValues, int(gid.x));
            writeAt(outputValues, int(gid.x), value + readItem(items, int(gid.x)));
            bumpNext(outputValues + 2, 1);
            return;
        }
    }
}
"""

CONST_HELPER = """
shader ConstHelper {
    float readAt(const buffer float* values, int index) { return values[index]; }
    compute {
        buffer float* inputValues;
        buffer float* outputValues;
        void main(uint3 gid @ gl_GlobalInvocationID) {
            outputValues[gid.x] = readAt(inputValues, int(gid.x));
            return;
        }
    }
}
"""

METAL_CONST_HELPERS = """
#include <metal_stdlib>
using namespace metal;

float leaf(const device float* v, uint i) { return v[i] * 2.0f; }
float mid(const device float* a, const device float* b, uint i) {
    return leaf(a, i) + leaf(b + 1, i);
}

kernel void k(const device float* src [[buffer(0)]],
              device float* dst [[buffer(1)]],
              device float* scratch [[buffer(2)]],
              uint gid [[thread_position_in_grid]]) {
    scratch[gid] = mid(src, dst, gid);
    dst[gid] = mid(scratch, src, gid) + leaf(src, gid);
}
"""

METAL_READONLY_ONLY = """
#include <metal_stdlib>
using namespace metal;

float leaf(const device float* v, uint i) { return v[i]; }

kernel void k(const device float* src [[buffer(0)]],
              device float* dst [[buffer(1)]],
              uint gid [[thread_position_in_grid]]) {
    dst[gid] = leaf(src, gid);
}
"""


def _hlsl(source):
    return HLSLCodeGen().generate(parse(source))


def _metal_hlsl(source, tmp_path):
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend="directx", format_output=False)


def _signatures(code, name):
    return sorted(
        {
            line.strip().rstrip(";{ ")
            for line in code.splitlines()
            if line.strip().startswith(("float " + name + "(", "void " + name + "("))
        }
    )


def test_crossgl_buffer_pointer_parameters_lower_to_structured_buffers():
    code = _hlsl(BUFFER_HELPERS)
    assert "*" not in "".join(
        line for line in code.splitlines() if "(" in line and "Buffer<" in line
    )
    assert (
        "float readAt(RWStructuredBuffer<float> values, int64_t values_offset, "
        "int index)" in code
    )
    assert (
        "void writeAt(RWStructuredBuffer<float> values, int64_t values_offset" in code
    )
    assert "float readItem(RWStructuredBuffer<Item> q, int64_t q_offset, int i)" in code
    assert "return readAt(values, int64_t(values_offset), index);" in code
    assert "p[uint((p_offset + (i + 1)))] = (p[uint((p_offset + i))] + 1.0);" in code
    assert "bumpNext(outputValues, int64_t(2), 1);" in code


def test_readonly_parameter_gets_a_writable_overload_for_writable_arguments():
    code = _hlsl(CONST_HELPER)
    assert _signatures(code, "readAt") == [
        "float readAt(RWStructuredBuffer<float> values, int64_t values_offset, int index)",
        "float readAt(StructuredBuffer<float> values, int64_t values_offset, int index)",
    ]
    # Prototype and definition of each overload.
    assert code.count("float readAt(RWStructuredBuffer<float> values") == 2
    assert "readAt(inputValues, int64_t(0), int(gid.x))" in code


def test_writable_overloads_follow_forwarded_parameters_and_combinations(tmp_path):
    code = _metal_hlsl(METAL_CONST_HELPERS, tmp_path)
    assert _signatures(code, "leaf") == [
        "float leaf(RWStructuredBuffer<float> v, int64_t v_offset, uint i)",
        "float leaf(StructuredBuffer<float> v, int64_t v_offset, uint i)",
    ]
    assert _signatures(code, "mid") == [
        "float mid(RWStructuredBuffer<float> a, int64_t a_offset, "
        "StructuredBuffer<float> b, int64_t b_offset, uint i)",
        "float mid(StructuredBuffer<float> a, int64_t a_offset, "
        "RWStructuredBuffer<float> b, int64_t b_offset, uint i)",
        "float mid(StructuredBuffer<float> a, int64_t a_offset, "
        "StructuredBuffer<float> b, int64_t b_offset, uint i)",
    ]


def test_readonly_arguments_keep_a_single_declared_overload(tmp_path):
    code = _metal_hlsl(METAL_READONLY_ONLY, tmp_path)
    assert _signatures(code, "leaf") == [
        "float leaf(StructuredBuffer<float> v, int64_t v_offset, uint i)"
    ]


def test_output_is_deterministic_across_generations(tmp_path):
    assert _metal_hlsl(METAL_CONST_HELPERS, tmp_path) == _metal_hlsl(
        METAL_CONST_HELPERS, tmp_path
    )


def test_writable_overload_limit_is_a_typed_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(HLSLCodeGen, "HLSL_RESOURCE_ACCESS_VARIANT_LIMIT", 1)
    with pytest.raises(DirectXResourcePointerParameterError) as error:
        _metal_hlsl(METAL_CONST_HELPERS, tmp_path)
    assert error.value.reason == "resource-access-variant-limit-exceeded"
    assert error.value.function_name == "mid"


def test_unconverged_overloads_are_a_typed_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(HLSLCodeGen, "HLSL_RESOURCE_ACCESS_PASS_LIMIT", 1)
    with pytest.raises(DirectXResourcePointerParameterError) as error:
        _metal_hlsl(METAL_CONST_HELPERS, tmp_path)
    assert error.value.reason == "resource-access-variants-unresolved"


@pytest.mark.parametrize(
    "name,generate",
    [
        ("buffer-helpers", lambda _path: _hlsl(BUFFER_HELPERS)),
        ("const-helper", lambda _path: _hlsl(CONST_HELPER)),
        ("metal-const-helpers", lambda path: _metal_hlsl(METAL_CONST_HELPERS, path)),
    ],
)
@pytest.mark.parametrize("spirv", [False, True])
def test_generated_helpers_compile_with_dxc(tmp_path, name, generate, spirv):
    compiler = shutil.which("dxc")
    if compiler is None:
        pytest.skip("DXC is not installed")
    source = tmp_path / f"{name}.hlsl"
    source.write_text(generate(tmp_path), encoding="utf-8")
    command = [compiler, "-T", "cs_6_6", "-E", "CSMain", "-WX", str(source)]
    if spirv:
        command.insert(1, "-spirv")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
