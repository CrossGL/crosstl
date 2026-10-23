"""``GetDimensions(count, stride)`` reports the element stride on targets without a stride query.

HLSL's structured-buffer ``GetDimensions`` writes the element count and the stride in bytes of the declared element
under HLSL's packing; DXC folds the stride to a literal. GLSL, SPIR-V, CUDA and HIP count elements only, so their
generators left the stride unassigned (GLSL, CUDA, HIP) or refused the query (SPIR-V).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest

import crosstl
from crosstl.translator.semantics.structured_layouts import (
    StructuredLayoutUnsupportedError,
)

REQUIRE_ENV = "CROSSTL_REQUIRE_BUFFER_STRIDE_EXECUTION"

# S is 20 bytes under HLSL packing (std430 would align it to 32); float3 is 12 (std430: 16).
HLSL = """
struct S { float4 a; float b; };
StructuredBuffer<S> items : register(t0);
StructuredBuffer<float4> wide : register(t1);
StructuredBuffer<float3> narrow : register(t2);
RWStructuredBuffer<uint> results : register(u0);
uint helper(StructuredBuffer<float4> b) {
    uint c, s;
    b.GetDimensions(c, s);
    return s;
}
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint count, stride;
    items.GetDimensions(count, stride);
    results[0] = count;
    results[1] = stride;
    if (id.x == 0)
        wide.GetDimensions(count, stride);
    results[2] = count;
    results[3] = stride;
    narrow.GetDimensions(count, stride);
    results[4] = count;
    results[5] = stride;
    results[6] = helper(wide);
}
"""

# Element counts of the buffers the execution check binds.
ITEMS, WIDE, NARROW = 3, 5, 7
EXPECTED = [ITEMS, 20, WIDE, 16, NARROW, 12, 16]


def translate(tmp_path, backend, source=HLSL, suffix="hlsl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def stride_assignments(code):
    return re.findall(r"\b(stride|s) = (\d+)u;", code)


def test_opengl_assigns_hlsl_strides(tmp_path):
    glsl = translate(tmp_path, "opengl")
    # The float3 buffer becomes words (structured_layouts), whose query reports 12 itself.
    assert sorted(stride_assignments(glsl)) == [
        ("s", "16"),
        ("stride", "12"),
        ("stride", "16"),
        ("stride", "20"),
    ]
    # The query under the if statement keeps both statements under it.
    assert re.search(
        r"if \(.*\) \{\s*count = wide\.length\(\);\s*stride = 16u;\s*\}", glsl
    ), glsl
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    artifact = tmp_path / "shader.comp"
    artifact.write_text(glsl, encoding="utf-8")
    result = subprocess.run(
        [glslang, str(artifact)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_spirv_stores_hlsl_strides(tmp_path):
    from tests.spirv_friendly_names import friendly_spirv

    assembly = friendly_spirv(translate(tmp_path, "vulkan"))
    for value in (20, 16, 12):
        assert re.search(rf"OpStore %stride\w* %uint_{value}\b", assembly), value
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("SPIRV-Tools is not installed")
    source = tmp_path / "shader.spvasm"
    binary = tmp_path / "shader.spv"
    source.write_text(translate(tmp_path, "vulkan"), encoding="utf-8")
    subprocess.run(
        [spirv_as, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        check=True,
    )
    result = subprocess.run(
        [spirv_val, "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_cuda_and_hip_assign_hlsl_strides(tmp_path, backend):
    source = HLSL.replace("register(u0)", "register(u3)")
    code = translate(tmp_path, backend, source)
    assert sorted(stride_assignments(code)) == [
        ("s", "16"),
        ("stride", "12"),
        ("stride", "16"),
        ("stride", "20"),
    ]


def test_native_stride_queries_are_kept(tmp_path):
    hlsl = translate(tmp_path, "directx")
    assert "items.GetDimensions(count, stride)" in hlsl
    slang = translate(tmp_path, "slang")
    assert "items.GetDimensions(count, stride)" in slang


def test_crossgl_buffers_without_layout_use_the_target_rules(tmp_path):
    source = """
shader main {
    StructuredBuffer<vec3> points;
    RWStructuredBuffer<uint> results;
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        uint count;
        uint stride;
        buffer_dimensions(points, count, stride);
        results[0] = stride;
    }
    }
}
"""
    # std430 aligns vec3 to 16 bytes.
    glsl = translate(tmp_path, "opengl", source, "cgl")
    assert stride_assignments(glsl) == [("stride", "16")]
    with pytest.raises(StructuredLayoutUnsupportedError, match="declares no buffer"):
        translate(tmp_path, "cuda", source, "cgl")


def test_helpers_take_the_stride_of_the_buffers_their_calls_pass(tmp_path):
    source = """
shader main {
    @ buffer_layout(hlsl_structured)
    StructuredBuffer<vec3> packed;
    @ buffer_layout(metal)
    StructuredBuffer<vec3> padded;
    RWStructuredBuffer<uint> results;
    uint inner(StructuredBuffer<vec3> b) {
        uint c;
        uint s;
        buffer_dimensions(b, c, s);
        return s;
    }
    uint outer(StructuredBuffer<vec3> b) {
        return inner(b);
    }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        results[0] = outer(packed);
        results[1] = outer(PADDED);
    }
    }
}
"""
    # HLSL packs float3 in 12 bytes, MSL pads it to 16; a helper reached by both cannot report one stride.
    one = source.replace("outer(PADDED)", "outer(packed)")
    assert stride_assignments(translate(tmp_path, "cuda", one, "cgl")) == [("s", "12")]
    other = source.replace("outer(packed)", "outer(padded)").replace("PADDED", "padded")
    assert stride_assignments(translate(tmp_path, "cuda", other, "cgl")) == [
        ("s", "16")
    ]
    mixed = source.replace("PADDED", "padded")
    with pytest.raises(StructuredLayoutUnsupportedError, match="strides 12, 16"):
        translate(tmp_path, "cuda", mixed, "cgl")


OVERLOADED_HLSL = """
RWTexture2D<float4> image : register(u0);
StructuredBuffer<float> weights : register(t0);
RWStructuredBuffer<uint> results : register(u1);

uint width(RWTexture2D<float4> resource)
{
    uint w, h;
    resource.GetDimensions(w, h);
    return w;
}

uint width(StructuredBuffer<float> resource)
{
    uint count, stride;
    resource.GetDimensions(count, stride);
    return stride;
}

[numthreads(1, 1, 1)]
void main()
{
    results[0] = width(image);
    results[1] = width(weights);
}
"""


@pytest.mark.parametrize("backend", ["opengl", "cuda"])
def test_texture_and_buffer_overloads_of_a_helper(tmp_path, backend):
    # Only the buffer overload is reached by a buffer, so it takes that buffer's 4-byte stride.
    code = translate(tmp_path, backend, OVERLOADED_HLSL)
    assert stride_assignments(code) == [("stride", "4")]


def test_buffer_overloads_reached_by_different_strides_are_refused(tmp_path):
    # Both overloads take a buffer, so each counts both calls, and float and float4 strides disagree.
    source = OVERLOADED_HLSL.replace(
        "RWTexture2D<float4> resource", "StructuredBuffer<float4> resource"
    ).replace(
        "RWTexture2D<float4> image : register(u0);",
        "StructuredBuffer<float4> image : register(t1);",
    )
    with pytest.raises(StructuredLayoutUnsupportedError, match="strides 4, 16"):
        translate(tmp_path, "opengl", source)


def test_queries_in_expressions_are_refused(tmp_path):
    from crosstl.translator import ast as A
    from crosstl.translator.semantics.buffer_strides import (
        materialize_buffer_strides,
    )

    query = A.FunctionCallNode(
        A.IdentifierNode("buffer_dimensions"),
        [A.IdentifierNode("items"), A.IdentifierNode("c"), A.IdentifierNode("s")],
    )
    function = A.FunctionNode(
        "main",
        A.PrimitiveType("void"),
        [],
        A.BlockNode([A.ReturnNode(query)]),
    )
    buffer = A.VariableNode(
        "items", A.NamedType("StructuredBuffer", [A.PrimitiveType("float")])
    )
    program = A.ShaderNode(
        "main", "compute", {}, [], [function], global_variables=[buffer]
    )
    with pytest.raises(StructuredLayoutUnsupportedError, match="in an expression"):
        materialize_buffer_strides(program, "opengl")


def test_strides_execute_on_opengl(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform == "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on Linux to require the OpenGL dispatch")
    import moderngl

    glsl = translate(tmp_path, "opengl")
    context = moderngl.create_standalone_context(require=430, backend="egl")
    buffers = []
    try:
        shader = context.compute_shader(glsl)
        bindings = {
            name: int(binding)
            for binding, name in re.findall(
                r"layout\(std430, binding = (\d+)\)[^{]*\{[^}]*?\b(\w+)\[\];", glsl
            )
        }
        sizes = {"items": 20 * ITEMS, "wide": 16 * WIDE, "narrow": 12 * NARROW}
        for name, size in sizes.items():
            buffer = context.buffer(reserve=size)
            buffer.bind_to_storage_buffer(bindings[name])
            buffers.append(buffer)
        results = context.buffer(reserve=4 * len(EXPECTED))
        results.bind_to_storage_buffer(bindings["results"])
        buffers.append(results)
        shader.run(group_x=1)
        context.finish()
        values = list(memoryview(results.read()).cast("I"))
        assert values == EXPECTED, (values, glsl)
    finally:
        for buffer in buffers:
            buffer.release()
        context.release()
