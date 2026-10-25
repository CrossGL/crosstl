"""``&&`` and ``||`` on boolean vectors, and HLSL's ``and``/``or``, on every target.

CrossGL, HLSL before 2021 and MSL combine vector operands of ``&&``/``||`` component by component. HLSL 2021 (DXC's
default), GLSL and WGSL accept only scalars: the HLSL generator emitted ``x && y`` on ``bool2`` (DXC rejects it),
GLSL refused it with a vector-to-scalar conversion error, and WGSL emitted ``x && y``, which naga rejects. WGSL has
no ``and``/``or`` built-in either, and SPIR-V refused them as unknown functions.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.vector_logic import lower_vector_logic

OPERATORS = """
RWStructuredBuffer<uint> outBuf : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float2 a = float2(id.x, 1.0);
    bool2 x = a > float2(0.5, 0.5);
    bool2 y = a < float2(2.0, 2.0);
    bool2 z = x && y;
    bool2 w = (x || y) && !z;
    bool2 q = x && (id.x > 1u);
    bool2 f = a && x;
    bool e = id.x > 0u && outBuf[id.x - 1u] > 0u;
    outBuf[0] = (z.x ? 1u : 0u) + (w.y ? 2u : 0u) + (q.x ? 4u : 0u) + (f.y ? 8u : 0u) + (e ? 16u : 0u);
}
"""

INTRINSICS = """
RWStructuredBuffer<uint> outBuf : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float2 a = float2(id.x, 1.0);
    bool2 x = a > float2(0.5, 0.5);
    bool2 y = a < float2(2.0, 2.0);
    bool2 z = and(x, y);
    bool2 w = or(and(x, y), !y);
    bool2 q = and(x, id.x > 1u);
    bool2 g = or(a, 1);
    bool s = and(id.x > 0u, id.y > 0u);
    outBuf[0] = (z.x ? 1u : 0u) + (w.y ? 2u : 0u) + (q.x ? 4u : 0u) + (g.y ? 8u : 0u) + (s ? 16u : 0u);
}
"""

METAL = """
#include <metal_stdlib>
using namespace metal;
kernel void main0(device uint* outBuf [[buffer(0)]], uint3 id [[thread_position_in_grid]]) {
    bool2 x = bool2(id.x > 1u, true);
    bool2 y = bool2(false, id.x < 3u);
    bool2 z = x && y;
    bool2 w = x || !y;
    outBuf[0] = (z.x ? 1u : 0u) + (w.y ? 2u : 0u);
}
"""


def _translate(tmp_path, source, backend, suffix="hlsl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    return " ".join(code.split())


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _validate(tmp_path, backend, code):
    if backend == "directx":
        if shutil.which("dxc") is None:
            pytest.skip("dxc is not installed")
        (tmp_path / "out.hlsl").write_text(code)
        _run(["dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"], tmp_path)
    elif backend == "opengl":
        if shutil.which("glslangValidator") is None:
            pytest.skip("glslangValidator is not installed")
        (tmp_path / "out.comp").write_text(code)
        _run(["glslangValidator", "out.comp"], tmp_path)
    elif backend == "wgsl":
        if shutil.which("naga") is None:
            pytest.skip("naga is not installed")
        (tmp_path / "out.wgsl").write_text(code)
        _run(["naga", "out.wgsl"], tmp_path)
    elif backend == "metal":
        if shutil.which("xcrun") is None:
            pytest.skip("the Metal compiler is not installed")
        (tmp_path / "out.metal").write_text(code)
        _run(["xcrun", "metal", "-c", "out.metal", "-o", "out.air"], tmp_path)
    elif backend == "vulkan":
        assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
        if assembler is None or validator is None:
            pytest.skip("SPIRV-Tools is not installed")
        (tmp_path / "out.spvasm").write_text(code)
        _run(
            [assembler, "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
            tmp_path,
        )
        _run([validator, "--target-env", "vulkan1.1", "out.spv"], tmp_path)


def test_hlsl_spells_vector_logic_with_and_or(tmp_path):
    flat = _flat(_translate(tmp_path, OPERATORS, "directx"))
    assert "bool2 z = and(x, y);" in flat
    assert "or (x, y)" in flat or "or(x, y)" in flat
    # Scalar operands keep the short-circuiting operator.
    assert "bool e = ((id.x > 0u) && " in flat
    assert "x && y" not in flat


def test_glsl_combines_vector_lanes(tmp_path):
    flat = _flat(_translate(tmp_path, OPERATORS, "opengl"))
    assert "bvec2 z = bvec2(uvec2(x) & uvec2(y));" in flat
    assert "bvec2 f = bvec2(uvec2(bvec2(a)) & uvec2(x));" in flat
    assert "bool e = ((gl_GlobalInvocationID.x > 0u) && " in flat


def test_glsl_types_nested_and_or(tmp_path):
    flat = _flat(_translate(tmp_path, INTRINSICS, "opengl"))
    assert "bvec2 w = bvec2(uvec2(bvec2(uvec2(x) & uvec2(y))) | uvec2(not(y)));" in flat
    assert "or (" not in flat and "or(" not in flat


def test_wgsl_uses_component_wise_operators(tmp_path):
    flat = _flat(_translate(tmp_path, OPERATORS, "wgsl"))
    assert "var z: vec2<bool> = (x & y);" in flat
    assert "var q: vec2<bool> = (x & vec2<bool>((id.x > 1u)));" in flat
    assert "var f: vec2<bool> = (vec2<bool>(a) & x);" in flat
    assert "var e: bool = ((id.x > 0u) && " in flat


def test_wgsl_lowers_and_or_calls(tmp_path):
    flat = _flat(_translate(tmp_path, INTRINSICS, "wgsl"))
    assert "var z: vec2<bool> = (x & y);" in flat
    assert "var g: vec2<bool> = (vec2<bool>(a) | vec2<bool>(bool(1)));" in flat
    assert "var s: bool = ((id.x > 0u) & (id.y > 0u));" in flat
    assert "and(" not in flat and "or(" not in flat


def test_spirv_lowers_and_or_calls(tmp_path):
    code = _translate(tmp_path, INTRINSICS, "vulkan")
    # The ids are symbolic when spirv-tools reformat the module and numeric otherwise, so resolve the type.
    boolean = re.search(r"^\s*(%\S+) = OpTypeBool$", code, re.MULTILINE)
    assert boolean is not None, code
    pattern = rf"^\s*(%\S+) = OpTypeVector {re.escape(boolean.group(1))} 2$"
    bool2 = re.search(pattern, code, re.MULTILINE)
    assert bool2 is not None, code
    for opcode in ("OpLogicalAnd", "OpLogicalOr"):
        assert re.search(rf"= {opcode} {re.escape(bool2.group(1))} ", code), code


def test_metal_converts_non_boolean_vector_operands(tmp_path):
    flat = _flat(_translate(tmp_path, OPERATORS, "metal"))
    assert "bool2 f = bool2(a) && x;" in flat
    assert "bool2 z = x && y;" in flat


@pytest.mark.parametrize("backend", ["directx", "opengl", "wgsl", "metal", "vulkan"])
@pytest.mark.parametrize("source", [OPERATORS, INTRINSICS], ids=["operators", "calls"])
def test_hlsl_vector_logic_validates(tmp_path, backend, source):
    _validate(tmp_path, backend, _translate(tmp_path, source, backend))


@pytest.mark.parametrize("backend", ["directx", "opengl", "wgsl", "vulkan"])
def test_metal_vector_logic_validates(tmp_path, backend):
    _validate(tmp_path, backend, _translate(tmp_path, METAL, backend, suffix="metal"))


def test_crossgl_scalar_logic_is_left_alone():
    source = """
    shader S {
        compute {
            void main() {
                bool a = true;
                bool b = a && false;
                bvec2 v = bvec2(a, b) && bvec2(b, a);
            }
        }
    }
    """
    program = Parser(Lexer(source).get_tokens()).parse()
    assert lower_vector_logic(program, "directx") == 1
    assert lower_vector_logic(program, "cgl") == 0
