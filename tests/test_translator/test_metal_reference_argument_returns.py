"""Metal copies reference arguments of a returned call back before the ``return``.

CrossGL copies an ``out``/``inout`` argument back when the called function returns (OpenGL Shading Language 4.60,
section 6.1.1), which precedes the ``return`` that evaluated the call. Metal binds such an argument through a
``thread`` temporary when it is threadgroup or device storage or a vector component, so the returned value is
held in a local while the temporary is stored back.
"""

import shutil
import subprocess

import pytest

import crosstl

THREADGROUP = """shader S {
    compute {
        layout(std430, binding = 0) buffer Seeds { int seed[]; } seeds;
        layout(std430, binding = 1) buffer Out { int values[]; } outBuf;
        layout(local_size_x = 1) in;
        shared int tile[4];
        int bump(inout int t) { t = t + 10; return t * 2; }
        float widen(inout int t) { t = t + 1; return float(t) + 0.5; }
        int fromTile(int i) { return bump(tile[i]); }
        float fromTileF(int i) { return widen(tile[i]); }
        int viaTail(int i) { bump(tile[i]) }
        int branchy(int i) {
            if (i > 1) return bump(tile[i]);
            switch (i) {
                case 0: return bump(tile[0]);
                case 1: return bump(tile[1]) - 1;
            }
            return 0;
        }
        void main() {
            int base = seeds.seed[0];
            tile[0] = base; tile[1] = 2 * base; tile[2] = 3 * base; tile[3] = 4 * base;
            int r0 = fromTile(0);
            float r1 = fromTileF(1);
            int r2 = viaTail(3);
            int r4 = branchy(2);
            outBuf.values[0] = r0 * 1000 + tile[0];
            outBuf.values[1] = int(r1 * 2.0) * 1000 + tile[1];
            outBuf.values[2] = r2 * 1000 + tile[3];
            outBuf.values[3] = r4 * 1000 + tile[2];
        }
    }
}
"""

HLSL = """StructuredBuffer<int> seeds : register(t0);
RWStructuredBuffer<int> outBuf : register(u0);
groupshared int tile[4];
int bump(inout int t) { t = t + 10; return t * 2; }
void clear(out int t) { t = -1; }
int2 fromComp(inout int2 v) { return int2(bump(v.y), v.x); }
void clearTile(int i) { return clear(tile[i]); }
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
    tile[0] = seeds[0];
    int2 v = int2(4, 5);
    int2 r = fromComp(v);
    clearTile(0);
    outBuf[0] = r.x * 1000 + r.y;
    outBuf[1] = v.y + tile[0];
}
"""

FRAGMENT = """shader S {
    struct FIn { vec4 color; };
    struct FOut { vec4 color; };
    fragment {
        float halve(inout float h) { h = h * 0.5; return h + 1.0; }
        FOut main(FIn input) {
            vec4 c = input.color;
            return FOut(vec4(halve(c.y), 0.0, 0.0, 1.0));
        }
    }
}
"""

POINTER = """shader S {
    compute {
        layout(local_size_x = 1) in;
        shared int tile[4];
        int bump(inout int t) { t = t + 10; return t * 2; }
        int* pick(int* p, int k) { return p; }
        int* fromTile(int* p) { return pick(p, bump(tile[0])); }
        void main() { int x = 0; int* q = fromTile(&x); }
    }
}
"""


def _translate(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _lines(code):
    return [line.strip() for line in code.splitlines() if line.strip()]


def _function(code, header):
    """The lines of the function whose signature starts with ``header``."""
    lines = code.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return [line.strip() for line in lines[start : end + 1] if line.strip()]


def test_returned_call_stores_threadgroup_argument_before_returning(tmp_path):
    body = _function(_translate(tmp_path, THREADGROUP), "int fromTile(")
    assert body[1:5] == [
        "int __crossgl_ref2 = tile[i];",
        "int __crossgl_return1 = bump(__crossgl_ref2);",
        "tile[i] = __crossgl_ref2;",
        "return __crossgl_return1;",
    ]


def test_returned_value_local_has_the_function_return_type(tmp_path):
    body = _function(_translate(tmp_path, THREADGROUP), "float fromTileF(")
    assert "float __crossgl_return3 = widen(__crossgl_ref4);" in body
    assert body.index("tile[i] = __crossgl_ref4;") < body.index(
        "return __crossgl_return3;"
    )


def test_tail_expression_stores_argument_before_returning(tmp_path):
    body = _function(_translate(tmp_path, THREADGROUP), "int viaTail(")
    store = body.index("tile[i] = __crossgl_ref6;")
    assert body[store + 1] == "return __crossgl_return5;"
    assert "return bump(__crossgl_ref6);" not in body


def test_returns_in_branches_and_cases_store_before_returning(tmp_path):
    body = _function(_translate(tmp_path, THREADGROUP), "int branchy(")
    returns = [index for index, line in enumerate(body) if line.startswith("return _")]
    assert len(returns) == 3
    for index in returns:
        assert (
            body[index - 1].startswith("tile[") and "__crossgl_ref" in body[index - 1]
        )


def test_hlsl_vector_component_and_void_return(tmp_path):
    code = _translate(tmp_path, HLSL, "shader.hlsl")
    # The call is evaluated, and ``v.y`` copied back, before the constructor reads ``v``.
    assert _function(code, "int2 fromComp(")[1:6] == [
        "int __crossgl_ref3 = v.y;",
        "int __crossgl_call2 = bump(__crossgl_ref3);",
        "v.y = __crossgl_ref3;",
        "int2 __crossgl_return1 = int2(__crossgl_call2, v.x);",
        "return __crossgl_return1;",
    ]
    assert _function(code, "void clearTile(")[1:5] == [
        "int __crossgl_ref4;",
        "clear(__crossgl_ref4);",
        "tile[i] = __crossgl_ref4;",
        "return;",
    ]


def test_entry_point_return_goes_through_the_stage_output(tmp_path):
    lines = _lines(_translate(tmp_path, FRAGMENT))
    store = lines.index("c.y = __crossgl_ref2;")
    assert lines[store - 1] == (
        "FOut __crossgl_return1 = FOut{float4(halve(__crossgl_ref2), 0.0, 0.0, 1.0)};"
    )
    assert lines[store + 1] == "return __crossgl_return1;"


def test_returned_pointer_with_a_copied_argument_is_refused(tmp_path):
    with pytest.raises(ValueError, match="copy a reference argument back"):
        _translate(tmp_path, POINTER)


def _metal_available():
    if shutil.which("xcrun") is None:
        return False
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    return found.returncode == 0


@pytest.mark.parametrize(
    "source, name",
    [(THREADGROUP, "shader.cgl"), (HLSL, "shader.hlsl"), (FRAGMENT, "shader.cgl")],
    ids=["threadgroup", "hlsl", "fragment"],
)
def test_return_copies_compile(tmp_path, source, name):
    code = _translate(tmp_path, source, name)
    if not _metal_available():
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
