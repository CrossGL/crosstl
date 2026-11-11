"""Metal evaluates a call whose copied reference arguments the rest of its statement observes first.

CrossGL copies an ``inout`` argument back when the called function returns (OpenGL Shading Language 4.60, section
6.1.1), so later operands, later calls and the store of the enclosing assignment see the new value. Metal binds
threadgroup or device storage and vector components through a ``thread`` temporary that is stored back after the
statement; such a call is therefore evaluated into a local of its own declaration first.
"""

import shutil
import subprocess

import pytest

import crosstl

ORDER = """shader S {
    compute {
        layout(std430, binding = 0) buffer Out { int values[]; } outBuf;
        layout(local_size_x = 4) in;
        shared int tile[4];
        int bump(inout int t) { t = t + 10; return t * 2; }
        int next(inout int k) { k = k + 1; return k; }
        int pair(int first, inout int second) { second = second + first; return second * 2; }
        void main() {
            uint l = gl_LocalInvocationIndex;
            int k = 0;
            tile[l] = 1;
            tile[l] = bump(tile[l]);
            ivec2 v = ivec2(3, 4);
            ivec2 r = ivec2(bump(v.y), v.y);
            int w = tile[l] * 10 + bump(tile[l]) + tile[l];
            tile[next(k)] = bump(tile[k]);
            int z = bump(v.x) - bump(v.x);
            int p = pair(bump(tile[l]), tile[l]);
            int arr[2] = {bump(tile[l]), tile[l]};
            int u = bump(tile[0]);
            outBuf.values[l] = r.x + w + z + p + arr[1] + u;
        }
    }
}
"""

HLSL = """RWStructuredBuffer<int> outBuf : register(u0);
groupshared int tile[4];
int bump(inout int t) { t = t + 10; return t * 2; }
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
    uint l = tid.x;
    tile[l] = 3;
    tile[l] = bump(tile[l]);
    int2 v = int2(4, 5);
    v.y += bump(v.y);
    outBuf[l] = tile[l] + v.y;
}
"""

CONDITIONAL = """shader S {
    compute {
        layout(local_size_x = 4) in;
        shared int tile[4];
        int bump(inout int t) { t = t + 10; return t * 2; }
        void main() {
            uint l = gl_LocalInvocationIndex;
            bool c = l > 1u;
            int q = c ? bump(tile[l]) : tile[l];
        }
    }
}
"""


def _translate(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _kernel(code):
    lines = code.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("kernel void"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return [line.strip() for line in lines[start + 1 : end] if line.strip()]


def _after(body, line, count):
    index = body.index(line)
    return body[index : index + count]


@pytest.fixture(scope="module")
def order_body(tmp_path_factory):
    return _kernel(_translate(tmp_path_factory.mktemp("order"), ORDER))


def test_assignment_stores_the_call_result_after_the_copy_back(order_body):
    assert _after(order_body, "int __crossgl_ref2 = tile[l];", 4) == [
        "int __crossgl_ref2 = tile[l];",
        "int __crossgl_call1 = bump(__crossgl_ref2);",
        "tile[l] = __crossgl_ref2;",
        "tile[l] = __crossgl_call1;",
    ]


def test_later_operand_reads_the_copied_back_component(order_body):
    assert _after(order_body, "int __crossgl_ref4 = v.y;", 4) == [
        "int __crossgl_ref4 = v.y;",
        "int __crossgl_call3 = bump(__crossgl_ref4);",
        "v.y = __crossgl_ref4;",
        "int2 r = int2(__crossgl_call3, v.y);",
    ]


def test_earlier_operand_is_evaluated_before_the_call(order_body):
    assert _after(order_body, "int __crossgl_arg5 = tile[l] * 10;", 5) == [
        "int __crossgl_arg5 = tile[l] * 10;",
        "int __crossgl_ref7 = tile[l];",
        "int __crossgl_call6 = bump(__crossgl_ref7);",
        "tile[l] = __crossgl_ref7;",
        "int w = __crossgl_arg5 + __crossgl_call6 + tile[l];",
    ]


def test_assignment_target_index_is_located_before_the_value(order_body):
    assert _after(order_body, "int __crossgl_arg8 = next(k);", 5) == [
        "int __crossgl_arg8 = next(k);",
        "int __crossgl_ref10 = tile[k];",
        "int __crossgl_call9 = bump(__crossgl_ref10);",
        "tile[k] = __crossgl_ref10;",
        "tile[__crossgl_arg8] = __crossgl_call9;",
    ]


def test_second_call_copies_in_the_value_the_first_call_stored(order_body):
    assert _after(order_body, "int __crossgl_ref12 = v.x;", 6) == [
        "int __crossgl_ref12 = v.x;",
        "int __crossgl_call11 = bump(__crossgl_ref12);",
        "v.x = __crossgl_ref12;",
        "int __crossgl_ref13 = v.x;",
        "int z = __crossgl_call11 - bump(__crossgl_ref13);",
        "v.x = __crossgl_ref13;",
    ]


def test_later_argument_is_copied_in_after_the_earlier_call(order_body):
    assert _after(order_body, "int __crossgl_ref15 = tile[l];", 6) == [
        "int __crossgl_ref15 = tile[l];",
        "int __crossgl_call14 = bump(__crossgl_ref15);",
        "tile[l] = __crossgl_ref15;",
        "int __crossgl_ref16 = tile[l];",
        "int p = pair(__crossgl_call14, __crossgl_ref16);",
        "tile[l] = __crossgl_ref16;",
    ]


def test_braced_initializer_element_reads_the_copied_back_value(order_body):
    assert _after(order_body, "int __crossgl_ref18 = tile[l];", 4)[-1] == (
        "int arr[2] = {__crossgl_call17, tile[l]};"
    )


def test_call_nothing_else_observes_keeps_its_place(order_body):
    assert _after(order_body, "int __crossgl_ref19 = tile[0];", 3) == [
        "int __crossgl_ref19 = tile[0];",
        "int u = bump(__crossgl_ref19);",
        "tile[0] = __crossgl_ref19;",
    ]


def test_hlsl_groupshared_store_and_compound_component(tmp_path):
    body = _kernel(_translate(tmp_path, HLSL, "shader.hlsl"))
    store = body.index("tile[l] = __crossgl_ref2;")
    assert body[store - 1] == "int __crossgl_call1 = bump(__crossgl_ref2);"
    assert body[store + 1] == "tile[l] = __crossgl_call1;"
    # DXC reads the target of a compound assignment after its value (C++17 [expr.ass]/1).
    copy_back = body.index("v.y = __crossgl_ref3;")
    assert body[copy_back - 1].endswith(" = bump(__crossgl_ref3);")
    assert body[copy_back + 1].startswith("v.y = as_type<int>(as_type<uint>(v.y) + ")


def test_conditionally_evaluated_copy_is_made_in_place(tmp_path):
    body = _kernel(_translate(tmp_path, CONDITIONAL))
    assert (
        "__attribute__((unused)) int q = c ? (__crossgl_ref1 = tile[l], "
        "__crossgl_call2 = bump(__crossgl_ref1), tile[l] = __crossgl_ref1, "
        "__crossgl_call2) : tile[l];"
    ) in body, body


def _metal_available():
    if shutil.which("xcrun") is None:
        return False
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    return found.returncode == 0


@pytest.mark.parametrize(
    "source, name",
    [(ORDER, "shader.cgl"), (HLSL, "shader.hlsl")],
    ids=["order", "hlsl"],
)
def test_call_results_compile(tmp_path, source, name):
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
