"""Function-scope ``static`` variables keep their value across calls.

``int next() { static int c = 0; return ++c; }`` returns 1, 2, 3, ... within an invocation. Every generator used
to emit the variable as an ordinary local, so the function returned 1 on every call.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import StaticLocalUnsupportedError
from crosstl.translator.semantics.static_locals import hoist_static_locals

COUNTER = """
RWStructuredBuffer<int> outBuf : register(u0);
int counter() {
    static int c = 0;
    c++;
    return c;
}
float3 accumulate(float3 v) {
    static float3 total;
    total += v;
    return total;
}
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int a = counter();
    int b = counter();
    float3 t = accumulate(float3(1.0, 2.0, 3.0));
    t = accumulate(t);
    outBuf[id.x] = a * 10 + b + int(t.z);
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


def test_glsl_keeps_the_counter_in_a_global(tmp_path):
    code = _translate(tmp_path, COUNTER, "opengl")
    flat = _flat(code)
    assert "int counter_c = 0;" in flat, code
    assert "vec3 accumulate_total = vec3(0.0);" in flat, code
    assert "int counter() { (counter_c++); return counter_c; }" in flat, code
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    (tmp_path / "out.comp").write_text(code)
    _run([validator, "out.comp"], tmp_path)


def test_wgsl_keeps_the_counter_in_a_private_variable(tmp_path):
    code = _translate(tmp_path, COUNTER, "wgsl")
    flat = _flat(code)
    assert "var<private> counter_c: i32 = 0;" in flat, code
    assert "var<private> accumulate_total: vec3<f32> = vec3<f32>(0.0);" in flat, code
    naga = shutil.which("naga")
    if naga is None:
        return
    (tmp_path / "out.wgsl").write_text(code)
    _run([naga, "out.wgsl"], tmp_path)


def test_metal_passes_the_counter_to_the_function(tmp_path):
    code = _translate(tmp_path, COUNTER, "metal")
    # clang-format binds the reference to the name; without it the generator writes ``int& name``.
    flat = re.sub(r"(\w)& (\w)", r"\1 &\2", _flat(code))
    assert "int counter(thread int &counter_c) {" in flat, code
    assert "int a = counter(counter_c); int b = counter(counter_c);" in flat, code
    if shutil.which("xcrun") is None:
        return
    (tmp_path / "out.metal").write_text(code)
    _run(["xcrun", "metal", "-c", "out.metal", "-o", "out.air"], tmp_path)


def test_spirv_keeps_the_counter_in_private_storage(tmp_path):
    code = _translate(tmp_path, COUNTER, "vulkan")
    # The ids are symbolic when spirv-tools reformat the module and numeric otherwise, so resolve them by name.
    name = re.search(r'OpName (%\S+) "counter_c"', code)
    assert name is not None, code
    variable = re.search(
        rf"^\s*{re.escape(name.group(1))} = OpVariable (%\S+) Private\b",
        code,
        re.MULTILINE,
    )
    assert variable is not None, code
    pointer = rf"^\s*{re.escape(variable.group(1))} = OpTypePointer Private (%\S+)$"
    pointee = re.search(pointer, code, re.MULTILINE)
    assert pointee is not None, code
    assert re.search(
        rf"^\s*{re.escape(pointee.group(1))} = OpTypeInt 32 1$", code, re.MULTILINE
    ), code
    assembler = shutil.which("spirv-as")
    validator = shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    (tmp_path / "out.spvasm").write_text(code)
    _run(
        [assembler, "out.spvasm", "-o", "out.spv", "--target-env", "vulkan1.1"],
        tmp_path,
    )
    _run([validator, "out.spv"], tmp_path)


def test_hlsl_keeps_a_static_global(tmp_path):
    flat = _flat(_translate(tmp_path, COUNTER, "directx"))
    assert "static int counter_c = 0;" in flat, flat
    assert "return counter_c;" in flat, flat


SCOPES = """
shader S {
    int counter_c;
    int counter(int p) {
        int before = c;
        static int c = 1;
        {
            int c = p;
            before += c;
        }
        for (int c = 0; c < 2; c++) {
            before += c;
        }
        c += before;
        return c;
    }
    int c;
    compute {
        void main() {
            int x = counter(3);
        }
    }
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _identifiers(node, found):
    from crosstl.translator.semantics.rewrite import child_slots, get_child

    if type(node).__name__ == "IdentifierNode":
        found.append(node.name)
    for field, key in child_slots(node):
        _identifiers(get_child(node, field, key), found)
    return found


def test_only_uses_in_the_declarations_scope_are_renamed():
    program = _parse(SCOPES)
    assert hoist_static_locals(program, "wgsl") == 1
    hoisted = program.global_variables[-1]
    # counter_c is already taken by a global.
    assert hoisted.name == "counter_c_1"
    function = next(f for f in program.functions if f.name == "counter")
    names = _identifiers(function.body, [])
    # before = c (the global); the block's c = p and before += c; the loop's c < 2, c++ and
    # before += c; then c += before and return c on the static.
    assert names == [
        "c",
        "p",
        "before",
        "c",
        "c",
        "c",
        "before",
        "c",
        "counter_c_1",
        "before",
        "counter_c_1",
    ], names


@pytest.mark.parametrize(
    "declaration, message",
    [
        ("static int c = p;", "not a constant expression"),
        ("static int c = counter2();", "not a constant expression"),
        ("static Pair c;", "is not a scalar, vector or matrix"),
    ],
)
def test_static_locals_that_cannot_move_are_refused(tmp_path, declaration, message):
    source = f"""
    struct Pair {{ int a; int b; }};
    RWStructuredBuffer<int> outBuf : register(u0);
    int counter2() {{ return 1; }}
    int counter(int p) {{ {declaration} return 0; }}
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {{ outBuf[0] = counter(1); }}
    """
    with pytest.raises(StaticLocalUnsupportedError, match=message):
        _translate(tmp_path, source, "wgsl")


def test_static_const_local_stays_a_local(tmp_path):
    source = """
    RWStructuredBuffer<int> outBuf : register(u0);
    int scale(int v) { static const int K = 3; return v * K; }
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) { outBuf[0] = scale(2); }
    """
    flat = _flat(_translate(tmp_path, source, "wgsl"))
    assert "scale_K" not in flat, flat


def test_hlsl_static_global_keeps_its_initializer(tmp_path):
    """HLSL starts a static global without an initializer from zero, so dropping ``= 5`` changed the value."""
    source = """
    RWStructuredBuffer<int> outBuf : register(u0);
    static int g = 5;
    static float2 h = float2(1.0, 2.0);
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) { g += 1; outBuf[0] = g + int(h.y); }
    """
    flat = _flat(_translate(tmp_path, source, "directx"))
    assert "static int g = 5;" in flat, flat
    assert "static float2 h = float2(1.0, 2.0);" in flat, flat
