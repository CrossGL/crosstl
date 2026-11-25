"""Slang declarations the Slang import refused or wrote as CrossGL that does not parse.

Each case was a refused row of the Slang input coverage (``multi-variable-loop``, ``array-return``,
``reserved-type-name`` and ``generic-function``): a ``for`` that declares two loop variables with one type, a C-style
array ``typedef``, a struct named by a CrossGL keyword (``match``), and a generic function, whose generic parameters
the import dropped. Each CrossGL program is generated back to Slang and compiled with slangc when it is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator as cgl_translator
from crosstl.backend.import_refusals import SlangImportRefused


def to_crossgl(tmp_path, source):
    path = tmp_path / "probe.slang"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="cgl")


def to_target(tmp_path, crossgl, backend):
    path = tmp_path / "round.cgl"
    path.write_text(crossgl, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def compile_slang(tmp_path, code):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source, binary = tmp_path / "out.slang", tmp_path / "out.spv"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [slangc, str(source), "-target", "spirv", "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def round_trip(tmp_path, source):
    """The CrossGL import of ``source``, after checking that it parses and its Slang output compiles."""
    crossgl = to_crossgl(tmp_path, source)
    cgl_translator.parse(crossgl)
    compile_slang(tmp_path, to_target(tmp_path, crossgl, "slang"))
    return crossgl


def squash(code):
    return re.sub(r"\s+", " ", code)


COMPUTE = """RWStructuredBuffer<float> fin;
RWStructuredBuffer<int> iin;
RWStructuredBuffer<float> outp;
"""


def entry(body):
    return (
        '[shader("compute")]\n[numthreads(1, 1, 1)]\n'
        "void main(uint3 id : SV_DispatchThreadID) {\n    uint i = id.x;\n"
        f"{body}\n}}\n"
    )


def test_for_init_declares_both_loop_variables_with_one_type(tmp_path):
    source = COMPUTE + entry("""    float acc = 0.0;
    int n = 6 + (iin[i] & 3);
    for (int a = 0, b = n; a < b; a++, b--) {
        acc += float(a * 16 + b);
    }
    outp[i] = acc;""")
    crossgl = round_trip(tmp_path, source)

    # A second ``int`` in the list is a syntax error in CrossGL, as in C.
    assert "for (int a = 0, b = n; a < b; a++, b--)" in crossgl


def test_for_init_keeps_a_type_for_each_declarator_of_another_type(tmp_path):
    source = COMPUTE + entry("""    float acc = 0.0;
    for (int a = 0, b = 4; a < b; a++) {
        acc += float(a + b);
    }
    outp[i] = acc;""")
    crossgl = round_trip(tmp_path, source)

    assert "for (int a = 0, b = 4;" in crossgl


def test_c_style_array_typedef_is_an_array_type(tmp_path):
    source = (
        COMPUTE
        + """typedef float Triple[3];
Triple spread(float x) {
    Triple t = { x, x * 2.0, x * 3.0 };
    return t;
}
"""
        + entry("""    Triple t = spread(fin[i]);
    outp[i] = t[0] + t[1] + t[2];""")
    )
    crossgl = round_trip(tmp_path, source)

    assert "typedef float[3] Triple;" in crossgl


def test_struct_named_by_a_crossgl_keyword_is_renamed(tmp_path):
    source = (
        COMPUTE
        + """struct match {
    float a;
    int b;
};
match make(float x) {
    match t;
    t.a = x * 2.0;
    t.b = 3;
    return t;
}
float weigh(match t) {
    return t.a + float(t.b);
}
"""
        + entry("""    match t = make(float(iin[i] & 7));
    outp[i] = weigh(t);""")
    )
    crossgl = squash(round_trip(tmp_path, source))

    assert "struct match_ {" in crossgl
    assert "match_ make(float x)" in crossgl
    assert "float weigh(match_ t)" in crossgl
    assert "match_ t = make(" in crossgl
    assert re.search(r"\bmatch\b", crossgl) is None


def test_struct_rename_avoids_another_struct_name(tmp_path):
    source = (
        COMPUTE
        + """struct match_ { float a; };
struct match { float b; };
"""
        + entry("""    match m;
    m.b = fin[i];
    match_ n;
    n.a = 1.0;
    outp[i] = m.b + n.a;""")
    )
    crossgl = squash(round_trip(tmp_path, source))

    assert "struct match_ { float a; }" in crossgl
    assert "struct match__ { float b; }" in crossgl
    assert "match__ m;" in crossgl


def test_generic_function_keeps_its_type_parameter(tmp_path):
    source = (
        COMPUTE
        + """T twice<T : IArithmetic>(T x) {
    return x + x;
}
"""
        + entry("""    float x = float(iin[i]);
    int k = iin[i] & 7;
    float2 w = twice(float2(x, 1.0));
    outp[i] = twice(x) + float(twice(k)) * 64.0 + w.y;""")
    )
    crossgl = round_trip(tmp_path, source)

    assert "T twice<T: IArithmetic>(T x)" in crossgl
    # Every target specializes the generic for each argument type.
    spirv = to_target(tmp_path, crossgl, "vulkan")
    assert spirv.count("OpFunction ") >= 4


def test_generic_prefix_and_value_parameters(tmp_path):
    source = (
        COMPUTE
        + """__generic<T : IFloat & IArithmetic>
T scaled(T x) {
    return x * T(2.0);
}
float repeat<let N : int>(float x) {
    float total = 0.0;
    for (int k = 0; k < N; k++) {
        total += x;
    }
    return total;
}
float widen<T, int M>(T x) {
    return float(M);
}
"""
        + entry(
            """    outp[i] = scaled(fin[i]) + repeat<3>(fin[i]) + widen<float, 4>(fin[i]);"""
        )
    )
    crossgl = round_trip(tmp_path, source)

    assert "T scaled<T: IFloat + IArithmetic>(T x)" in crossgl
    assert "float repeat<N>(float x)" in crossgl
    assert "float widen<T, M>(T x)" in crossgl


@pytest.mark.parametrize(
    "declaration,reason",
    [
        (
            "int count<each T>(expand each T terms) where T == int { return 0; }",
            "variadic generic parameter 'each T' of function 'count'",
        ),
        (
            "float repeat<let N : int = 2>(float x) { return x * float(N); }",
            "default value of generic parameter 'let N:int=2' of function 'repeat'",
        ),
    ],
)
def test_generic_parameters_crossgl_cannot_declare_are_refused(
    tmp_path, declaration, reason
):
    source = COMPUTE + declaration + "\n" + entry("    outp[i] = fin[i];")
    with pytest.raises(SlangImportRefused, match=re.escape(reason)):
        to_crossgl(tmp_path, source)
