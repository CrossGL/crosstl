"""Slang helper parameters keep their direction in CrossGL (issue #3215).

The importer wrote ``in``, ``out`` and ``inout`` only for entry points, so a helper's ``inout int k`` became the
by-value ``int k`` and the callee's writes never reached the caller. ``out`` and ``inout`` now reach CrossGL for
every function. ``constref`` is a read-only reference, which is the value it reads. A ``ref`` parameter names the
caller's storage, which CrossGL's copy-in/copy-out ``inout`` does not, so it is refused.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator as cgl_translator
from crosstl.backend.import_refusals import SlangImportRefused
from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser

DIRECTIONS = """
RWStructuredBuffer<int> data;

int bump(inout int k) {
    k = k + 1;
    return k;
}

void pair(out int2 v, in int a) { v = int2(a, a + 1); }

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int k = data[id.x];
    int r = bump(k);
    int2 w;
    pair(w, r);
    data[id.x] = r + k + w.y;
}
"""


def to_crossgl(code):
    ast = SlangParser(SlangLexer(code).tokenize()).parse()
    return SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)


def _compact(code):
    return " ".join(code.split())


def test_helper_parameters_keep_out_and_inout():
    code = to_crossgl(DIRECTIONS)
    compact = _compact(code)
    assert "int bump(inout int k)" in compact
    assert re.search(r"void pair\(out ivec2 v, int a\)", compact)
    cgl_translator.parse(code)


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal"])
def test_directions_reach_the_targets(tmp_path, backend):
    source = tmp_path / "directions.slang"
    source.write_text(DIRECTIONS)
    code = crosstl.translate(str(source), backend=backend, format_output=False)
    compact = _compact(code)
    if backend == "metal":
        assert re.search(r"int bump\(thread int ?& ?k\)", compact)
    else:
        assert "int bump(inout int k)" in compact
        assert re.search(r"void pair\(out (ivec2|int2) v, int a\)", compact)
    if backend == "opengl" and shutil.which("glslangValidator"):
        shader = tmp_path / "directions.comp"
        shader.write_text(code)
        result = subprocess.run(
            ["glslangValidator", str(shader)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("modifier", ["constref", "__constref"])
def test_constref_parameters_read_by_value(modifier):
    code = to_crossgl(f"""
    RWStructuredBuffer<int> data;
    int twice({modifier} int k) {{ return k * 2; }}
    [shader("compute")]
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {{ data[id.x] = twice(data[id.x]); }}
    """)
    assert "int twice(int k)" in _compact(code)
    cgl_translator.parse(code)


@pytest.mark.parametrize("modifier", ["ref", "__ref"])
def test_ref_parameters_are_refused(modifier):
    with pytest.raises(SlangImportRefused) as error:
        to_crossgl(f"""
        RWStructuredBuffer<int> data;
        void bump({modifier} int k) {{ k = k + 1; }}
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {{
            int k = data[id.x];
            bump(k);
            data[id.x] = k;
        }}
        """)
    assert "ref parameter k" in str(error.value)


def test_a_parameter_named_ref_is_still_a_name():
    code = to_crossgl("""
    RWStructuredBuffer<float> data;
    float scale(float ref, float x) { return ref * x; }
    [shader("compute")]
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) { data[id.x] = scale(2.0, data[id.x]); }
    """)
    # CrossGL reserves ref, so the importer renames the parameter.
    assert "float scale(float ref_, float x) { return ref_ * x; }" in _compact(code)
