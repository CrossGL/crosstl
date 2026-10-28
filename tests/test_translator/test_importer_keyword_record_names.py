"""The CUDA, HIP and OpenCL C importers rename a struct named after a CrossGL keyword (#3386).

``let``, ``var``, ``match`` and ``kernel`` are CrossGL lexer keywords. The HLSL importer already
renames a struct with one of those names (``struct match`` becomes ``struct match_``). The CUDA,
HIP and OpenCL C importers kept the name, so the CrossGL they wrote did not parse
(``SyntaxError: Expected binding name, got LBRACE``). The CUDA importer renamed the declaration
of a ``kernel`` or ``var`` struct but not the places that use it as a type.

A rename skips every name the source already uses, so ``let`` becomes ``let__`` where ``let_``
is taken.
"""

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

KEYWORDS = ("let", "var", "match", "kernel")

CUDA = """\
struct {name} {{
    float a;
}};

__device__ float get({name} value) {{
    return value.a;
}}

__global__ void probe(float* outp) {{
    {name} t;
    t.a = 2.0f;
    outp[threadIdx.x] = get(t);
}}
"""

OPENCL = """\
struct {name} {{
    float a;
}};

float get(struct {name} value) {{
    return value.a;
}}

__kernel void probe(__global float* outp) {{
    struct {name} t;
    t.a = 2.0f;
    outp[get_global_id(0)] = get(t);
}}
"""

CASES = [
    *(pytest.param("probe.cu", CUDA, name, id=f"cu-{name}") for name in KEYWORDS),
    # HIP host code names launch handles ``kernel``, so the HIP importer keeps that name.
    *(
        pytest.param("probe.hip", CUDA, name, id=f"hip-{name}")
        for name in KEYWORDS
        if name != "kernel"
    ),
    # ``kernel`` is an OpenCL C keyword, so OpenCL C cannot name a struct ``kernel``.
    *(
        pytest.param("probe.cl", OPENCL, name, id=f"cl-{name}")
        for name in KEYWORDS
        if name != "kernel"
    ),
]


def _translate(tmp_path, file_name, source, backend="cgl"):
    path = tmp_path / file_name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize(("file_name", "template", "name"), CASES)
def test_importer_renames_keyword_struct(tmp_path, file_name, template, name):
    code = _translate(tmp_path, file_name, template.format(name=name))

    assert f"struct {name}_ {{" in code
    assert f"{name}_ value" in code
    assert f"var t: {name}_;" in code
    assert f"struct {name} {{" not in code
    Parser(Lexer(code).tokens).parse()


@pytest.mark.parametrize(("file_name", "template", "name"), CASES)
def test_keyword_struct_translates_to_metal(tmp_path, file_name, template, name):
    code = _translate(tmp_path, file_name, template.format(name=name), backend="metal")

    assert f"struct {name}_" in code


SHADOW = """\
__global__ void probe(float* outp) {{
    float {name}_ = 1.0f;
    {{
        float {name} = 2.0f;
        {name}_ = {name}_ + {name};
    }}
    outp[threadIdx.x] = {name}_;
}}
"""


def _cuda_and_hip(template):
    return [
        *(
            pytest.param("probe.cu", template, name, id=f"cu-{name}")
            for name in KEYWORDS
        ),
        *(
            pytest.param("probe.hip", template, name, id=f"hip-{name}")
            for name in KEYWORDS
            if name != "kernel"
        ),
    ]


@pytest.mark.parametrize(("file_name", "template", "name"), _cuda_and_hip(SHADOW))
def test_rename_skips_names_the_source_uses(tmp_path, file_name, template, name):
    code = _translate(tmp_path, file_name, template.format(name=name))

    # ``let_`` is taken, so the inner ``let`` becomes ``let__`` rather than hide it.
    assert f"var {name}__: f32 = 2.0f;" in code
    assert f"{name}_ = ({name}_ + {name}__);" in code
    Parser(Lexer(code).tokens).parse()
