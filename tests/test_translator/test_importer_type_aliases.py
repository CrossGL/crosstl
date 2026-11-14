"""Importers carry ``typedef``, ``using`` and ``typealias`` into CrossGL as ``typedef`` (#3309).

CrossGL reads only the C spelling ``typedef float real;`` as an alias. Before this, the HLSL importer wrote
``type real = float;`` (a variable of type ``type``), the Slang importer wrote ``typealias Couple = Pair;`` verbatim,
the CUDA importer wrote the alias as a comment, and the HIP importer erased the alias but kept ``real(x)`` as a call
of an undeclared function. Every target then refused the program as using an undeclared type, or (HIP into Metal)
reported success for code that calls ``real``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import DirectXImportRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

HLSL = """\
RWStructuredBuffer<float> outp : register(u0);
typedef float real;

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    real x = real(outp[id.x]);
    outp[id.x] = x * 2.0;
}
"""

SLANG = """\
RWStructuredBuffer<float> outp;
struct Pair { float a; float b; };
typealias Couple = Pair;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    Couple c;
    c.a = outp[id.x];
    c.b = 2.0;
    outp[id.x] = c.a * c.b;
}
"""

CUDA = """\
typedef float real;

__global__ void probe(float* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = real(outp[i]);
    outp[i] = x * 2.0f;
}
"""

REPRODUCERS = {
    "td.hlsl": HLSL,
    "td.slang": SLANG,
    "td.cu": CUDA,
    "td.hip": CUDA,
}


def _translate(tmp_path, name, source, backend="cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _body(code, signature):
    """Return the lines of ``code`` from the line holding ``signature`` on."""
    lines = code.splitlines()
    start = next(index for index, line in enumerate(lines) if signature in line)
    return [line.strip() for line in lines[start:]]


@pytest.mark.parametrize(
    ("name", "present", "absent"),
    [
        (
            "td.hlsl",
            ["typedef float real;", "real x = real(outp[id.x]);"],
            ["type real"],
        ),
        ("td.slang", ["typedef Pair Couple;", "Couple c;"], ["typealias"]),
        ("td.cu", ["typedef f32 real;", "var x: real = real(outp[i]);"], ["//"]),
        # HIP erases a concrete alias, so the cast through it names the type too.
        ("td.hip", ["var x: f32 = f32(outp[i]);"], ["real"]),
    ],
)
def test_importer_writes_the_alias_crossgl_reads(tmp_path, name, present, absent):
    code = _translate(tmp_path, name, REPRODUCERS[name])

    for text in present:
        assert text in code
    alias_lines = [
        line for line in code.splitlines() if "real" in line or "Couple" in line
    ]
    for text in absent:
        assert all(text not in line for line in alias_lines), alias_lines
    Parser(Lexer(code).tokens).parse()


@pytest.mark.parametrize("name", sorted(REPRODUCERS))
def test_aliased_program_translates_to_metal(tmp_path, name):
    code = _translate(tmp_path, name, REPRODUCERS[name], backend="metal")

    assert "undeclared" not in code
    assert "real(" not in code or "typedef float real;" in code


def test_hlsl_block_scope_aliases_are_statements(tmp_path):
    code = _translate(
        tmp_path,
        "block.hlsl",
        """\
RWStructuredBuffer<float> outp : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    typedef float2 pair;
    using count = uint;
    pair p = pair(outp[id.x], 2.0);
    count n = (count)p.y;
    outp[id.x] = p.x * p.y + n;
}
""",
    )

    body = _body(code, "void main(")
    assert body[1:5] == [
        "typedef vec2 pair;",
        "typedef uint count;",
        "pair p = pair(outp[id.x], 2.0);",
        "count n = count(p.y);",
    ]
    Parser(Lexer(code).tokens).parse()


def test_hlsl_long_vector_alias_is_refused_not_narrowed(tmp_path):
    # vector<half, 16> used to become vec4, which the alias would now carry into every use.
    with pytest.raises(
        DirectXImportRefused, match=r"vector<half, 16> has 16 components"
    ):
        _translate(
            tmp_path,
            "long.hlsl",
            """\
RWStructuredBuffer<float> outp : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    typedef vector<half, 16> half16;
    half16 lanes;
    outp[id.x] = lanes[15];
}
""",
        )


def test_slang_block_scope_aliases_are_statements(tmp_path):
    code = _translate(
        tmp_path,
        "block.slang",
        """\
RWStructuredBuffer<float> outp;
struct Pair { float a; float b; };

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    typealias GVec = vector<float, 2>;
    typedef Pair P2;
    GVec g = GVec(outp[id.x], 2.0);
    P2 d;
    d.a = g.x;
    outp[id.x] = d.a * g.y;
}
""",
    )

    body = _body(code, "void main(")
    assert body[1:3] == ["typedef vec2 GVec;", "typedef Pair P2;"]
    Parser(Lexer(code).tokens).parse()


def test_cuda_aliases_of_records_name_the_record(tmp_path):
    code = _translate(
        tmp_path,
        "records.cu",
        """\
struct Pair { float a; float b; };
typedef struct Pair PairT;
typedef struct Q { int v; } Q;
using idx = int;

__global__ void probe(float* outp) {
    typedef float2 duo;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    duo d = make_float2(outp[i], 1.0f);
    PairT p;
    p.a = d.x;
    idx k = (idx)d.y;
    outp[i] = p.a + k;
}
""",
    )

    assert "typedef Pair PairT;" in code
    assert "typedef i32 idx;" in code
    assert "    typedef vec2<f32> duo;" in code
    # ``typedef struct Q {...} Q;`` declares the record and no alias.
    assert "Q Q" not in code
    assert "// typedef" not in code
    Parser(Lexer(code).tokens).parse()


def _metal_compiler_available():
    return (
        shutil.which("xcrun") is not None
        and subprocess.run(
            ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
        == 0
    )


@pytest.mark.skipif(
    not _metal_compiler_available(), reason="the Metal compiler is not installed"
)
@pytest.mark.parametrize("name", sorted(REPRODUCERS))
def test_aliased_program_compiles_as_metal(tmp_path, name):
    path = tmp_path / "kernel.metal"
    path.write_text(_translate(tmp_path, name, REPRODUCERS[name], backend="metal"))
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", "/dev/null"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
