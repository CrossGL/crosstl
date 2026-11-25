"""Methods, static members and overloads of structs declared in namespaces, through Metal input.

A receiver declared with a namespace-qualified type (``ops::ReduceOp op; op();``) binds to the struct's lowered
methods like an unqualified one does, and structs of one name declared in several namespaces (``ops::P`` and
``alt::P``) keep distinct identities: each gets a program-scope name of its own, every reference binds to the struct
C++ name lookup selects, and the translation computes on the GPU what the source computes. A reference that C++
finds ambiguous still fails closed.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.struct_homonyms import (
    AmbiguousStructNameError,
    distinct_struct_homonyms,
)

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

QUALIFIED_RECEIVER = PREFIX + """
namespace ops {
struct ReduceOp {
    static constexpr constant int value = 7;
    int operator()() const { return *(&ReduceOp::value); }
    int twice(int x) const { return 2 * x + value; }
};
}
kernel void k(device int* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    ops::ReduceOp a;
    ::ops::ReduceOp b;
    out[3 * i] = a();
    out[3 * i + 1] = a.twice(int(i));
    out[3 * i + 2] = b.twice(1);
}
"""

USING_DIRECTIVE_RECEIVER = PREFIX + """
namespace ops {
struct ReduceOp {
    static constexpr constant int value = 7;
    int operator()() const { return *(&ReduceOp::value); }
    int twice(int x) const { return 2 * x + value; }
};
}
using namespace ops;
kernel void k(device int* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    ReduceOp a;
    out[3 * i] = a();
    out[3 * i + 1] = a.twice(int(i));
    out[3 * i + 2] = 0;
}
"""

HOMONYM_METHODS = PREFIX + """
namespace ops {
struct ReduceOp {
    static constexpr constant int value = 7;
    int operator()() const { return *(&ReduceOp::value); }
    int twice(int x) const { return 2 * x + value; }
};
}
namespace alt {
struct ReduceOp {
    int base;
    int operator()() const { return base + 100; }
};
int run(ReduceOp op) { return op() * 2; }
}
using namespace ops;
kernel void k(device int* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    ops::ReduceOp a;
    ReduceOp b;
    alt::ReduceOp c{int(i)};
    out[4 * i] = a();
    out[4 * i + 1] = b.twice(int(i));
    out[4 * i + 2] = c();
    out[4 * i + 3] = alt::run(c);
}
"""

HOMONYM_OVERLOADS = PREFIX + """
struct P { uint base; };
namespace ops {
struct P { int base; };
int get(P p) { return p.base + 7; }
namespace inner {
struct P { int base; int scale; };
int get(P p) { return p.base * p.scale; }
int outer(ops::P p) { return get(P{p.base, 3}); }
}
}
namespace alt {
struct P { float base; };
int get(P p) { return int(p.base) + 100; }
}
int get(P p) { return int(p.base) - 1; }
kernel void k(device int* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    ops::P a{int(i)};
    alt::P c{float(i) + 3.0f};
    ::ops::inner::P d{int(i), 5};
    P e{i + 2u};
    out[5 * i] = ops::get(a);
    out[5 * i + 1] = alt::get(c);
    out[5 * i + 2] = ops::inner::get(d);
    out[5 * i + 3] = ops::inner::outer(a);
    out[5 * i + 4] = get(e);
}
"""

AMBIGUOUS = PREFIX + """
namespace ops { struct P { int base; }; }
namespace alt { struct P { int base; }; }
using namespace ops;
using namespace alt;
kernel void k(device int* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    P a{1};
    out[i] = a.base;
}
"""

THREADS = 4


def _translate(tmp_path, source, backend="metal", name="unit.metal"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _xcrun_metal(tmp_path, source, name):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / f"{name}.metal"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "--sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / f"{name}.air")],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    return result


def _run(source, entry, words):
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        source, metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    size = 4 * words * THREADS
    output = device.newBufferWithLength_options_(size, 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(output, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(THREADS, 1, 1), metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return struct.unpack(f"<{size // 4}i", output.contents().as_buffer(size))


def _entry(metal):
    found = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", metal)
    assert found, metal
    return found.group(1)


@pytest.mark.parametrize(
    "source, words",
    [
        (QUALIFIED_RECEIVER, 3),
        (USING_DIRECTIVE_RECEIVER, 3),
        (HOMONYM_METHODS, 4),
        (HOMONYM_OVERLOADS, 5),
    ],
    ids=[
        "qualified-receiver",
        "using-directive",
        "homonym-methods",
        "homonym-overloads",
    ],
)
def test_namespaced_structs_compute_the_source_results_on_metal(
    tmp_path, source, words
):
    metal = _translate(tmp_path, source)
    compiled = _xcrun_metal(tmp_path, metal, "translated")
    assert compiled.returncode == 0, compiled.stderr + metal
    assert _run(metal, _entry(metal), words) == _run(source, "k", words)


def test_homonym_structs_keep_distinct_names(tmp_path):
    crossgl = _translate(tmp_path, HOMONYM_METHODS, backend="cgl")
    assert re.search(r"\bstruct ops_ReduceOp\b", crossgl), crossgl
    assert re.search(r"\bstruct alt_ReduceOp\b", crossgl), crossgl
    assert not re.search(r"\bstruct ReduceOp\b", crossgl), crossgl
    crossgl = _translate(tmp_path, HOMONYM_OVERLOADS, backend="cgl")
    names = re.findall(r"\bstruct (\w*P)\b", crossgl)
    assert sorted(names) == ["P", "alt_P", "ops_P", "ops_inner_P"], crossgl


def test_qualified_receiver_lowers_through_the_struct_methods(tmp_path):
    metal = _translate(tmp_path, QUALIFIED_RECEIVER)
    assert "ReduceOp__operator_call(a)" in metal, metal
    assert "ReduceOp__twice(b, 1)" in metal, metal


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_homonym_structs_translate_to_other_targets(tmp_path, backend):
    output = _translate(tmp_path, HOMONYM_OVERLOADS, backend=backend)
    for name in ("ops_P", "alt_P", "ops_inner_P"):
        assert re.search(rf"\bstruct {name}\b", output), output


def test_ambiguous_unqualified_homonym_fails_closed(tmp_path):
    # C++ rejects the reference too ([namespace.udir]/6): both directives nominate a P.
    native = _xcrun_metal(tmp_path, AMBIGUOUS, "native")
    assert native.returncode != 0
    with pytest.raises(
        AmbiguousStructNameError, match=r"ambiguous: lookup finds alt::P and ops::P"
    ):
        _translate(tmp_path, AMBIGUOUS)


def test_homonym_rename_follows_lookup():
    source = """
namespace ops { struct P { int v; }; int f(P p); }
namespace alt { struct P { int v; }; }
namespace ops { int g() { P local{1}; return local.v; } }
int h(::ops::P a, alt::P b);
using namespace alt;
int k(P c);
"""
    renamed = distinct_struct_homonyms(source)
    assert "namespace ops { struct ops_P { int v; }; int f(ops_P p); }" in renamed
    assert "namespace alt { struct alt_P { int v; }; }" in renamed
    assert "ops_P local{1}" in renamed
    assert "int h(::ops::ops_P a, alt::alt_P b);" in renamed
    assert "int k(alt_P c);" in renamed


@pytest.mark.parametrize(
    "source",
    [
        "struct P { int v; };\nint f(P p);\n",
        "namespace ops { struct P { int v; }; }\nint f(ops::P p);\n",
        # A member named like the struct keeps every spelling, so the struct passes still fail closed on it.
        "namespace a { struct P { int v; }; }\nnamespace b { struct P { int P; }; }\n"
        "int f(b::P q) { return q.P; }\n",
        # So does a name an unnamed namespace declares.
        "namespace { struct P { int v; }; }\nnamespace b { struct P { int v; }; }\n",
    ],
    ids=["global-only", "one-namespace", "member-name", "unnamed-namespace"],
)
def test_homonym_rename_leaves_other_units_unchanged(source):
    assert distinct_struct_homonyms(source) == source
