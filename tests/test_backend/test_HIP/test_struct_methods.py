"""HIP struct member functions import as CrossGL free functions that take the receiver first.

HIP input used to refuse every call of a member function (``HIP record method call 'width' is unsupported``). It
now runs the CUDA importer's struct-method pass (:mod:`crosstl.backend.CUDA.struct_methods`), so a member function
defined in a record body becomes ``Span_width(Span this_)``, with an ``inout`` receiver when it writes a data
member, and the two importers write the same CrossGL for the same source.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.common_ast import MemberAccessNode
from crosstl.backend.CUDA.reference_parameters import ReferenceParameterError
from crosstl.backend.HIP.HipCrossGLCodeGen import HipRecordSemanticError
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser
from crosstl.backend.import_refusals import HipImportRefused
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

SPAN = """
struct Span {
    float lo;
    float hi;
    __device__ float width() {
        return hi - lo;
    }
    __device__ float at(float t) {
        return lo + t * width();
    }
    __device__ void widen(float d) {
        hi += d;
    }
};
__global__ void probe(const int* iin, float* outp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Span s;
    s.lo = float(iin[i]);
    s.hi = s.lo * 2.0f + 1.0f;
    float w = s.width();
    s.widen(3.0f);
    outp[i] = w + s.at(0.5f) * 64.0f + s.hi * 4096.0f;
}
"""

REFUSED = (HipImportRefused, HipRecordSemanticError, ReferenceParameterError)


def _translate(tmp_path, source, backend="cgl", suffix="hip"):
    path = tmp_path / f"kernel.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compact(text):
    return re.sub(r"\s+", " ", text)


def _free_functions(crossgl):
    return re.findall(r"^\w+ \w+_\w+\(.*?^}", crossgl, re.M | re.S)


def test_member_functions_become_free_functions_with_the_receiver_first(tmp_path):
    crossgl = _translate(tmp_path, SPAN)
    cgl = _compact(crossgl)

    assert "f32 Span_width(Span this_) { return (this_.hi - this_.lo); }" in cgl
    assert (
        "f32 Span_at(Span this_, f32 t) { return (this_.lo + (t * Span_width(this_))); }"
        in cgl
    )
    assert "void Span_widen(inout Span this_, f32 d) { this_.hi += d; }" in cgl
    assert "var w: f32 = Span_width(s);" in cgl
    assert "Span_widen(s, 3.0f);" in cgl
    assert "Span_at(s, 0.5f)" in cgl
    assert "s.width" not in cgl and "s.widen" not in cgl
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()


def test_hip_and_cuda_import_the_same_free_functions(tmp_path):
    hip = _free_functions(_translate(tmp_path, SPAN))
    cuda = _free_functions(_translate(tmp_path, SPAN, suffix="cu"))

    assert len(hip) == 3
    assert hip == cuda


def test_defaults_this_hidden_members_loops_and_pointer_receivers(tmp_path):
    source = """
struct Range {
    float lo;
    float hi;
    __device__ void set(float lo, float hi) { this->lo = lo; this->hi = hi; }
    __device__ float mid(float k = 0.5f) const { return (*this).lo * k + this->hi * k; }
    __device__ int odd(int n) const {
        int c = 0;
        while (n > 0) { n--; if (n % 2) c++; }
        return c;
    }
};
__global__ void k(float* o) {
    Range r;
    r.set(o[0], o[1]);
    o[2] = r.mid() + r.mid(0.25f);
    Range q = r;
    Range* p = &q;
    o[3] = float(p->odd(5));
}
"""
    cgl = _compact(_translate(tmp_path, source))

    assert (
        "void Range_set(inout Range this_, f32 lo, f32 hi) { this_.lo = lo; this_.hi = hi; }"
        in cgl
    )
    assert "f32 Range_mid(Range this_, f32 k" in cgl
    assert "((this_.lo * k) + (this_.hi * k))" in cgl
    assert "Range_set(r, o[0], o[1]);" in cgl
    assert "(Range_mid(r) + Range_mid(r, 0.25f))" in cgl
    assert "i32 Range_odd(Range this_, i32 n)" in cgl
    assert "Range_odd(p[0], 5)" in cgl


def test_free_functions_follow_their_record_and_precede_their_callers(tmp_path):
    source = """
__device__ float scale(float x) { return x * 2.0f; }
struct Acc {
    float total;
    __device__ void add(float x) { total += twice(x); }
    __device__ float twice(float x) const { return scale(x); }
};
__global__ void k(float* o) {
    Acc a;
    a.total = 0.0f;
    a.add(o[threadIdx.x]);
    o[threadIdx.x] = a.total;
}
"""
    cgl = _translate(tmp_path, source)

    order = [
        cgl.index(text)
        for text in (
            "f32 scale(",
            "struct Acc",
            "f32 Acc_twice(",
            "void Acc_add(",
            "fn k(",
        )
    ]
    assert order == sorted(order)
    assert "this_.total += Acc_twice(this_, x);" in cgl


_S = "struct S {{ float v; {} }};"


@pytest.mark.parametrize(
    ("records", "call"),
    [
        (_S.format("__device__ static float unit() { return 1.0f; }"), "s.unit()"),
        (_S.format("__device__ float later() const;"), "s.later()"),
        (_S.format("__device__ virtual float poly() const { return v; }"), "s.poly()"),
        (_S.format("__device__ float& ref() { return v; }"), "s.ref()"),
        (
            _S.format(
                "template <int N> __device__ float times() const { return v * N; }"
            ),
            "s.times<2>()",
        ),
        (_S.format("__device__ float get() { float v = 1.0f; return v; }"), "s.get()"),
        (_S.format("__device__ S* self() { return this; }"), "s.self()->v"),
        (
            "struct B { float v; __device__ float get() const { return v; } }; "
            "struct S : B { float w; };",
            "s.get()",
        ),
        (
            "template <typename T> struct Box { T v; __device__ T get() const { return v; } }; "
            "struct S { float v; Box<float> b; };",
            "s.b.get()",
        ),
        (
            _S.format("__device__ float get() const { return v; }"),
            "(o[2] > 0.0f ? s : s).get()",
        ),
    ],
    ids=[
        "static",
        "declared-only",
        "virtual",
        "reference-return",
        "member-template",
        "local-hides-member",
        "this-pointer",
        "inherited",
        "record-template",
        "unknown-receiver",
    ],
)
def test_member_calls_the_import_cannot_lower_are_refused(tmp_path, records, call):
    source = (
        f"{records}\n"
        "__global__ void k(float* o) {\n"
        "    S s;\n"
        "    s.v = o[0];\n"
        f"    o[1] = {call};\n"
        "}\n"
    )
    with pytest.raises(REFUSED, match="member|method"):
        _translate(tmp_path, source)


def test_class_methods_and_factory_receivers(tmp_path):
    # These two used to be refused as "HIP record method call".
    source = """
class State { public: int value; int get() const { return value; } };
State make_state(int value) { State state; state.value = value; return state; }
int host() { State state; state.value = 4; return state.get(); }
__global__ void probe(const int* input, int* output) {
    output[0] = make_state(input[0]).get() + host();
}
"""
    cgl = _compact(_translate(tmp_path, source))

    assert "i32 State_get(State this_) { return this_.value; }" in cgl
    assert "return State_get(state);" in cgl
    assert "(State_get(make_state(input[0])) + host())" in cgl


def test_the_probe_translates_to_hlsl_with_the_free_functions(tmp_path):
    hlsl = _translate(tmp_path, SPAN, backend="directx")

    assert "Span_widen(inout Span this_, float d)" in _compact(hlsl)
    assert ".width(" not in hlsl


def test_a_written_receiver_in_global_memory_binds_an_isolated_method(tmp_path):
    # The method reaches no memory but its receiver and does not synchronize, so a copy of the element in and out
    # is one of the behaviours a reference allows.
    source = """
struct S {
    float v;
    __device__ void bump() { v += 1.0f; }
};
__global__ void k(S* items) {
    items[threadIdx.x].bump();
}
"""
    cgl = _compact(_translate(tmp_path, source))

    assert "void S_bump(inout S this_)" in cgl
    assert "S_bump(items[gl_LocalInvocationID.x]);" in cgl


def test_a_written_receiver_in_global_memory_is_refused_when_the_method_synchronizes(
    tmp_path,
):
    source = """
struct S {
    float v;
    __device__ void bump() { __syncthreads(); v += 1.0f; }
};
__global__ void k(S* items) {
    items[threadIdx.x].bump();
}
"""
    with pytest.raises(REFUSED):
        _translate(tmp_path, source)


def test_member_functions_nothing_calls_are_left_out(tmp_path):
    source = """
struct S {
    float v;
    __device__ float& ref() { return v; }
    __device__ float get() const { return v; }
};
__global__ void k(float* o) {
    S s;
    s.v = o[0];
    o[1] = s.v;
}
"""
    cgl = _translate(tmp_path, source)

    assert "S_ref" not in cgl and "S_get" not in cgl
    assert "o[1] = s.v;" in cgl


def test_a_member_template_call_parses_as_a_member_call():
    # ``s.as<float>()`` used to reach the importer as the text of the member access node.
    program = HipParser(
        HipLexer("__device__ float f(S s) { return s.as<float>(); }").tokenize()
    ).parse()
    call = program.statements[0].body[0].value

    assert isinstance(call.name, MemberAccessNode)
    assert call.name.object == "s"
    assert call.name.member == "as<float>"


@pytest.mark.parametrize("backend", ["hip", "cuda"])
def test_the_round_trip_compiles(tmp_path, backend):
    compiler = shutil.which("hipcc" if backend == "hip" else "nvcc")
    if compiler is None:
        pytest.skip(f"{backend} compiler is not installed")
    output = _translate(tmp_path, SPAN, backend=backend)
    assert "Span_widen(Span& this_, float d)" in output
    source = tmp_path / f"round_trip.{'hip' if backend == 'hip' else 'cu'}"
    source.write_text(output, encoding="utf-8")
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "round_trip.o"),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
