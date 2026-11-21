"""CUDA struct member functions import as CrossGL free functions that take the receiver first.

The CUDA parser used to skip every member function in a struct body, so a call ``s.width()`` reached CrossGL as a
member call that no target declares, and nvcc rejected the round trip with ``class "Span" has no member "width"``.
Each member function defined in a struct body now becomes ``Span_width(Span this_)``, with an ``inout`` receiver when
it writes a data member; a member call the importer cannot resolve to one of these functions is refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.CUDA.reference_parameters import ReferenceParameterError
from crosstl.backend.import_refusals import CudaImportRefused

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
extern "C" __global__ void probe(int* iin, float* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    Span s;
    s.lo = (float)(iin[i]);
    s.hi = s.lo * 2.0f + 1.0f;
    float w = s.width();
    s.widen(3.0f);
    outp[i] = w + s.at(0.5f) * 64.0f + s.hi * 4096.0f;
}
"""


def _translate(tmp_path, source, backend="cgl"):
    path = tmp_path / "kernel.cu"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compact(text):
    return re.sub(r"\s+", " ", text)


def _refusal(tmp_path, source):
    with pytest.raises(CudaImportRefused) as raised:
        _translate(tmp_path, source)
    return str(raised.value)


def test_member_functions_become_free_functions_with_the_receiver_first(tmp_path):
    cgl = _compact(_translate(tmp_path, SPAN))

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


def test_free_functions_precede_their_callers_and_follow_earlier_helpers(tmp_path):
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
        cgl.index(name) for name in ("f32 scale(", "f32 Acc_twice(", "void Acc_add(")
    ]
    assert order == sorted(order)
    assert "this_.total += Acc_twice(this_, x);" in cgl
    assert "Acc_add(a, o[gl_LocalInvocationID.x]);" in cgl


def test_this_parameters_that_hide_members_and_const_methods(tmp_path):
    source = """
struct Range {
    float lo;
    float hi;
    __device__ void set(float lo, float hi) { this->lo = lo; this->hi = hi; }
    __device__ float mid() const { return (*this).lo * 0.5f + this->hi * 0.5f; }
};
__global__ void k(float* o) {
    Range r;
    r.set(o[0], o[1]);
    o[2] = r.mid();
}
"""
    cgl = _compact(_translate(tmp_path, source))

    assert (
        "void Range_set(inout Range this_, f32 lo, f32 hi) { this_.lo = lo; this_.hi = hi; }"
        in cgl
    )
    assert (
        "f32 Range_mid(Range this_) { return ((this_.lo * 0.5f) + (this_.hi * 0.5f)); }"
        in cgl
    )
    assert "Range_set(r, o[0], o[1]);" in cgl
    assert "o[2] = Range_mid(r);" in cgl


def test_receivers_through_pointers_elements_members_and_overloads(tmp_path):
    source = """
struct Span {
    float lo;
    float hi;
    __device__ float width() const { return hi - lo; }
    __device__ float width(float scale) const { return (hi - lo) * scale; }
};
struct Pair { Span a; Span b; };
__global__ void k(Span* spans, float* o) {
    Span local[2];
    Pair p;
    local[0] = spans[1];
    p.a = spans[0];
    o[0] = spans->width();
    o[1] = spans[1].width(2.0f);
    o[2] = local[0].width();
    o[3] = p.a.width(3.0f);
}
"""
    cgl = _compact(_translate(tmp_path, source))

    assert "f32 Span_width_0(Span this_)" in cgl
    assert "f32 Span_width_1(Span this_, f32 scale)" in cgl
    assert "Span_width_0(spans[0])" in cgl
    assert "Span_width_1(spans[1], 2.0f)" in cgl
    assert "Span_width_0(local[0])" in cgl
    assert "Span_width_1(p.a, 3.0f)" in cgl


def test_a_member_call_on_a_value_that_is_not_a_program_struct_is_kept(tmp_path):
    source = """
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
struct Gate {
    int open;
    __device__ void sync() { open = 1; }
};
__global__ void k(int* o) {
    cg::thread_block block = cg::this_thread_block();
    Gate g;
    g.sync();
    block.sync();
    o[threadIdx.x] = g.open;
}
"""
    cgl = _translate(tmp_path, source)

    assert "Gate_sync(g);" in cgl
    assert "workgroupBarrier();" in cgl
    assert "block.sync" not in cgl


@pytest.mark.parametrize(
    ("body", "member", "reason"),
    [
        (
            "__device__ static float unit() { return 1.0f; }",
            "s.unit()",
            "no single member function",
        ),
        (
            "template <int N> __device__ float times() const { return v * N; }",
            "s.times<2>()",
            "",
        ),
        ("__device__ float later() const;", "s.later()", "no single member function"),
        (
            "__device__ virtual float poly() const { return v; }",
            "s.poly()",
            "no single member function",
        ),
    ],
)
def test_calls_of_member_functions_the_importer_skips_are_refused(
    tmp_path, body, member, reason
):
    source = f"""
struct S {{
    float v;
    {body}
}};
__global__ void k(float* o) {{
    S s;
    s.v = o[0];
    o[1] = {member};
}}
"""
    message = _refusal(tmp_path, source)

    assert "member" in message
    assert reason in message


def test_a_receiver_of_unknown_type_is_refused(tmp_path):
    source = """
struct S {
    float v;
    __device__ float get() const { return v; }
};
__device__ S make(float x) { S s; s.v = x; return s; }
__global__ void k(float* o) {
    auto s = o[0] > 0.0f ? make(o[0]) : make(1.0f);
    o[1] = s.get();
}
"""
    message = _refusal(tmp_path, source)

    assert "member call 'get'" in message
    assert "type of its receiver" in message


@pytest.mark.parametrize(
    ("method", "reason"),
    [
        ("__device__ float get() { float v = 1.0f; return v; }", "hides member 'v'"),
        ("__device__ S* self() { return this; }", "'this' as a pointer value"),
        ("__device__ float& ref() { return v; }", "returns a reference"),
        ("__device__ float get() { return [&]() { return v; }(); }", "does not handle"),
    ],
)
def test_member_function_bodies_the_import_cannot_carry_are_refused(
    tmp_path, method, reason
):
    source = f"""
struct S {{
    float v;
    {method}
}};
__global__ void k(float* o) {{
    S s;
    s.v = o[0];
    s.{method.split('(')[0].split()[-1].lstrip('*&')}();
    o[1] = s.v;
}}
"""
    message = _refusal(tmp_path, source)

    assert reason in message


def test_a_written_receiver_in_global_memory_is_refused(tmp_path):
    source = """
struct S {
    float v;
    __device__ void bump() { v += 1.0f; }
};
__global__ void k(S* items) {
    items[threadIdx.x].bump();
}
"""
    with pytest.raises((CudaImportRefused, ReferenceParameterError)):
        _translate(tmp_path, source)


def test_an_auto_receiver_takes_the_type_of_its_initializer(tmp_path):
    source = """
struct S {
    float v;
    __device__ float get() const { return v; }
};
__device__ S make(float x) { S s; s.v = x; return s; }
__global__ void k(float* o) {
    auto s = make(o[0]);
    o[1] = s.get();
}
"""
    assert "o[1] = S_get(s);" in _compact(_translate(tmp_path, source))


def test_struct_template_member_calls_are_refused(tmp_path):
    source = """
template <typename T>
struct Box {
    T v;
    __device__ T get() const { return v; }
};
__global__ void k(float* o) {
    Box<float> b;
    b.v = o[0];
    o[1] = b.get();
}
"""
    message = _refusal(tmp_path, source)

    assert "member call 'Box::get'" in message


def test_the_cuda_round_trip_compiles_with_nvcc(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    cuda = _translate(tmp_path, SPAN, backend="cuda")
    assert "Span_widen(Span& this_, float d)" in cuda
    source = tmp_path / "round_trip.cu"
    source.write_text(cuda, encoding="utf-8")
    result = subprocess.run(
        [nvcc, "-std=c++17", "-c", str(source), "-o", str(tmp_path / "round_trip.o")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


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
