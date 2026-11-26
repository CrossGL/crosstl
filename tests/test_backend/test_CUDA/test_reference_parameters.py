"""CUDA and HIP reference parameters import as inout when written and by value when only read."""

import re

import pytest

import crosstl
from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter
from crosstl.backend.CUDA.reference_parameters import (
    ReferenceParameterError,
    reference_kind,
)

SOURCE = """struct Pair {
    float x;
    float y;
};
__device__ void grow(float& a) { a = a * 2.0f; }
__device__ void settle(float& a) { __syncthreads(); a = a * 2.0f; }
__device__ void swap_pair(Pair& q, const float& c) {
    float t = q.x;
    q.x = q.y + c;
    q.y = t;
}
__device__ void fill(float (&v)[2]) { v[0] += 1.0f; }
__device__ float peek(float& r, Pair& s) { return r + s.x; }
__device__ void twice(float& a) { grow(a); }
__device__ void step(int& n) { n++; }
__device__ float first(float*& p) { p[0] = 1.0f; return *p; }
extern "C" __global__ void k(float* outp) {
    float x = 3.0f;
    grow(x);
    twice(x);
    Pair p;
    p.x = 1.0f;
    p.y = 2.0f;
    swap_pair(p, x);
    float w[2] = {1.0f, 2.0f};
    fill(w);
    int n = 0;
    step(n);
    float* q = outp;
    outp[0] = x + p.x + w[0] + peek(outp[1], p) + first(q) + (float)n;
    BODY
}
"""


def _translate(tmp_path, extension, body=""):
    path = tmp_path / f"refs.{extension}"
    path.write_text(SOURCE.replace("BODY", body))
    return crosstl.translate(str(path), backend="cgl", format_output=False)


def _signatures(code):
    return {
        match.group(1): match.group(0)
        for match in re.finditer(r"^\S+ (\w+)\([^)]*\)", code, re.MULTILINE)
    }


@pytest.mark.parametrize("extension", ["cu", "hip"])
@pytest.mark.parametrize("record_name", ["Pair_", "_Half"])
def test_record_type_underscores_are_preserved_on_import(
    tmp_path, extension, record_name
):
    path = tmp_path / f"record.{extension}"
    path.write_text(
        f"struct {record_name} {{ float a; }};\n"
        f"__global__ void probe(float* outp) {{\n"
        f"    {record_name} t;\n"
        "    t.a = 2.0f;\n"
        "    outp[0] = t.a;\n"
        "}\n"
    )
    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert f"struct {record_name}" in crossgl
    assert f"var t: {record_name}" in crossgl


def test_non_ascii_type_name_sanitizes_to_crossgl_identifier():
    converter = CudaToCrossGLConverter()
    sanitized = converter.sanitize_crossgl_type_identifier("café")
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", sanitized)


@pytest.mark.parametrize("extension", ["cu", "hip"])
def test_written_references_import_as_inout(tmp_path, extension):
    signatures = _signatures(_translate(tmp_path, extension))
    assert "(inout f32 a)" in signatures["grow"]
    assert "inout Pair q" in signatures["swap_pair"]
    assert "inout array<f32, 2> v" in signatures["fill"]
    assert "(inout f32 a)" in signatures["twice"], "a write through a callee is a write"
    assert "(inout i32 n)" in signatures["step"]


@pytest.mark.parametrize("extension", ["cu", "hip"])
def test_read_references_import_by_value(tmp_path, extension):
    signatures = _signatures(_translate(tmp_path, extension))
    assert (
        "f32 c)" in signatures["swap_pair"]
        and "inout f32 c" not in signatures["swap_pair"]
    )
    assert (
        "inout" not in signatures["peek"]
    ), "read-only references bind buffer elements by value"
    assert (
        "inout" not in signatures["first"]
    ), "writing through a pointer reference keeps the pointer"


@pytest.mark.parametrize("extension", ["cu", "hip"])
@pytest.mark.parametrize(
    "body, message",
    [
        (
            "settle(outp[2]);",
            "argument 1 of 'settle' binds a reference parameter to storage",
        ),
        (
            "settle(q[1]);",
            "argument 1 of 'settle' binds a reference parameter to storage",
        ),
        (
            "float* r = &x; settle(x);",
            "argument 1 of 'settle' binds a reference parameter to storage",
        ),
        (
            "float& r = x; grow(r);",
            "argument 1 of 'grow' binds a reference parameter to storage",
        ),
        (
            "for (auto& e : w) { grow(e); }",
            "argument 1 of 'grow' binds a reference parameter to storage",
        ),
        ("swap_pair(p, p.x);", "overlapping parts of 'p'"),
    ],
)
def test_refuses_arguments_copy_in_and_out_cannot_keep(
    tmp_path, extension, body, message
):
    with pytest.raises(ReferenceParameterError, match=re.escape(message)):
        _translate(tmp_path, extension, body)


ISOLATED = """__device__ void grow(float& x, int k) { x = x * 2.0f + (float)(k); }
__device__ void twice(float& x, int k) { grow(x, k); grow(x, k); }
__device__ void pair(float& a, float& b) { a += b; b += 1.0f; }
extern "C" __global__ void k(const int* iin, float* outp) {
    __shared__ float tile[64];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 1.0f;
    outp[i] = (float)(iin[i]) * 0.5f;
    grow(outp[i], iin[i]);
    twice(outp[i + 1u], 3);
    tile[threadIdx.x] = outp[i];
    grow(tile[threadIdx.x], 2);
    pair(outp[i], local);
    outp[i] += tile[threadIdx.x] + local;
}
"""


@pytest.mark.parametrize("extension", ["cu", "hip"])
def test_isolated_callees_bind_buffer_and_shared_elements(tmp_path, extension):
    """A callee that reaches no memory but its references cannot tell a copy in and out from a reference."""
    path = tmp_path / f"isolated.{extension}"
    path.write_text(ISOLATED)
    code = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "void grow(inout f32 x, i32 k)" in code
    assert "grow(outp[i], iin[i]);" in code
    assert "twice(outp[(i + 1u)], 3);" in code or "twice(outp[i + 1u], 3);" in code
    assert "pair(outp[i], local);" in code
    metal = crosstl.translate(str(path), backend="metal")
    assert "device float &" not in metal, "Metal binds the element through a copy"


NOT_ISOLATED_PRELUDE = """struct View { float* p; };
struct Outer { View view; };
__device__ int counter;
__device__ void settle(float& a) { __syncthreads(); a = a * 2.0f; }
__device__ void outer(float& a) { settle(a); }
__device__ void global_write(float& a) { counter = 1; a += 1.0f; }
__device__ void atomic_bump(float& a, int k) { atomicAdd(&counter, k); a += 1.0f; }
__device__ void scaled(float& a, const float* s) { a *= s[0]; }
__device__ void viewed(float& a, Outer o) { o.view.p[0] = 0.0f; a += 1.0f; }
__device__ void cast_store(float& a, unsigned long long at) { *(float*)at = 0.0f; a += 1.0f; }
__device__ void grow(float& a) { a = a * 2.0f; }
__device__ void two(float& a, float& b) { a += 1.0f; b += a; }
"""


@pytest.mark.parametrize("extension", ["cu", "hip"])
@pytest.mark.parametrize(
    "call, callee",
    [
        ("settle(outp[i]);", "settle"),
        ("outer(outp[i]);", "outer"),
        ("global_write(outp[i]);", "global_write"),
        ("atomic_bump(outp[i], 1);", "atomic_bump"),
        ("scaled(outp[i], outp);", "scaled"),
        ("Outer o; o.view.p = outp; viewed(outp[i], o);", "viewed"),
        (
            "cast_store(outp[i], (unsigned long long)outp);",
            "cast_store",
        ),
        ("unsigned int j = i; grow(outp[j++]);", "grow"),
        ("two(outp[i], outp[i + 1u]);", "two"),
    ],
)
def test_buffer_elements_bind_only_isolated_callees(tmp_path, extension, call, callee):
    """Another thread, another argument or the callee itself may reach the element, so a copy would miss writes."""
    path = tmp_path / f"shared.{extension}"
    path.write_text(
        NOT_ISOLATED_PRELUDE
        + 'extern "C" __global__ void k(float* outp) {\n'
        + "    unsigned int i = threadIdx.x;\n"
        + f"    {call}\n"
        + "}\n"
    )
    with pytest.raises(
        ReferenceParameterError,
        match=re.escape(f"of '{callee}' binds a reference parameter to storage"),
    ):
        crosstl.translate(str(path), backend="cgl", format_output=False)


@pytest.mark.parametrize("extension", ["cu", "hip"])
def test_disjoint_members_and_constant_elements_may_be_passed_together(
    tmp_path, extension
):
    source = """__device__ void two(float& a, float& b) { a += 1.0f; b += a; }
struct Pair { float x; float y; };
extern "C" __global__ void k(float* outp) {
    Pair p; p.x = 1.0f; p.y = 2.0f;
    float w[2] = {1.0f, 2.0f};
    two(p.x, p.y);
    two(w[0], w[1]);
    outp[0] = p.y + w[1];
}
"""
    path = tmp_path / f"two.{extension}"
    path.write_text(source)
    code = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "(inout f32 a, inout f32 b)" in code


@pytest.mark.parametrize("extension", ["cu", "hip"])
@pytest.mark.parametrize(
    "declaration, message",
    [
        (
            "__device__ void sink(float&& v) { v = 1.0f; }",
            "rvalue reference the function writes",
        ),
        (
            "__device__ void move(float*& p) { p = p + 1; }",
            "reference to a pointer the function writes",
        ),
    ],
)
def test_refuses_written_rvalue_and_pointer_references(
    tmp_path, extension, declaration, message
):
    path = tmp_path / f"kind.{extension}"
    path.write_text(
        f'{declaration}\nextern "C" __global__ void k(float* outp) {{ outp[0] = 1.0f; }}\n'
    )
    with pytest.raises(ReferenceParameterError, match=re.escape(message)):
        crosstl.translate(str(path), backend="cgl", format_output=False)


def test_hip_overlapping_record_references_are_refused(tmp_path):
    path = tmp_path / "step.hip"
    path.write_text(
        "struct State { int value; };\n"
        "void step(State& first, State& second) { first.value += 1; second.value += first.value; }\n"
        "__global__ void probe(const int* input, int* output) {\n"
        "    State state{input[0]}; step(state, state); output[0] = state.value;\n"
        "}\n"
    )
    with pytest.raises(ReferenceParameterError, match="overlapping parts of 'state'"):
        crosstl.translate(str(path), backend="cgl", format_output=False)


def test_hip_parses_reference_to_array_parameters(tmp_path):
    path = tmp_path / "array.hip"
    path.write_text(
        "__device__ float total(const float (&c)[3], float (&v)[2]) { v[1] = c[0]; return c[1] + v[0]; }\n"
        "__global__ void k(float* outp) { float a[3] = {1.0f, 2.0f, 3.0f}; float b[2] = {0.0f, 0.0f};\n"
        "    outp[0] = total(a, b) + b[1]; }\n"
    )
    code = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "f32 total(array<f32, 3> c, inout array<f32, 2> v)" in code


def test_hip_reference_aliases_resolve(tmp_path):
    path = tmp_path / "alias.hip"
    path.write_text(
        "struct State { int value; }; using Ref = State&;\n"
        "void increment(Ref state, int delta) { state.value += delta; }\n"
        "__global__ void probe(const int* input, int* output) {\n"
        "    State state{input[0]}; increment(state, input[1]); output[0] = state.value;\n"
        "}\n"
    )
    code = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "void increment(inout State state, i32 delta)" in code


@pytest.mark.parametrize(
    "text, kind",
    [
        ("float", None),
        ("float &", "lvalue"),
        ("Pair &", "lvalue"),
        ("float & [2]", "lvalue"),
        ("const float &", "const"),
        ("float const &", "const"),
        ("const float & [2]", "const"),
        ("float * const &", "const"),
        ("const float * &", "pointer"),
        ("float * &", "pointer"),
        ("float &&", "rvalue"),
        ("T &&", "rvalue"),
        ("vec<float&, 2>", None),
    ],
)
def test_reference_kind(text, kind):
    assert reference_kind(text) == kind
