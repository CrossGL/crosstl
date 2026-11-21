"""CUDA default arguments import as CrossGL parameter defaults.

The CUDA parser used to skip every ``= value`` after a parameter, so ``blend(x)`` reached CrossGL one argument
short of ``blend(float a, float b = 2.0f, int k = 3)`` and nvcc rejected the round trip with ``identifier "blend"
is undefined``. Each default now becomes the CrossGL parameter's default, which the translator fills into every
call that omits it; a default given on an earlier declaration reaches the definition, and a default the parser
cannot read is refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.backend.CUDA.parameter_defaults import (
    UnparsedParameterDefault,
    merge_declared_defaults,
)
from crosstl.backend.import_refusals import CudaImportRefused

BLEND = """
__device__ float blend(float a, float b = 2.0f, int k = 3) {
    return a * b + (float)(k);
}
extern "C" __global__ void probe(int* iin, float* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)(iin[i]);
    outp[i] = blend(x) + blend(x, 4.0f) * 64.0f + blend(x, 0.5f, 1) * 4096.0f;
}
"""


def _translate(tmp_path, source, backend="cgl"):
    path = tmp_path / "kernel.cu"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compact(text):
    return re.sub(r"\s+", " ", text)


def _parse(source):
    return CudaParser(CudaLexer(source).tokenize()).parse()


def test_defaults_become_crossgl_parameter_defaults(tmp_path):
    crossgl = _compact(_translate(tmp_path, BLEND))

    assert "f32 blend(f32 a, f32 b = 2.0f, i32 k = 3)" in crossgl
    assert "blend(x, 2.0f, 3)" in crossgl
    assert "blend(x, 4.0f, 3)" in crossgl
    assert "blend(x, 0.5f, 1)" in crossgl


@pytest.mark.parametrize(
    "backend, calls",
    [
        ("cuda", ("blend(x, 2.0f, 3)", "blend(x, 4.0f, 3)", "blend(x, 0.5f, 1)")),
        ("hip", ("blend(x, 2.0f, 3)", "blend(x, 4.0f, 3)", "blend(x, 0.5f, 1)")),
        ("directx", ("blend(x, 2.0, 3)", "blend(x, 4.0, 3)", "blend(x, 0.5, 1)")),
        ("opengl", ("blend(x, 2.0, 3)", "blend(x, 4.0, 3)", "blend(x, 0.5, 1)")),
        ("metal", ("blend(x, 2.0, 3)", "blend(x, 4.0, 3)", "blend(x, 0.5, 1)")),
    ],
)
def test_every_target_receives_the_omitted_arguments(tmp_path, backend, calls):
    output = _compact(_translate(tmp_path, BLEND, backend))

    for call in calls:
        assert call in output
    assert "blend(x)" not in output


def test_vector_macro_conditional_and_boolean_defaults(tmp_path):
    source = """
#define SCALE 3.0f
__device__ float2 shift(float2 p, float2 d = make_float2(1.0f, -1.0f), float s = SCALE * 2.0f,
                        bool flip = false) {
    return flip ? make_float2(p.y, p.x) : make_float2(p.x * s + d.x, p.y * s + d.y);
}
__device__ float lift(float x, float b = (float)(4 > 3 ? 2 : 1)) { return x + b; }
extern "C" __global__ void probe(float* outp) {
    float2 r = shift(make_float2(outp[0], outp[1]));
    outp[0] = r.x + lift(r.y);
}
"""
    crossgl = _compact(_translate(tmp_path, source))

    assert (
        "vec2<f32> shift(vec2<f32> p, vec2<f32> d = vec2<f32>(1.0f, (-1.0f)), "
        "f32 s = (3.0f * 2.0f), bool flip = false)"
    ) in crossgl
    assert "f32 lift(f32 x, f32 b = f32(((4 > 3) ? 2 : 1)))" in crossgl
    assert (
        "shift(vec2<f32>(outp[0], outp[1]), vec2<f32>(1.0f, (-1.0f)), (3.0f * 2.0f), false)"
        in crossgl
    )
    assert "lift(r.y, f32(((4 > 3) ? 2 : 1)))" in crossgl


def test_a_default_on_an_earlier_declaration_reaches_the_definition(tmp_path):
    source = """
__device__ float blend(float a, float b = 2.0f);
__device__ float blend(float a, float b) { return a * b; }
extern "C" __global__ void probe(float* outp) { outp[0] = blend(outp[1]); }
"""
    crossgl = _compact(_translate(tmp_path, source))

    assert "f32 blend(f32 a, f32 b = 2.0f)" in crossgl
    assert "blend(outp[1], 2.0f)" in crossgl


def test_declared_defaults_pick_the_overload_with_the_same_parameter_types():
    program = _parse("""
__device__ float pick(float a, float b = 2.0f);
__device__ float pick(float a, float b) { return a + b; }
__device__ float pick(int a, int b) { return (float)(a - b); }
""")
    functions = [f for f in program.functions if f.name == "pick"]
    merge_declared_defaults(functions)

    by_first_type = {f.params[0].vtype: f for f in functions if f.body is not None}
    assert by_first_type["float"].params[1].default_value is not None
    assert by_first_type["int"].params[1].default_value is None


def test_a_definition_keeps_its_own_defaults():
    program = _parse("""
__device__ float keep(float a, float b);
__device__ float keep(float a, float b = 5.0f) { return a + b; }
""")
    merge_declared_defaults(program.functions)
    definition = next(f for f in program.functions if f.body is not None)

    assert definition.params[1].default_value is not None
    assert definition.params[0].default_value is None


def test_brace_initialized_defaults_are_kept(tmp_path):
    source = """
struct P { float a; };
__device__ float g(float x, P p = {}) { return x + p.a; }
extern "C" __global__ void probe(float* outp) { outp[0] = g(outp[0]); }
"""
    crossgl = _compact(_translate(tmp_path, source))

    assert "f32 g(f32 x, P p = {})" in crossgl
    assert "g(outp[0], {})" in crossgl


def test_a_default_the_parser_cannot_read_is_refused(tmp_path):
    program = _parse("""
__device__ float h(float x, float y = throw 1) { return x + y; }
""")
    default = program.functions[0].params[1].default_value
    assert isinstance(default, UnparsedParameterDefault)
    assert default.text == "throw 1"

    source = """
__device__ float h(float x, float y = throw 1) { return x + y; }
extern "C" __global__ void probe(float* outp) { outp[0] = h(outp[0]); }
"""
    with pytest.raises(CudaImportRefused) as raised:
        _translate(tmp_path, source)
    message = str(raised.value)
    assert "default argument" in message
    assert "parameter 'y' of 'h'" in message


def test_parameters_without_defaults_are_unchanged(tmp_path):
    source = """
__device__ float add(float a, float b) { return a + b; }
extern "C" __global__ void probe(float* outp) { outp[0] = add(outp[0], outp[1]); }
"""
    crossgl = _compact(_translate(tmp_path, source))

    assert "f32 add(f32 a, f32 b) {" in crossgl


def test_the_cuda_round_trip_compiles_with_nvcc(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    source = tmp_path / "round_trip.cu"
    source.write_text(_translate(tmp_path, BLEND, "cuda"), encoding="utf-8")
    result = subprocess.run(
        [nvcc, "-std=c++17", "-c", str(source), "-o", str(tmp_path / "round_trip.o")],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
