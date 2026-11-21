"""HIP default arguments import as CrossGL parameter defaults.

The HIP importer dropped every ``= value`` after a parameter, so ``blend(x)`` reached CrossGL one argument short of
``blend(float a, float b = 2.0f, int k = 3)`` and hipcc rejected the round trip with ``use of undeclared identifier
'blend'``. Each default now becomes the CrossGL parameter's default, which the translator fills into every call that
omits it; a default given on an earlier declaration reaches the definition, and a default the parser cannot read is
refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.CUDA.parameter_defaults import UnparsedParameterDefault
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser
from crosstl.backend.import_refusals import HipImportRefused

BLEND = """
#include <hip/hip_runtime.h>
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
    path = tmp_path / "kernel.hip"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compact(text):
    return re.sub(r"\s+", " ", text)


def test_defaults_become_crossgl_parameter_defaults(tmp_path):
    crossgl = _compact(_translate(tmp_path, BLEND))

    assert "f32 blend(f32 a, f32 b = 2.0f, i32 k = 3)" in crossgl
    # The CrossGL keeps each call as written; every generator fills the omitted arguments.
    assert "blend(x)" in crossgl
    assert "blend(x, 4.0f)" in crossgl
    assert "blend(x, 0.5f, 1)" in crossgl


@pytest.mark.parametrize(
    "backend, calls",
    [
        ("hip", ("blend(x, 2.0f, 3)", "blend(x, 4.0f, 3)", "blend(x, 0.5f, 1)")),
        ("cuda", ("blend(x, 2.0f, 3)", "blend(x, 4.0f, 3)", "blend(x, 0.5f, 1)")),
        ("directx", ("blend(x, 2.0, 3)", "blend(x, 4.0, 3)", "blend(x, 0.5, 1)")),
        ("metal", ("blend(x, 2.0, 3)", "blend(x, 4.0, 3)", "blend(x, 0.5, 1)")),
    ],
)
def test_every_target_receives_the_omitted_arguments(tmp_path, backend, calls):
    output = _compact(_translate(tmp_path, BLEND, backend))

    for call in calls:
        assert call in output
    assert "blend(x)" not in output


def test_a_default_on_an_earlier_declaration_reaches_the_definition(tmp_path):
    source = """
#include <hip/hip_runtime.h>
__device__ float blend(float a, float b = 2.0f);
__device__ float blend(float a, float b) { return a * b; }
extern "C" __global__ void probe(float* outp) { outp[0] = blend(outp[1]); }
"""
    crossgl = _compact(_translate(tmp_path, source))
    definition = re.findall(r"f32 blend\(f32 a, f32 b( = 2\.0f)?\) \{ return", crossgl)
    assert definition == [" = 2.0f"]

    assert "blend(outp[1], 2.0f)" in _compact(_translate(tmp_path, source, "hip"))


def test_a_default_the_parser_cannot_read_is_refused(tmp_path):
    source = """
#include <hip/hip_runtime.h>
__device__ float h(float x, float y = throw 1) { return x + y; }
extern "C" __global__ void probe(float* outp) { outp[0] = h(outp[0]); }
"""
    program = HipParser(HipLexer(source).tokenize()).parse()
    function = next(
        stmt for stmt in program.statements if getattr(stmt, "name", None) == "h"
    )
    default = function.params[1]["default"]
    assert isinstance(default, UnparsedParameterDefault)
    assert default.text == "throw 1"

    with pytest.raises(HipImportRefused) as raised:
        _translate(tmp_path, source)
    message = str(raised.value)
    assert "HIP default argument" in message
    assert "parameter 'y' of 'h'" in message


def test_parameters_without_defaults_are_unchanged(tmp_path):
    source = """
#include <hip/hip_runtime.h>
__device__ float add(float a, float b) { return a + b; }
extern "C" __global__ void probe(float* outp) { outp[0] = add(outp[0], outp[1]); }
"""
    crossgl = _compact(_translate(tmp_path, source))

    assert "f32 add(f32 a, f32 b) {" in crossgl


def test_the_hip_round_trip_compiles_with_hipcc(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    source = tmp_path / "round_trip.hip"
    source.write_text(_translate(tmp_path, BLEND, "hip"), encoding="utf-8")
    result = subprocess.run(
        [
            hipcc,
            "-std=c++17",
            "--offload-arch=gfx90a",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "round_trip.o"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
