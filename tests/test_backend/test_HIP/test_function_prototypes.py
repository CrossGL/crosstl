"""HIP function prototypes import as the definitions they declare (#3260)."""

import re

import pytest

from crosstl import translate
from crosstl.backend.import_refusals import HipImportRefused

FORWARD_DECLARED = """#include <hip/hip_runtime.h>
__device__ float twice(float x);
__device__ float quadruple(float x) { return twice(twice(x)); }
__device__ float twice(float x) { return x * 2.0f; }
"""


def _translate(tmp_path, source_text, backend="cgl"):
    source = tmp_path / "probe.hip"
    source.write_text(source_text, encoding="utf-8")
    return translate(str(source), backend=backend, save_shader=None)


def test_prototype_of_a_function_defined_later_is_dropped(tmp_path):
    crossgl = _translate(tmp_path, FORWARD_DECLARED)

    assert len(re.findall(r"\bf32 twice\(f32 x\) \{", crossgl)) == 1
    assert "return (x * 2.0f);" in crossgl
    assert "return twice(twice(x));" in crossgl


def test_prototype_round_trips_to_one_hip_definition(tmp_path):
    hip = _translate(tmp_path, FORWARD_DECLARED, backend="hip")

    assert len(re.findall(r"__device__ float twice\(float x\)", hip)) == 1


def test_kernel_calling_a_prototype_defined_later_keeps_one_definition(tmp_path):
    crossgl = _translate(
        tmp_path,
        "__device__ float twice(float x);\n"
        "__global__ void probe(float* out) { out[0] = twice(out[0]); }\n"
        "__device__ float twice(float x) { return x * 2.0f; }\n",
    )

    assert len(re.findall(r"\bf32 twice\(f32 x\) \{", crossgl)) == 1
    assert "out[0] = twice(out[0]);" in crossgl


def test_called_prototype_without_a_definition_is_refused(tmp_path):
    with pytest.raises(HipImportRefused) as raised:
        _translate(
            tmp_path,
            "__device__ float twice(float x);\n"
            "__global__ void probe(float* out) { out[0] = twice(out[0]); }\n",
        )

    assert "HIP function 'twice' is called but never defined" in str(raised.value)
