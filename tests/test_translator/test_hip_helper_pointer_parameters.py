"""Keep imported helper pointer parameters valid in HIP output."""

import pytest

from crosstl import translate
from crosstl.translator.codegen.hip_codegen import HipCodeGen


@pytest.mark.parametrize("suffix", ["hip", "cu"])
@pytest.mark.parametrize("element_type", ["float", "int", "unsigned int"])
def test_helper_pointer_parameter_round_trip(tmp_path, suffix, element_type):
    source = tmp_path / f"put.{suffix}"
    source.write_text(
        f"""
__device__ void put({element_type}* outp, unsigned int j, {element_type} x) {{
    outp[j] = x;
}}
extern "C" __global__ void probe({element_type}* outp) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    put(outp, i, 1);
}}
""",
        encoding="utf-8",
    )
    result = translate(str(source), backend="hip", format_output=False)
    assert f"void put({element_type}* outp," in result
    assert "ptr<" not in result
    assert "put(outp, i," in result


@pytest.mark.parametrize(
    ("source_type", "expected"),
    [("ptr<f32>", "float*"), ("ptr<i32>", "int*"), ("ptr<ptr<f32>>", "float**")],
)
def test_hip_maps_pointer_element_types_recursively(source_type, expected):
    assert HipCodeGen().map_type(source_type) == expected
