"""HLSL's canonical 64-bit aliases reach CUDA and HIP as native types."""

import re

import pytest

import crosstl


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_hlsl_64bit_locals_and_shared_variables_have_native_types(tmp_path, backend):
    source = tmp_path / "wide.hlsl"
    source.write_text(
        """
RWStructuredBuffer<uint64_t> results : register(u0);
RWStructuredBuffer<int64_t> signed_results : register(u1);
groupshared uint64_t total;
[numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {
    uint64_t v = results[1];
    int64_t s = signed_results[1];
    total = v;
    results[0] = total + v;
    signed_results[0] = s;
}
"""
    )
    output = crosstl.translate(str(source), backend=backend, format_output=False)
    assert "__shared__ unsigned long long total;" in output, output
    assert "unsigned long long v = results[1];" in output, output
    assert "long long s = signed_results[1];" in output, output
    assert not re.search(r"\b(?:uint64|int64)\b", output), output


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize(
    ("alias", "native"), [("int64", "long long"), ("uint64", "unsigned long long")]
)
def test_64bit_alias_constructors_use_valid_multiword_casts(
    tmp_path, backend, alias, native
):
    source = tmp_path / "cast.cgl"
    source.write_text(
        "shader Wide { compute { void main() { "
        + alias
        + " value = "
        + alias
        + "(7); } } }"
    )
    output = crosstl.translate(str(source), backend=backend, format_output=False)
    assert f"{native} value = static_cast<{native}>(7);" in output, output
    assert not re.search(r"\b(?:uint64|int64)\b", output), output
