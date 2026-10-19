"""CrossGL's 64-bit integer spellings ``i64`` and ``u64`` reach Direct3D and Metal as their native 64-bit types."""

import re

import pytest

import crosstl

CUDA_PRODUCT = """__global__ void kernel(const unsigned int *inp, unsigned int *outp, long long *wide) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long p = (unsigned long long)inp[i] * 0xFFFFFFFFull;
    long long s = (long long)inp[i] - 4294967296ll;
    wide[i] = s;
    outp[i] = (unsigned int)(p >> 32);
}
"""

BARE_HUB_SPELLING = re.compile(r"\b[iu]64\b")


@pytest.mark.parametrize("backend", ["directx", "metal"])
def test_hub_64bit_integer_types_and_conversions_are_mapped(tmp_path, backend):
    path = tmp_path / "kernel.cu"
    path.write_text(CUDA_PRODUCT)
    output = crosstl.translate(str(path), backend=backend, format_output=False)
    assert not BARE_HUB_SPELLING.search(output), output
    assert re.search(r"uint64_t p = \(?uint64_t\(inp\[i\]\)", output), output
    assert re.search(r"int64_t s = \(?int64_t\(inp\[i\]\)", output), output
