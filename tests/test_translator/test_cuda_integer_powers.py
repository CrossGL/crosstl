"""CrossGL ``powi``: HIP's ``powif``/``powi`` in both directions and CUDA's double-precision ``pow``."""

import pytest

import crosstl
from crosstl.translator.codegen.cuda_refusals import CudaLoweringRefused
from crosstl.translator.codegen.hip_refusals import HipLoweringRefused

CROSSGL = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* iin;
        buffer uint* uin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            BODY
        }
    }
}
"""

HIP = """#include <hip/hip_runtime.h>
extern "C" __global__ void probe(const float* fin, const int* iin, float* outp, double* dout) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = powif(fin[i], iin[i]);
    dout[i] = powi((double)fin[i], -iin[i]);
}
"""


def _translate(tmp_path, body, backend, name="probe.cgl"):
    path = tmp_path / name
    path.write_text(CROSSGL.replace("BODY", body))
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize(
    "body, hip, cuda",
    [
        (
            "outp[i] = powi(fin[i], iin[i]);",
            "powif(fin[i], iin[i])",
            "(float)pow((double)(fin[i]), (double)(iin[i]))",
        ),
        (
            "double d = powi(double(fin[i]), iin[i]); outp[i] = float(d);",
            "powi(double(fin[i]), iin[i])",
            "pow((double)(double(fin[i])), (double)(iin[i]))",
        ),
        (
            "outp[i] = powi(fin[i], uin[i]);",
            "powif(fin[i], (int)(uin[i]))",
            "(float)pow((double)(fin[i]), (double)((int)(uin[i])))",
        ),
    ],
)
def test_powi_writes_each_precision(tmp_path, body, hip, cuda):
    assert hip in _translate(tmp_path, body, "hip")
    assert cuda in _translate(tmp_path, body, "cuda")


@pytest.mark.parametrize(
    "backend, refusal", [("hip", HipLoweringRefused), ("cuda", CudaLoweringRefused)]
)
def test_powi_of_a_vector_base_is_refused(tmp_path, backend, refusal):
    body = "vec2 v = powi(vec2(fin[i], 1.0), 2); outp[i] = v.x;"
    with pytest.raises(refusal, match="integer power: powi of a vector base"):
        _translate(tmp_path, body, backend)


def test_user_function_named_powi_is_called(tmp_path):
    source = CROSSGL.replace(
        "        void main() {",
        "        float powi(float x, int n) { return x + float(n); }\n        void main() {",
    ).replace("BODY", "outp[i] = powi(fin[i], iin[i]);")
    path = tmp_path / "user.cgl"
    path.write_text(source)
    for backend in ("hip", "cuda"):
        code = crosstl.translate(str(path), backend=backend)
        assert "powif(" not in code and "pow((double)" not in code


def test_hip_input_reads_both_precisions_as_powi(tmp_path):
    path = tmp_path / "powers.hip"
    path.write_text(HIP)
    crossgl = crosstl.translate(str(path), backend="cgl")
    assert "powi(fin[i], iin[i])" in crossgl
    assert "powif" not in crossgl
    hip = crosstl.translate(str(path), backend="hip")
    assert "powif(fin[i], iin[i])" in hip
    assert "powi((double)(" in hip or "powi(f64(" in hip or "powi(" in hip
    cuda = crosstl.translate(str(path), backend="cuda")
    assert "powi" not in cuda and "powif" not in cuda
    assert "(float)pow((double)(fin[i]), (double)(iin[i]))" in cuda


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "vulkan", "wgsl"])
def test_targets_without_an_integer_power_refuse_it(tmp_path, backend):
    path = tmp_path / "powers.hip"
    single = HIP.replace("    dout[i] = powi((double)fin[i], -iin[i]);\n", "")
    path.write_text(single.replace(", double* dout", ""))
    with pytest.raises(Exception, match="powi") as refusal:
        crosstl.translate(str(path), backend=backend)
    assert type(refusal.value).__module__.startswith("crosstl."), "a typed refusal"
