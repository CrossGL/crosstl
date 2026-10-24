"""Type traits in explicitly instantiated Metal kernel templates fold to their values.

ggml instantiates kernel templates with ``template [[host_name("...")]] kernel void k<float4>(...);`` and branches on
``is_same<float4, T0>::value`` inside them. The preprocessor folded the traits of the source's own concrete code but
not of the instantiations it materialized, so the importer mangled the trait into an undeclared identifier
(``is_same_u3cfloat4_u2cfloat_u3e_u3a_u3avalue``).
"""

import shutil
import subprocess

import pytest

import crosstl

SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename T0>
kernel void sum_rows(device const T0 *src [[buffer(0)]],
                     device float *dst [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
    if (is_same<float4, T0>::value) {
        dst[tid] = 4.0f;
    } else {
        dst[tid] = 1.0f;
    }
}

template [[host_name("sum_rows_f32")]] kernel void sum_rows<float>(device const float *, device float *, uint);
template [[host_name("sum_rows_f32_4")]] kernel void sum_rows<float4>(device const float4 *, device float *, uint);
"""


def _translate(tmp_path, backend):
    path = tmp_path / "sum_rows.metal"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, source_options={"threadgroup_size": (32, 1, 1)}
    )


def test_each_instantiation_folds_its_trait(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "is_same" not in crossgl
    assert crossgl.count("if (false)") == 1
    assert crossgl.count("if (true)") == 1


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "vulkan"])
def test_instantiations_translate(tmp_path, backend):
    code = _translate(tmp_path, backend)
    assert "is_same" not in code
    if backend == "metal" and shutil.which("xcrun"):
        if subprocess.run(
            ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode:
            return
        out = tmp_path / "out.metal"
        out.write_text(code, encoding="utf-8")
        completed = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(out),
                "-o",
                str(tmp_path / "out.air"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
