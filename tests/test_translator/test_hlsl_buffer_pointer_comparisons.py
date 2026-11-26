"""HLSL output of comparisons and truth tests of CUDA, HIP and OpenCL buffer pointers.

The generator keeps a pointer into a buffer as the buffer and an element offset. Two pointers into one buffer
compare as their offsets; a truth test of a pointer, or a comparison with a null pointer, another buffer or an
integer, has no HLSL meaning (a bound resource has no null state) and is refused with a typed diagnostic.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused

KERNEL = """
__global__ void k(const float *inp, float *out, int n, bool flag)
{{
    int i = threadIdx.x;
    float *p = out + i;
    float *q = out;
    {body}
}}
"""


def _translate(tmp_path, body, suffix=".cu"):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(KERNEL.format(body=body), encoding="utf-8")
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _dxc(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "kernel.hlsl"
    path.write_text(code, encoding="utf-8")
    sizes = [f"-DCROSSGL_WORKGROUP_SIZE_{axis}=1" for axis in "XYZ"]
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", "-WX", *sizes, str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize(
    "body",
    [
        "out[i] = inp ? inp[i] : out[i];",
        "if (inp) { out[i] = 1.0f; }",
        "if (!inp) { out[i] = 1.0f; }",
        "while (p && flag) { *p = 1.0f; break; }",
        "out[i] = (flag || q) ? 1.0f : 0.0f;",
        "bool has = inp; out[i] = has ? 1.0f : 0.0f;",
    ],
)
@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_truth_tests_of_buffer_pointers_are_refused(tmp_path, body, suffix):
    with pytest.raises(DirectXLoweringRefused, match="used as a truth value"):
        _translate(tmp_path, body, suffix)


@pytest.mark.parametrize(
    "body",
    [
        "if (inp == nullptr) { out[i] = 1.0f; }",
        "if (inp != 0) { out[i] = 1.0f; }",
        "if (p != NULL) { *p = 1.0f; }",
    ],
)
def test_null_comparisons_of_buffer_pointers_are_refused(tmp_path, body):
    with pytest.raises(DirectXLoweringRefused, match="compared with a non-pointer"):
        _translate(tmp_path, body)


def test_pointers_into_different_buffers_are_refused(tmp_path):
    with pytest.raises(
        DirectXLoweringRefused, match="comparison of buffer pointers"
    ) as error:
        _translate(tmp_path, "if (p == inp) { out[i] = 1.0f; }")
    assert error.value.reasons and len(error.value.reasons) == 1


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("if (p != q) { *p = 1.0f; }", r"if \(\(p_offset\) != \(q_offset\)\)"),
        ("if (p >= q) { *p = 1.0f; }", r"if \(\(p_offset\) >= \(q_offset\)\)"),
        (
            "out[i] = (p == out + i) ? 1.0f : 0.0f;",
            r"\(\(\(p_offset\) == \(i\)\) \?",
        ),
        (
            "float *e = out + n; while (p < e) { *p = 2.0f; ++p; }",
            r"while \(+\(p_offset\) < \(e_offset\)\)",
        ),
    ],
)
def test_pointers_into_one_buffer_compare_their_offsets(tmp_path, body, expected):
    code = _translate(tmp_path, body)
    assert re.search(expected, code), code[code.find("CSMain") :]
    # No pointer local the generator replaced by an offset is left in the output.
    assert not re.search(r"\b[pqe]\b(?!_offset)", code[code.find("CSMain") :])
    _dxc(tmp_path, code)
