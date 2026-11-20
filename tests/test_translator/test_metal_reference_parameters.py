"""Arguments of Metal helper parameters that are emitted as ``constant T&`` or ``device T&`` references.

A reference in the device or constant address space binds the argument object; no temporary exists there, so a
value conversion around the argument leaves the call without a viable overload. The generator wrapped ``bfloat``
arguments in ``bfloat(...)`` and declared 8- and 16-bit integer references in the widened arithmetic type
(``constant uint&`` for a ``constant uchar&`` buffer), and both outputs failed to compile.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

BFLOAT = PRELUDE + """
void fill_with(device bfloat *out, constant bfloat &value, uint tid) {
    out[tid] = value;
}
kernel void fill(device bfloat *out [[buffer(0)]], constant bfloat &value [[buffer(1)]],
                 uint tid [[thread_position_in_grid]]) {
    fill_with(out, value, tid);
}
"""

NARROW = PRELUDE + """
void bump(device uchar &x, constant uchar &step) { x += step; x = x * 3; }
ushort widen(constant ushort &v) { return v + ushort(1); }
kernel void k(device uchar *out [[buffer(0)]], constant uchar &s [[buffer(1)]],
              constant ushort &w [[buffer(2)]], device ushort *o2 [[buffer(3)]],
              uint i [[thread_position_in_grid]]) {
    bump(out[i], s);
    o2[i] = widen(w);
}
"""


def _translate(tmp_path, source, backend="metal"):
    path = tmp_path / "unit.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile(tmp_path, output):
    if shutil.which("xcrun") is None:
        pytest.skip("requires the Metal compiler")
    artifact = tmp_path / "out.metal"
    artifact.write_text(output)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr


def test_bfloat_reference_argument_is_passed_unconverted(tmp_path):
    output = _translate(tmp_path, BFLOAT)

    assert "constant bfloat& value, uint tid)" in output
    assert "fill_with(out_, value, tid);" in output
    assert "bfloat(value), tid" not in output


def test_narrow_integer_buffer_references_keep_the_storage_type(tmp_path):
    output = _translate(tmp_path, NARROW)

    assert "void bump(device uchar& x, constant uchar& step)" in output
    assert "widen(constant ushort& v)" in output
    assert "bump(out_[i], s);" in output
    assert "widen(w)" in output


@pytest.mark.parametrize("source", (BFLOAT, NARROW), ids=("bfloat", "narrow"))
def test_reference_argument_output_compiles(tmp_path, source):
    _compile(tmp_path, _translate(tmp_path, source))


def test_widened_program_scope_constant_cannot_bind_a_buffer_reference(tmp_path):
    source = NARROW.replace("void bump", "constant uchar K = 7;\nvoid bump", 1).replace(
        "bump(out[i], s);", "bump(out[i], K);"
    )

    with pytest.raises(MetalLoweringRefused) as refusal:
        _translate(tmp_path, source)

    assert refusal.value.code == "project.translate.metal-lowering-unsupported"
    assert "argument 2 of 'bump' is a program-scope uchar object" in str(refusal.value)
