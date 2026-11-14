"""A scalar kernel argument bound to a buffer slot is ``constant T&`` in Metal, whatever its width.

Only 32-bit and floating scalars were recognised, so a 64-bit, 16-bit or 8-bit uniform became a bare kernel
parameter (``uint64_t size``), which ``xcrun metal`` rejects as a stage input of a kernel. The reference keeps the
source width, so a two-byte host value is not read as four bytes.
"""

import shutil
import subprocess

import pytest

import crosstl

PROGRAM = """
@compute
@workgroup_size(64, 1, 1)
fn scale(
    @group(0) @binding(0) var<storage, read_write> data: array<u32>,
    @group(0) @binding(1) var<uniform> count: {scalar}
) {{
    var i: u32 = gl_GlobalInvocationID.x;
    if ({compare}) {{
        data[i] = data[i] + 1u;
    }}
}}
"""


@pytest.mark.parametrize(
    ("scalar", "metal_type", "compare"),
    [
        ("u64", "uint64_t", "u64(i) < count"),
        ("i64", "int64_t", "i64(i) < count"),
        ("u16", "ushort", "u32(count) > i"),
        ("i8", "char", "i32(count) > i32(i)"),
    ],
)
def test_narrow_and_wide_scalar_uniforms_are_constant_references(
    tmp_path, scalar, metal_type, compare
):
    source = tmp_path / "scale.cgl"
    source.write_text(PROGRAM.format(scalar=scalar, compare=compare), encoding="utf-8")

    metal = crosstl.translate(str(source), backend="metal", format_output=False)

    assert f"constant {metal_type}& count [[buffer(1)]]" in metal

    compiler = shutil.which("xcrun")
    if compiler is None:
        return
    shader = tmp_path / "scale.metal"
    shader.write_text(metal, encoding="utf-8")
    completed = subprocess.run(
        [
            compiler,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "scale.air"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
