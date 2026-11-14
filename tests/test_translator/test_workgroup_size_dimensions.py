"""A workgroup size that names fewer than three dimensions keeps them, and the omitted ones are 1.

``@ workgroup_size(64)`` used to reach every target as a 1x1x1 workgroup, because only three-argument
attributes were read. CrossGL follows the WGSL ``workgroup_size`` attribute: omitted dimensions are 1, and an
attribute with no dimension or more than three is refused.
"""

import re

import pytest

import crosstl
from crosstl.translator.refusals import CrossGLRefusal

STAGE = """
shader K {{
    compute {{
        buffer float data[];
        {attribute}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            data[i] = data[i] * 2.0;
        }}
    }}
}}
"""

ENTRY = """
@compute
{attribute}
fn scale(@group(0) @binding(0) var<storage, read_write> A: array<f32>) {{
    var i: u32 = gl_GlobalInvocationID.x;
    A[i] = A[i] * 2.0;
}}
"""

# How each target spells a workgroup of (x, y, z).
SPELLINGS = {
    "directx": r"\[numthreads\({x}, {y}, {z}\)\]",
    "opengl": r"local_size_x = {x}, local_size_y = {y}, local_size_z = {z}",
    "vulkan": r"OpExecutionMode %\w+ LocalSize {x} {y} {z}\b",
    "wgsl": r"@workgroup_size\({x}, {y}, {z}\)",
    "slang": r"\[numthreads\({x}, {y}, {z}\)\]",
}


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("template", [STAGE, ENTRY], ids=["stage", "entry"])
@pytest.mark.parametrize(
    ("attribute", "size"),
    [
        ("@workgroup_size(64)", (64, 1, 1)),
        ("@workgroup_size(8, 4)", (8, 4, 1)),
        ("@numthreads(16)", (16, 1, 1)),
        ("@workgroup_size(4, 2, 2)", (4, 2, 2)),
    ],
)
@pytest.mark.parametrize("backend", sorted(SPELLINGS))
def test_omitted_dimensions_are_one(tmp_path, template, attribute, size, backend):
    if template is ENTRY and backend == "slang":
        pytest.skip("the Slang generator does not lower top-level entry functions")
    code = _translate(tmp_path, template.format(attribute=attribute), backend)
    x, y, z = size
    assert re.search(SPELLINGS[backend].format(x=x, y=y, z=z), code), code


@pytest.mark.parametrize("template", [STAGE, ENTRY], ids=["stage", "entry"])
def test_hip_launch_bounds_use_the_whole_size(tmp_path, template):
    code = _translate(
        tmp_path, template.format(attribute="@workgroup_size(8, 4)"), "hip"
    )
    assert "__launch_bounds__(32)" in code, code


@pytest.mark.parametrize("template", [STAGE, ENTRY], ids=["stage", "entry"])
@pytest.mark.parametrize(
    "attribute", ["@workgroup_size()", "@workgroup_size(4, 4, 4, 4)"]
)
def test_a_size_with_no_or_four_dimensions_is_refused(tmp_path, template, attribute):
    with pytest.raises(CrossGLRefusal, match="one, two or three dimensions"):
        _translate(tmp_path, template.format(attribute=attribute), "directx")
