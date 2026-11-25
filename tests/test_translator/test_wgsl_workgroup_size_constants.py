"""WGSL output sizes a workgroup through the program's own override of the dimension's specialization id.

``const uint size @constant_id(0) = 1u;`` beside ``layout(local_size_x_id = 0) in;`` (GLSL's
``layout(constant_id = 0)`` and the WGSL front end's import of ``override size`` named in ``@workgroup_size``) is the
module's ``@id(0) override``. WGSL output used to declare a second ``@id(0) override crossgl_workgroup_size_x``, which
naga rejects because pipeline constant ids are unique within a module; it now names ``size`` in ``@workgroup_size``.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.specialized_workgroup_size import (
    SpecializedWorkgroupSizeError,
)

PROGRAM = """shader Probe {{
    const {decl} @constant_id(0) = {default};
    RWStructuredBuffer<float> fin @set(0) @binding(0);
    RWStructuredBuffer<float> outp @set(0) @binding(1);
    compute {{
        layout({layout}) in;
        void main(uvec3 id @ gl_GlobalInvocationID) {{
            uint i = id.x;
            uvec3 g = gl_WorkGroupSize;
            outp[i] = fin[i] * float(size) + float(g.x + g.y);
        }}
    }}
}}
"""


def _translate(tmp_path, decl="uint size", default="64u", layout="local_size_x_id = 0"):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM.format(decl=decl, default=default, layout=layout))
    code = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(code)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return code


def test_dimension_names_the_constant_of_its_id(tmp_path):
    code = _translate(tmp_path)
    assert "@id(0)\noverride size: u32 = 64u;" in code
    assert "@workgroup_size(size, 1, 1)" in code
    assert "crossgl_workgroup_size_x" not in code
    assert code.count("@id(0)") == 1
    assert "vec3<u32>(size, 1u, 1u)" in code


def test_other_dimensions_keep_their_own_overrides(tmp_path):
    code = _translate(tmp_path, layout="local_size_x_id = 0, local_size_y_id = 5")
    assert "@workgroup_size(size, crossgl_workgroup_size_y, 1)" in code
    assert "@id(5)\noverride crossgl_workgroup_size_y: u32 = 1u;" in code


def test_signed_constant_gives_every_override_its_type(tmp_path):
    code = _translate(
        tmp_path,
        decl="int size",
        default="8",
        layout="local_size_x_id = 0, local_size_y_id = 1",
    )
    assert "@id(0)\noverride size: i32 = 8;" in code
    # @workgroup_size arguments all have one type (WGSL section 9.3.6).
    assert "@id(1)\noverride crossgl_workgroup_size_y: i32 = 1;" in code
    assert "@workgroup_size(size, crossgl_workgroup_size_y, 1)" in code
    # gl_WorkGroupSize is a vec3<u32>.
    assert "vec3<u32>(u32(size), u32(crossgl_workgroup_size_y), 1u)" in code


def test_constant_of_a_type_wgsl_cannot_size_with_is_refused(tmp_path):
    with pytest.raises(
        SpecializedWorkgroupSizeError, match="is an i32 or u32"
    ) as refused:
        _translate(tmp_path, decl="float size", default="8.0")
    assert refused.value.specialization_ids == (0,)
