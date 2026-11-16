"""Early refusal of Metal unions that word-based targets cannot store.

The HLSL and GLSL generators refuse non-word unions late, after reverse
translation and the semantics passes; on large Metal kernels that lands past
the 120s corpus worker timeout instead of refusing.  The check in
:func:`crosstl.backend.Metal.early_union_layout.refuse_unsupported_metal_union_layouts`
runs the same validation right after the Metal parse (issue #4525), raising
the exact typed error the late check raises.
"""

import crosstl
import pytest

from crosstl.backend.Metal.early_union_layout import (
    refuse_unsupported_metal_union_layouts,
)
from crosstl.translator.codegen.directx_codegen import DirectXUnionLayoutError
from crosstl.translator.codegen.glsl_union_storage import OpenGLUnionLayoutError
from crosstl.translator.source_registry import SOURCE_REGISTRY

DIRECTX_CODE = "project.translate.directx-union-layout-unsupported"
OPENGL_CODE = "project.translate.opengl-union-layout-unsupported"

TWO_BYTE_UNION = """#include <metal_stdlib>
using namespace metal;
typedef union { half f16; uint16_t u16; } scale_t;
kernel void k(device uint *out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    scale_t s;
    s.u16 = uint16_t(id);
    out[id] = uint(s.f16);
}
"""

FOUR_BYTE_UNION = TWO_BYTE_UNION.replace(
    "typedef union { half f16; uint16_t u16; } scale_t;",
    "typedef union { float f32; uint32_t u32; } scale_t;",
).replace(
    "s.u16 = uint16_t(id);", "s.u32 = uint32_t(id);"
).replace(
    "uint(s.f16)", "uint(s.f32)"
)

MISMATCHED_UNION = TWO_BYTE_UNION.replace(
    "typedef union { half f16; uint16_t u16; } scale_t;",
    "typedef union { float f32; half h16; } scale_t;",
)

POINTER_MEMBER_UNION = TWO_BYTE_UNION.replace(
    "typedef union { half f16; uint16_t u16; } scale_t;",
    "typedef union { half *p16; uint16_t u16; } scale_t;",
)


def _parse_metal(source):
    return SOURCE_REGISTRY.get("metal").parse(
        source,
        file_path="probe.metal",
        include_paths=(),
        defines=None,
        source_options=None,
    )


def _translate(tmp_path, source, backend):
    path = tmp_path / "probe.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def test_two_byte_union_refused_early_for_directx(tmp_path):
    with pytest.raises(DirectXUnionLayoutError) as refusal:
        _translate(tmp_path, TWO_BYTE_UNION, "directx")
    error = refusal.value
    assert error.project_diagnostic_code == DIRECTX_CODE
    assert error.union_name == "scale_t"
    assert error.size == 2
    assert error.reason == "storage-size-unsupported"
    assert "2-byte storage is outside the supported 4/8/16-byte value contract" in str(
        error
    )


def test_two_byte_union_refused_early_for_opengl(tmp_path):
    with pytest.raises(OpenGLUnionLayoutError) as refusal:
        _translate(tmp_path, TWO_BYTE_UNION, "opengl")
    error = refusal.value
    assert error.project_diagnostic_code == OPENGL_CODE
    assert error.union_name == "scale_t"
    assert error.size == 2
    assert error.reason == "storage-size-unsupported"


def test_four_byte_union_translates_to_directx(tmp_path):
    output = _translate(tmp_path, FOUR_BYTE_UNION, "directx")
    assert "CrossGLUnionStorage" in output


def test_vulkan_and_metal_targets_do_not_refuse_union_early():
    program = _parse_metal(TWO_BYTE_UNION)
    assert refuse_unsupported_metal_union_layouts(program, "vulkan") is None
    assert refuse_unsupported_metal_union_layouts(program, "metal") is None
    assert refuse_unsupported_metal_union_layouts(program, "cgl") is None


def test_member_size_mismatch_refused_early():
    program = _parse_metal(MISMATCHED_UNION)
    with pytest.raises(DirectXUnionLayoutError) as refusal:
        refuse_unsupported_metal_union_layouts(program, "directx")
    assert refusal.value.reason == "member-size-mismatch"
    assert refusal.value.size == 4


def test_unprovable_union_member_fails_open():
    program = _parse_metal(POINTER_MEMBER_UNION)
    assert refuse_unsupported_metal_union_layouts(program, "directx") is None
    assert refuse_unsupported_metal_union_layouts(program, "opengl") is None
