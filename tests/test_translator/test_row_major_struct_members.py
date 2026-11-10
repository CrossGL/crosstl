"""Row-major uniform-block members of structure type reach the matrices the structure holds.

GLSL's ``layout(row_major)`` on a uniform block applies to the matrices inside its structure members (GLSL 4.60
section 4.4.5). The HLSL output wrote ``row_major L l;``, which DXC rejects; the Metal output read each nested matrix
transposed; the SPIR-V output refused the member, and the Slang output refused it too. Such a member now takes a
row-major copy of its structure (semantics/row_major_struct_members.py). Runs of the outputs on MoltenVK (SPIR-V,
DXC -spirv) and Metal matched glslang's reference.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import RowMajorStructMemberUnsupportedError

NESTED = """#version 450
layout(local_size_x = 1) in;
struct L { mat2 n; vec4 c; };
layout(std140, row_major, binding = 0) uniform I { L l; mat2 top; } src;
layout(std430, binding = 1) buffer O { vec4 o[2]; } dst;
void main() { dst.o[0] = vec4(src.l.n[0], src.l.n[1]); dst.o[1] = vec4(src.top[0], src.top[1]); }
"""

DEEP = """#version 450
layout(local_size_x = 1) in;
struct Inner { mat2x3 n; };
struct L { Inner inner; vec4 c; mat3x2 k; };
layout(std140, row_major, binding = 0) uniform I { L ls[2]; float f; } src;
layout(std430, binding = 1) buffer O { vec4 o[2]; } dst;
void main() {
    dst.o[0] = vec4(src.ls[1].inner.n[0], src.ls[0].inner.n[1].z);
    dst.o[1] = vec4(src.ls[1].k[2], src.ls[0].k[1].y, src.f);
}
"""

WHOLE = """#version 450
layout(local_size_x = 1) in;
struct L { mat2 n; vec4 c; };
layout(std140, row_major, binding = 0) uniform I { L l; } src;
layout(std430, binding = 1) buffer O { vec4 o[2]; } dst;
void main() { L copy = src.l; dst.o[0] = vec4(copy.n[0], copy.n[1]); }
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.comp"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_hlsl_family_declares_a_row_major_copy(tmp_path, backend):
    code = _flat(_translate(tmp_path, NESTED, backend))
    assert re.search(
        r"struct L_row_major \{ row_major float2x2 n; float4 c; \};", code
    ), code
    assert re.search(r"\bL_row_major l;", code), code
    assert "row_major L l" not in code, code


def test_spirv_decorates_the_copy_row_major(tmp_path):
    code = _translate(tmp_path, NESTED, "vulkan")
    names = dict(re.findall(r'OpName %(\S+) "(\w+)"', code))
    row_major = {
        names.get(type_id)
        for type_id in re.findall(r"OpMemberDecorate %(\S+) 0 RowMajor", code)
    }
    # The std140 layout copy of L_row_major decorates its matrix member n RowMajor.
    assert any(name and name.startswith("L_row_major") for name in row_major), code


def test_metal_reads_the_nested_matrix_as_row_major(tmp_path):
    code = _flat(_translate(tmp_path, NESTED, "metal"))
    assert "crosstl_hlsl_cbuffer_float2x2_row_major_padded_load(i.l.n)" in code, code


def test_glsl_keeps_the_member_order(tmp_path):
    code = _flat(_translate(tmp_path, NESTED, "opengl"))
    assert "layout(row_major) L l;" in code, code
    assert "L_row_major" not in code


def test_nested_arrays_and_non_square_matrices_lay_out_validly(tmp_path):
    code = _flat(_translate(tmp_path, DEEP, "directx"))
    assert re.search(r"struct Inner_row_major \{ row_major float3x2 n; \};", code), code
    assert re.search(
        r"Inner_row_major inner; float4 c; row_major float2x3 k;", code
    ), code
    assert re.search(r"\bL_row_major ls\[2\]\s*(:\s*packoffset\(c0\))?;", code), code
    spirv = _translate(tmp_path, DEEP, "vulkan")
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        source = tmp_path / "deep.spvasm"
        source.write_text(spirv, encoding="utf-8")
        binary = tmp_path / "deep.spv"
        assembled = subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
            capture_output=True,
            text=True,
        )
        assert assembled.returncode == 0, assembled.stderr
        # A row-major mat2x3 holds three 16-byte rows, so the next member starts at offset 48, not 32.
        validated = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
            capture_output=True,
            text=True,
        )
        assert validated.returncode == 0, validated.stderr


@pytest.mark.parametrize("backend", ["directx", "metal", "vulkan", "slang"])
def test_whole_member_reads_are_refused(tmp_path, backend):
    with pytest.raises(
        RowMajorStructMemberUnsupportedError,
        match="reads the whole row-major member 'l'",
    ):
        _translate(tmp_path, WHOLE, backend)


def test_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    source = tmp_path / "deep.hlsl"
    source.write_text(_translate(tmp_path, DEEP, "directx"), encoding="utf-8")
    completed = subprocess.run(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(source),
            "-Fo",
            str(tmp_path / "d.dxo"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


BUFFER = """
shader main {
    struct Inner {
        mat2x3 n;
    };
    struct L {
        Inner inner;
        vec4 c;
        mat3x2 k;
    };
    struct I {
        L ls[2] @row_major;
        float f;
    };
    struct O {
        vec4 o[2];
    };
    I src @glsl_buffer_block(std430) @binding(0);
    O dst @glsl_buffer_block(std430) @binding(1);
    compute {
        layout(local_size_x = 1) in;
        void main() {
            dst.o[0] = vec4(src.ls[1].inner.n[0], src.ls[0].inner.n[1].z);
            dst.o[1] = vec4(src.ls[1].k[2], src.ls[0].k[1].y, src.f);
        }
    }
}
"""

BUFFER_WHOLE = """
shader main {
    struct L {
        mat2 n;
        vec4 c;
    };
    struct Src {
        L l @row_major;
    };
    struct O {
        vec4 o[2];
    };
    Src src @glsl_buffer_block(std430) @binding(0);
    O dst @glsl_buffer_block(std430) @binding(1);
    compute {
        layout(local_size_x = 1) in;
        void main() {
            L copy = src.l;
            dst.o[0] = vec4(copy.n[0], copy.n[1]);
        }
    }
}
"""


def _translate_cgl(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_buffer_block_copies_reach_the_byte_address_loads(tmp_path):
    # The copy stores its matrices transposed (row_major_members.py): CrossGL mat2x3 is HLSL float3x2, and its
    # row-major copy is declared float2x3 and read back through transpose.
    code = _flat(_translate_cgl(tmp_path, BUFFER, "directx"))
    assert "struct Inner_row_major { float2x3 n; };" in code, code
    assert "transpose(transpose(" in code, code


@pytest.mark.parametrize("backend", ["vulkan", "metal", "slang"])
def test_buffer_block_copies_translate(tmp_path, backend):
    code = _translate_cgl(tmp_path, BUFFER, backend)
    assert "L_row_major" in code


@pytest.mark.parametrize("backend", ["directx", "metal", "vulkan", "slang"])
def test_whole_buffer_member_reads_are_refused(tmp_path, backend):
    with pytest.raises(
        RowMajorStructMemberUnsupportedError,
        match="reads the whole row-major member 'l' of 'Src'",
    ):
        _translate_cgl(tmp_path, BUFFER_WHOLE, backend)


def test_buffer_block_hlsl_compiles_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    source = tmp_path / "buffer.hlsl"
    source.write_text(_translate_cgl(tmp_path, BUFFER, "directx"), encoding="utf-8")
    completed = subprocess.run(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(source),
            "-Fo",
            str(tmp_path / "b.dxo"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
