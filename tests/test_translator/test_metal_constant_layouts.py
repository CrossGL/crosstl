"""MSL constant buffers keep their natural layout, packed vectors included, on every target.

An MSL ``constant Params& params [[buffer(N)]]`` holds one ``Params`` in the natural layout of MSL: ``float3`` takes
16 bytes, so a following ``float`` sits 16 bytes on, and a ``packed_float3`` takes 12 bytes at 4-byte alignment
(MSL 3.2 section 2.2 and table 2.4). OpenGL and SPIR-V would bind the reference as a std140 block and HLSL as a
``ConstantBuffer<Params>``, which place both members elsewhere. CrossGL keeps packed members as ``@ packed`` and a
reference in MSL's layout as ``@ buffer_layout(metal) constant Params& params``; the other targets then read every
member from the bytes the host wrote. The expected offsets are the ones the Metal compiler reports through
``__builtin_offsetof``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_block_layout import (
    METAL_LAYOUT,
    SCALAR_BLOCK_LAYOUT,
    STD140_LAYOUT,
    block_struct_placements,
    packed_layout_type,
)
from crosstl.translator.codegen.glsl_hlsl_cbuffer import layout_scalar_offsets
from crosstl.translator.codegen.hlsl_buffer_layout import (
    HLSL_CBUFFER_LAYOUT,
    HlslLayoutError,
    array_type,
    oriented_matrix_type,
    scalar_type,
    struct_type,
    vector_type,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    ConstantReferenceLayoutError,
    PackedMemberLayoutError,
)
from crosstl.translator.semantics.constant_reference_layouts import (
    hoist_constant_reference_layouts,
)
from crosstl.translator.semantics.packed_members import refuse_unplaced_packed_members
from tests.spirv_friendly_names import friendly_spirv

METAL_PARAMS = """#include <metal_stdlib>
using namespace metal;
struct Params {
    float3 scale;
    float bias;
    packed_float3 shift;
    float2 offsets[3];
    float3x3 rot;
    int mode;
    float weights[2];
    float2 tail;
};
kernel void hazard(device float* o [[buffer(0)]], constant Params& params [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    float3 r = params.rot * float3(1.0, 2.0, 3.0);
    o[i] = params.scale.y + params.bias + params.shift.z + params.offsets[i % 3].y + r[i % 3]
         + float(params.mode) + params.weights[i % 2] + params.tail.y;
}
"""
# The member offsets ``__builtin_offsetof`` reports for ``Params``.
PARAMS_OFFSETS = {
    "scale": 0,
    "bias": 16,
    "shift": 20,
    "offsets": 32,
    "rot": 64,
    "mode": 112,
    "weights": 116,
    "tail": 128,
}

METAL_SCALARS = """#include <metal_stdlib>
using namespace metal;
struct Scalars {
    float a;
    int b;
    uint c;
};
kernel void scalars(device float* o [[buffer(0)]], constant Scalars& s [[buffer(1)]],
                    constant float& k [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    o[i] = s.a + float(s.b) + float(s.c) + k;
}
"""

# METAL_PARAMS in CrossGL, with the constant reference marked as holding MSL's layout.
CROSSGL_FULL = """shader main {
    struct Params {
        vec3 scale;
        float bias;
        @packed vec3 shift;
        vec2[3] offsets;
        mat3 rot;
        int mode;
        float[2] weights;
        vec2 tail;
    }
    compute {
        @stage_entry
        void hazard(RWStructuredBuffer<float> o @buffer(0),
                    @buffer_layout(metal) constant Params& params @buffer(2),
                    uint i @gl_GlobalInvocationID) {
            vec3 r = params.rot * vec3(1.0, 2.0, 3.0);
            buffer_store(o, i, params.scale.y + params.bias + params.shift.z + params.offsets[i % 3].y + r.y
                + float(params.mode) + params.weights[i % 2] + params.tail.y);
        }
    }
}
"""

CROSSGL_PARAMS = """shader main {
    struct Params {
        vec3 scale;
        float bias;
        @packed vec3 shift;
        vec2 offsets[3];
    }
    compute {
        @stage_entry
        void hazard(RWStructuredBuffer<float> o @buffer(0),
                    @buffer_layout(metal) constant Params& params @buffer(2),
                    uint i @gl_GlobalInvocationID) {
            buffer_store(o, i, params.scale.y + params.bias + params.shift.z + params.offsets[i % 3].y);
        }
    }
}
"""


def _params_layout_type():
    return struct_type(
        "Params",
        [
            ("scale", vector_type("float", 3)),
            ("bias", scalar_type("float")),
            ("shift", packed_layout_type(vector_type("float", 3), "shift")),
            ("offsets", array_type(vector_type("float", 2), 3)),
            ("rot", oriented_matrix_type("float", 3, 3)),
            ("mode", scalar_type("int")),
            ("weights", array_type(scalar_type("float"), 2)),
            ("tail", vector_type("float", 2)),
        ],
    )


def _translate(tmp_path, source, name, backend, source_backend=None):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    if backend == "vulkan":
        return friendly_spirv(
            crosstl.translate(
                str(path),
                backend=backend,
                source_backend=source_backend,
                format_output=False,
            )
        )
    return crosstl.translate(str(path), backend=backend, source_backend=source_backend)


def _run(command, cwd=None):
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, cwd=cwd
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    return result


def _metal_compiler():
    if not shutil.which("xcrun"):
        return False
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True, text=True
    )
    return probe.returncode == 0


def _metal_offsets(tmp_path, code, struct, offsets):
    """Compile ``code`` with static assertions of the byte offsets of ``struct``'s members."""
    if not _metal_compiler():
        pytest.skip("the Metal compiler is not available")
    checks = "".join(
        f'static_assert(__builtin_offsetof({struct}, {member}) == {offset}, "{member}");\n'
        for member, offset in offsets.items()
    )
    path = tmp_path / "offsets.metal"
    path.write_text(code + "\n" + checks, encoding="utf-8")
    _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "offsets.air"),
        ]
    )


METAL_MIXED = """#include <metal_stdlib>
using namespace metal;
struct Q {
    float3 a;
    float s;
    packed_float3 b;
    float c;
    float2 d;
};
kernel void mixed(device const Q* src [[buffer(0)]], device float* o [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    Q q = src[i];
    o[i] = q.a.z + q.s + q.b.x + q.c + q.d.y;
}
"""
# ``__builtin_offsetof`` of Q's members; the element takes 48 bytes (12 words).
MIXED_OFFSETS = {"a": 0, "s": 16, "b": 20, "c": 32, "d": 40}


# Planner ---------------------------------------------------------------------
def test_metal_rules_place_members_at_msl_offsets():
    placements, size = block_struct_placements(_params_layout_type(), METAL_LAYOUT)
    assert {p.name: p.offset for p in placements} == PARAMS_OFFSETS
    # The struct is aligned to its 16-byte members.
    assert size == 144


def test_metal_rules_size_vec3_as_four_components():
    layout_type = struct_type(
        "S", [("a", vector_type("float", 3)), ("b", scalar_type("float"))]
    )
    assert layout_scalar_offsets(layout_type, METAL_LAYOUT) == [0, 4, 8, 16]
    assert layout_scalar_offsets(layout_type, STD140_LAYOUT) == [0, 4, 8, 12]


def test_packed_vectors_take_their_components_under_metal_and_scalar_rules():
    packed = packed_layout_type(array_type(vector_type("float", 3), 2), "v")
    layout_type = struct_type("S", [("f", scalar_type("float")), ("v", packed)])
    expected = [0, 4, 8, 12, 16, 20, 24]
    assert layout_scalar_offsets(layout_type, METAL_LAYOUT) == expected
    assert layout_scalar_offsets(layout_type, SCALAR_BLOCK_LAYOUT) == expected


@pytest.mark.parametrize("layout", ["std140", "std430", HLSL_CBUFFER_LAYOUT])
def test_packed_vectors_are_refused_by_rules_without_packed_storage(layout):
    layout_type = struct_type(
        "S", [("v", packed_layout_type(vector_type("float", 3), "v"))]
    )
    with pytest.raises(HlslLayoutError, match="packed vector"):
        layout_scalar_offsets(layout_type, layout)


def test_metal_bool_and_packed_scalars_are_refused():
    with pytest.raises(HlslLayoutError, match="one byte"):
        layout_scalar_offsets(
            struct_type("S", [("b", scalar_type("bool"))]), METAL_LAYOUT
        )
    with pytest.raises(HlslLayoutError, match="only vectors"):
        packed_layout_type(scalar_type("float"), "s")


# Metal import ----------------------------------------------------------------
def test_metal_import_keeps_packed_members_and_marks_the_constant_reference(
    tmp_path,
):
    code = _translate(tmp_path, METAL_PARAMS, "shader.metal", "cgl")
    assert re.search(r"@packed vec3 shift;", code)
    assert re.search(r"\n\s*vec3 scale;", code)
    # std140 and HLSL constant packing have no packed vector: the reference keeps MSL's layout.
    assert "@buffer_layout(metal) constant Params& params @buffer(2)" in code


def test_metal_constant_reference_without_packed_members_keeps_the_target_layout(
    tmp_path,
):
    # The project reflection reports the target's constant-buffer layout to the host for such references.
    source = METAL_PARAMS.replace("packed_float3 shift;", "float3 shift;")
    code = _translate(tmp_path, source, "shader.metal", "cgl")
    assert "buffer_layout" not in code
    assert "constant Params& params @buffer(2)" in code


def test_metal_import_leaves_constant_references_every_target_lays_out_alike(
    tmp_path,
):
    code = _translate(tmp_path, METAL_SCALARS, "shader.metal", "cgl")
    assert "buffer_layout" not in code
    assert "constant Scalars& s @buffer(1)" in code


# Targets ---------------------------------------------------------------------
def test_metal_round_trip_declares_packed_storage(tmp_path):
    code = _translate(tmp_path, METAL_PARAMS, "shader.metal", "metal")
    assert "packed_float3 shift;" in code
    assert re.search(r"constant Params\s*&\s*params\s*\[\[buffer\(2\)\]\]", code)
    _metal_offsets(tmp_path, code, "Params", PARAMS_OFFSETS)


def test_crossgl_packed_member_compiles_as_msl_packed_vector(tmp_path):
    code = _translate(tmp_path, CROSSGL_PARAMS, "shader.cgl", "metal")
    assert "packed_float3 shift;" in code
    assert "[[packed]]" not in code
    _metal_offsets(
        tmp_path, code, "Params", {"scale": 0, "bias": 16, "shift": 20, "offsets": 32}
    )


def test_opengl_reads_constant_reference_at_msl_offsets(tmp_path):
    code = _translate(tmp_path, METAL_PARAMS, "shader.metal", "opengl")
    assert re.search(r"binding = 2\) uniform params_block \{", code)
    # bias at byte 16 and shift at 20..32: register 1, then the matrix columns in registers 4 to 6.
    assert "params.bias = uintBitsToFloat(crossgl_cbuffer_registers[1].x);" in code
    assert "params.shift = uintBitsToFloat(crossgl_cbuffer_registers[1].yzw);" in code
    assert "params.rot[2] = uintBitsToFloat(crossgl_cbuffer_registers[6].xyz);" in code
    assert "params.mode = int(crossgl_cbuffer_registers[7].x);" in code
    # weights at 116 and 120 are words 29 and 30; tail at 128.
    assert "+ 29));" in code
    assert "params.tail = uintBitsToFloat(crossgl_cbuffer_registers[8].xy);" in code
    validator = shutil.which("glslangValidator")
    if validator is not None:
        path = tmp_path / "shader.comp"
        path.write_text(code, encoding="utf-8")
        _run([validator, str(path)])


def test_hlsl_reads_constant_reference_at_msl_offsets(tmp_path):
    code = _translate(tmp_path, METAL_PARAMS, "shader.metal", "directx")
    assert "cbuffer params_block : register(b2)" in code
    # ``@ packed`` is a storage layout, not a semantic.
    assert ": packed" not in code
    assert "uint4 __crossgl_params_block_registers[9];" in code
    assert "params.shift = asfloat(__crossgl_params_block_registers[1].yzw);" in code
    assert (
        "params.rot._m00_m10_m20 = asfloat(__crossgl_params_block_registers[4].xyz);"
        in code
    )
    assert "params.tail = asfloat(__crossgl_params_block_registers[8].xy);" in code
    compiler = shutil.which("dxc")
    if compiler is not None:
        path = tmp_path / "shader.hlsl"
        path.write_text(code, encoding="utf-8")
        _run(
            [compiler, "-T", "cs_6_0", "-E", "CSMain", "-Fo", "shader.dxil", path.name],
            cwd=tmp_path,
        )


def test_spirv_decorates_constant_reference_with_msl_offsets(tmp_path):
    code = _translate(tmp_path, METAL_PARAMS, "shader.metal", "vulkan")
    struct = re.search(r"OpName %(Params_placed_metal\w*) ", code).group(1)
    offsets = {
        int(index): int(offset)
        for index, offset in re.findall(
            rf"OpMemberDecorate %{struct} (\d+) Offset (\d+)", code
        )
    }
    assert [offsets[index] for index in sorted(offsets)] == list(
        PARAMS_OFFSETS.values()
    )
    assert 'OpSourceExtension "GL_EXT_scalar_block_layout"' in code
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    asm, binary = tmp_path / "shader.spvasm", tmp_path / "shader.spv"
    asm.write_text(code, encoding="utf-8")
    _run([assembler, "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
    _run([validator, "--target-env", "vulkan1.1", "--scalar-block-layout", str(binary)])


# Refusals ----------------------------------------------------------------------
def _program(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_marked_constant_reference_of_a_helper_is_refused():
    source = """shader main {
        struct P { vec3 a; float b; }
        float read(@buffer_layout(metal) constant P& p) { return p.b; }
        compute {
            @stage_entry
            void k(RWStructuredBuffer<float> o @buffer(0), uint i @gl_GlobalInvocationID) {
                buffer_store(o, i, 1.0);
            }
        }
    }
    """
    with pytest.raises(ConstantReferenceLayoutError, match="only a stage entry"):
        hoist_constant_reference_layouts(_program(source), "opengl")


def test_conflicting_constant_references_of_two_entries_are_refused():
    source = """shader main {
        struct P { vec3 a; float b; }
        struct Q { vec3 a; float c; }
        vertex {
            @stage_entry
            vec4 vs(@buffer_layout(metal) constant P& p @buffer(1)) { return vec4(p.a, p.b); }
        }
        fragment {
            @stage_entry
            vec4 fs(@buffer_layout(metal) constant Q& p @buffer(1)) { return vec4(p.a, p.c); }
        }
    }
    """
    with pytest.raises(ConstantReferenceLayoutError, match="differs in type"):
        hoist_constant_reference_layouts(_program(source), "directx")


def test_metal_and_other_targets_keep_the_parameter():
    program = _program(CROSSGL_PARAMS)
    assert hoist_constant_reference_layouts(program, "metal") == 0
    assert hoist_constant_reference_layouts(program, "cuda") == 0
    assert not program.cbuffers


def test_packed_member_in_a_std140_block_is_refused(tmp_path):
    source = """shader main {
    struct P { float f; @packed vec3 v; }
    cbuffer Block @buffer(0) @buffer_layout(std140) { P p; }
    compute {
        @stage_entry
        void k(RWStructuredBuffer<float> o @buffer(1), uint i @gl_GlobalInvocationID) {
            buffer_store(o, i, p.v.y);
        }
    }
}
"""
    for backend in ("opengl", "directx", "vulkan"):
        with pytest.raises(PackedMemberLayoutError, match="std140 layout"):
            _translate(tmp_path, source, "shader.cgl", backend)


def test_unmarked_constant_reference_with_moving_packed_members_is_refused(tmp_path):
    # Without a declared layout the targets' default rules would move shift and every member after it.
    source = CROSSGL_FULL.replace("@buffer_layout(metal) constant", "constant")
    for backend in ("opengl", "directx", "vulkan"):
        with pytest.raises(PackedMemberLayoutError, match="'hazard.params'"):
            _translate(tmp_path, source, "shader.cgl", backend)


def test_helper_reference_to_a_packed_struct_is_not_storage():
    source = """shader main {
    struct P { float f; @packed vec3 v; float g; }
    float read(constant P& p) { return p.v.y + p.g; }
    compute {
        @stage_entry
        void k(RWStructuredBuffer<float> o @buffer(0), uint i @gl_GlobalInvocationID) {
            buffer_store(o, i, 1.0);
        }
    }
}
"""
    refuse_unplaced_packed_members(_program(source), "opengl")


def test_packed_members_that_move_nothing_keep_the_default_layout():
    source = """shader main {
    struct P { vec4 a; @packed vec3 b; float c; }
    compute {
        @stage_entry
        void k(StructuredBuffer<P> src @buffer(0), RWStructuredBuffer<float> o @buffer(1),
               uint i @gl_GlobalInvocationID) {
            buffer_store(o, i, buffer_load(src, i).b.y);
        }
    }
}
"""
    # The packed b puts c at 28, where a float3 would put it at 32.
    with pytest.raises(PackedMemberLayoutError, match="'k.src'"):
        refuse_unplaced_packed_members(_program(source), "opengl")
    # A packed vector followed by a 16-byte-aligned member moves nothing.
    source = source.replace(
        "vec4 a; @packed vec3 b; float c;", "@packed vec3 b; vec4 a;"
    )
    refuse_unplaced_packed_members(_program(source), "opengl")


# Device buffers ----------------------------------------------------------------
def test_metal_device_buffer_mixing_packed_and_aligned_vectors(tmp_path):
    code = _translate(tmp_path, METAL_MIXED, "shader.metal", "cgl")
    assert "@packed vec3 b;" in code
    assert "@buffer_layout(metal) StructuredBuffer<Q> src @buffer(0)" in code
    metal = _translate(tmp_path, METAL_MIXED, "shader.metal", "metal")
    assert "packed_float3 b;" in metal
    _metal_offsets(tmp_path, metal, "Q", MIXED_OFFSETS)
    for backend in ("opengl", "directx"):
        lowered = _translate(tmp_path, METAL_MIXED, "shader.metal", backend)
        # s, b, c and d are read from words 4, 5..7, 8 and 10..11 of each 12-word element.
        for word in (4, 5, 7, 8, 10, 11):
            assert f"* 12u) + {word}u)]" in lowered, (backend, word)
        assert "* 12u) + 9u)]" not in lowered
    compiler = shutil.which("dxc")
    if compiler is not None:
        path = tmp_path / "mixed.hlsl"
        path.write_text(lowered, encoding="utf-8")
        _run(
            [compiler, "-T", "cs_6_0", "-E", "CSMain", "-Fo", "mixed.dxil", path.name],
            cwd=tmp_path,
        )
