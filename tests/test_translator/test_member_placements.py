"""Constant-buffer member placements (``@offset``, ``@align``) on every target.

A target either reproduces a member's explicit placement or refuses it with a typed diagnostic. Metal declared a
constant block it does not plan in its natural layout, and Slang declared push-constant structs and constant blocks
without the offsets, so both read members from other bytes than the host wrote. A vector may also sit at a relaxed
offset (Vulkan 1.3 section 15.6.4): SPIR-V and HLSL output keep it, GLSL output, whose ``layout(offset)`` cannot write
it, refuses it.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_block_layout import (
    improperly_straddles,
    relaxed_vector_offset,
)
from crosstl.translator.codegen.hlsl_buffer_layout import (
    array_type,
    scalar_type,
    vector_type,
)
from crosstl.translator.codegen.push_constant_blocks import PushConstantBlockError
from crosstl.translator.semantics.member_placements import MemberPlacementError

BODY = """
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = %s;
        }
    }
}
"""

TARGETS = ("vulkan", "directx", "opengl", "metal", "slang")


def _cbuffer(qualifiers, members, expression):
    return (
        f"shader Probe {{\n    cbuffer C{qualifiers} {{ {members} }};\n"
        + BODY % expression
    )


def _struct_buffer(members, declaration, expression):
    return (
        f"shader Probe {{\n    struct T {{ {members} }};\n    {declaration}\n"
        + BODY % expression
    )


def _translate(tmp_path, source, backend):
    path = tmp_path / f"{backend}.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _outcomes(tmp_path, source):
    outcomes = {}
    for backend in TARGETS:
        try:
            _translate(tmp_path, source, backend)
            outcomes[backend] = "ok"
        except MemberPlacementError:
            outcomes[backend] = "placement"
        except PushConstantBlockError:
            outcomes[backend] = "push-constant"
    return outcomes


@pytest.mark.parametrize(
    "source, expected",
    [
        # Metal and Slang declare a block without a declared layout with no offsets.
        (
            _cbuffer("", "float s; vec3 v @offset(32);", "s + v.y"),
            {"metal": "placement", "slang": "placement"},
        ),
        # HLSL packing puts v at byte 4, so Slang's Direct3D layout would not agree with the offset.
        (
            _cbuffer("", "float s; vec3 v @offset(16);", "s + v.y"),
            {"slang": "placement"},
        ),
        # Metal plans a block with a declared layout and places its offsets.
        (
            _cbuffer(
                " @buffer_layout(std140)", "float s; vec3 v @offset(32);", "s + v.y"
            ),
            {"slang": "placement"},
        ),
        # slangc moves a push-constant [[vk::offset]] to the member's std430 alignment; GLSL cannot write it.
        # Metal plans a push-constant block with its push-constant layout.
        (
            _cbuffer(" @push_constant", "float s; vec3 v @offset(4);", "s + v.y"),
            {"opengl": "push-constant", "slang": "placement"},
        ),
        (_cbuffer(" @push_constant", "float s; vec3 v @offset(16);", "s + v.y"), {}),
        (
            _struct_buffer(
                "float s; vec3 v @offset(4);",
                "ConstantBuffer<T> pc @push_constant;",
                "pc.s + pc.v.y",
            ),
            {"opengl": "push-constant", "slang": "placement"},
        ),
        # SPIR-V places the members of T itself; the other targets declare T without its offsets.
        (
            _struct_buffer(
                "float s; vec3 v @offset(32);",
                "ConstantBuffer<T> cb @binding(0);",
                "cb.s + cb.v.y",
            ),
            {
                "directx": "placement",
                "opengl": "placement",
                "metal": "placement",
                "slang": "placement",
            },
        ),
        # A declared layout declares a block holding one T, placed by the layout's rules; HLSL reads it from
        # registers at the planned offsets, Metal pads T to its members' placements, and SPIR-V decorates T's
        # members with the placed offsets.
        (
            _struct_buffer(
                "float s; vec3 v @offset(32);",
                "ConstantBuffer<T> cb @binding(0) @buffer_layout(std140);",
                "cb.s + cb.v.y",
            ),
            {
                "opengl": "placement",
                "slang": "placement",
            },
        ),
        # Offsets every target's own rules agree with are accepted.
        (
            _struct_buffer(
                "vec4 a @offset(0); float s @offset(16);",
                "ConstantBuffer<T> cb @binding(0);",
                "cb.a.y + cb.s",
            ),
            {},
        ),
    ],
)
def test_targets_reproduce_or_refuse_member_placements(tmp_path, source, expected):
    outcomes = _outcomes(tmp_path, source)
    assert outcomes == {backend: expected.get(backend, "ok") for backend in TARGETS}


def test_placement_refusal_names_the_buffer_and_struct(tmp_path):
    source = _struct_buffer(
        "float s; vec3 v @offset(32);",
        "ConstantBuffer<T> cb @binding(0);",
        "cb.s + cb.v.y",
    )
    with pytest.raises(MemberPlacementError) as caught:
        _translate(tmp_path, source, "metal")
    assert caught.value.code == "project.translate.member-placement-unsupported"
    assert "constant buffer 'cb'" in str(caught.value)
    assert "struct 'T'" in str(caught.value)


@pytest.mark.parametrize("backend", ["directx", "metal", "slang"])
def test_placement_qualifiers_are_not_written_as_semantics(tmp_path, backend):
    source = _cbuffer(
        " @buffer_layout(std140)", "float s; vec4 v @align(32);", "s + v.y"
    )
    if backend == "slang":
        source = _cbuffer(" @push_constant", "float s; vec4 v @offset(16);", "s + v.y")
    code = _translate(tmp_path, source, backend)
    assert not re.search(r"(?<!:):\s*(offset|align)\b", code)
    assert not re.search(r"\[\[\s*(offset|align)\b", code)


def test_glsl_push_constant_cbuffer_refuses_an_offset_layout_offset_cannot_write(
    tmp_path,
):
    with pytest.raises(
        PushConstantBlockError,
        match=re.escape(
            "member 'v' is at byte 4, which is not a multiple of its std430 alignment 16"
        ),
    ):
        _translate(
            tmp_path,
            _cbuffer(" @push_constant", "float s; vec3 v @offset(4);", "s + v.y"),
            "opengl",
        )


def test_hlsl_push_constant_cbuffer_keeps_a_relaxed_vector_offset(tmp_path):
    hlsl = _translate(
        tmp_path,
        _cbuffer(
            " @push_constant", "float s; vec3 v @offset(4); vec2 w;", "s + v.y + w.x"
        ),
        "directx",
    )
    spirv_branch = hlsl.split("#ifdef __spirv__\n", 1)[1].split("#else\n", 1)[0]
    found = re.findall(r"\[\[vk::offset\((\d+)\)\]\] (\w+) (\w+);", spirv_branch)
    assert found == [("0", "float", "s"), ("4", "float3", "v"), ("16", "float2", "w")]
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    (tmp_path / "out.hlsl").write_text(hlsl)
    result = subprocess.run(
        [dxc, "-spirv", "-fspv-target-env=vulkan1.1", "-T", "cs_6_0", "-E", "CSMain"]
        + ["-HV", "2021", "out.hlsl", "-Fo", "out.spv"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "offset, size, straddles",
    [
        (4, 12, False),
        (8, 12, True),
        (12, 4, False),
        (12, 8, True),
        (16, 32, False),
        (8, 32, True),
    ],
)
def test_improper_straddle(offset, size, straddles):
    assert improperly_straddles(offset, size) is straddles


@pytest.mark.parametrize(
    "layout_type, offset, layout, relaxed",
    [
        (vector_type("float", 3), 4, "std430", True),
        (vector_type("float", 3), 4, "std140", True),
        (vector_type("float", 3), 8, "std430", False),
        (vector_type("float", 2), 6, "std430", False),
        (vector_type("float", 3), 4, "metal", False),
        (scalar_type("float"), 2, "std430", False),
        (array_type(vector_type("float", 3), 2), 4, "std430", False),
    ],
)
def test_relaxed_vector_offset(layout_type, offset, layout, relaxed):
    assert relaxed_vector_offset(layout_type, offset, layout) is relaxed


NESTED_PLACEMENT = """
shader Probe {
    struct Inner { float s; vec4 v @offset(OFFSET); };
    cbuffer C { Inner n; };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = n.s + n.v.y;
        }
    }
}
"""


@pytest.mark.parametrize("backend", ["metal", "slang"])
def test_a_placement_inside_a_struct_held_by_a_constant_buffer_is_refused(
    tmp_path, backend
):
    # Issue #4245: SPIR-V wrote 'Offset 16', HLSL a semantic named 'offset' and GLSL a plain 'vec4 v;'. SPIR-V now
    # places the member at byte 32 (test_spirv_nested_member_placements.py runs it on Vulkan), and GLSL and HLSL
    # read it from registers (test_nested_member_placement_registers.py).
    path = tmp_path / "nested.cgl"
    path.write_text(NESTED_PLACEMENT.replace("OFFSET", "32"))

    with pytest.raises(MemberPlacementError, match="constant buffer 'C'"):
        crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", ["vulkan", "directx", "opengl"])
def test_a_nested_placement_that_matches_the_layout_is_accepted(tmp_path, backend):
    # Importers place every member. std140 and HLSL packing both start the vec4 at byte 16.
    path = tmp_path / "nested.cgl"
    path.write_text(
        NESTED_PLACEMENT.replace("OFFSET", "16").replace(
            "float s;", "float s @offset(0);"
        )
    )

    assert crosstl.translate(str(path), backend=backend)


def _metal_offsets_hold(tmp_path, metal, offsets):
    """The Metal compiler accepts ``metal`` with ``__builtin_offsetof(struct, member) == offset`` asserted for each item of
    ``offsets``; ``None`` without the compiler."""
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    asserts = "".join(
        f'static_assert(__builtin_offsetof({struct}, {member}) == {offset}, "{struct}.{member}");\n'
        for struct, member, offset in offsets
    )
    source = tmp_path / "placed.metal"
    source.write_text(metal + "\n" + asserts)
    result = subprocess.run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "placed.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return None
    assert result.returncode == 0, result.stderr
    return True


@pytest.mark.parametrize(
    "source, offsets",
    [
        # std430 puts s at byte 12, right after v; the natural Metal layout put it at 16.
        (
            _cbuffer(" @push_constant", "vec3 v; float s;", "s + v.y"),
            [("C", "v", 0), ("C", "s", 12)],
        ),
        (
            _cbuffer(" @push_constant", "float s; vec3 v @offset(4);", "s + v.y"),
            [("C", "s", 0), ("C", "v", 4)],
        ),
        (
            _struct_buffer(
                "vec3 v; float s;",
                "ConstantBuffer<T> pc @push_constant;",
                "pc.s + pc.v.y",
            ),
            [("T", "v", 0), ("T", "s", 12)],
        ),
        (
            _struct_buffer(
                "float s; vec3 v @offset(4);",
                "ConstantBuffer<T> pc @push_constant;",
                "pc.s + pc.v.y",
            ),
            [("T", "s", 0), ("T", "v", 4)],
        ),
        (
            _struct_buffer(
                "float s; vec3 v @offset(32);",
                "ConstantBuffer<T> cb @binding(0) @buffer_layout(std140);",
                "cb.s + cb.v.y",
            ),
            [("T", "s", 0), ("T", "v", 32)],
        ),
        (
            _struct_buffer(
                "float a @offset(272);",
                "ConstantBuffer<T> cb @binding(0) @buffer_layout(hlsl_cbuffer);",
                "cb.a",
            ),
            [("T", "a", 272)],
        ),
    ],
)
def test_metal_places_push_constant_blocks_and_the_structs_planned_blocks_hold(
    tmp_path, source, offsets
):
    """Metal output lays a push-constant block out by its push-constant layout (std430 here), and pads a struct
    that a planned block holds to its members' ``@offset``, where it had declared both in the natural MSL layout.
    """
    metal = _translate(tmp_path, source, "metal")
    _metal_offsets_hold(tmp_path, metal, offsets)
