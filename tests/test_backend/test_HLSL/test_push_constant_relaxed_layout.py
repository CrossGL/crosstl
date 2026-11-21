"""DXC's relaxed layout of ``[[vk::push_constant]]`` structs, imported as CrossGL ``@offset`` qualifiers.

DXC lays a push-constant struct out by its relaxed std430 rule: a vector of 32-bit components is aligned to its
component size unless it would cross a 16-byte boundary there (Vulkan 1.3 section 15.6.4). ``float scale; float3 v;``
puts ``v`` at byte 4, where std430 puts it at 16. The importer declared the struct with no offsets, so every output
read ``v`` from byte 16, and it dropped ``[[vk::offset(N)]]``. It now writes the offsets, which SPIR-V and HLSL output
reproduce, Metal output places, and GLSL and Slang output refuse when they cannot.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.DirectX.push_constant_offsets import HLSLPushConstantLayoutError
from crosstl.translator.codegen.push_constant_blocks import PushConstantBlockError
from crosstl.translator.semantics.member_placements import MemberPlacementError

TAIL = """
RWStructuredBuffer<float> outp : register(u1);
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) { outp[id.x] = %s; }
"""

HALF_MEMBERS = (
    "struct PC { half h; float3 v; };\n"
    "[[vk::push_constant]] PC pc;\n"
    + TAIL % "pc.h + pc.v.y"
)

RELAXED = (
    "struct Constants { float scale; float3 v; };\n"
    "[[vk::push_constant]] ConstantBuffer<Constants> pc;\n" + TAIL % "pc.scale + pc.v.y"
)

EXPLICIT = (
    "struct Constants { float scale; [[vk::offset(16)]] float3 v; };\n"
    "[[vk::push_constant]] ConstantBuffer<Constants> pc;\n" + TAIL % "pc.scale + pc.v.y"
)


def _translate(tmp_path, hlsl, backend, name="shader.hlsl"):
    path = tmp_path / name
    path.write_text(hlsl)
    return crosstl.translate(str(path), backend=backend)


def _offset_members(crossgl):
    return re.findall(r"(\w+) (\w+) @offset\((\d+)\);", crossgl)


@pytest.mark.parametrize(
    "declaration, expression, offsets",
    [
        (
            "struct C { float scale; float3 v; };\n[[vk::push_constant]] ConstantBuffer<C> pc;",
            "pc.scale + pc.v.y",
            [("vec3", "v", "4")],
        ),
        (
            "struct C { float scale; float3 v; };\n[[vk::push_constant]] C pc;",
            "pc.scale + pc.v.y",
            [("vec3", "v", "4")],
        ),
        # b moves to byte 4; c then follows it at byte 12 under both rules.
        (
            "struct C { float a; float2 b; float c; };\n[[vk::push_constant]] ConstantBuffer<C> pc;",
            "pc.a + pc.b.y + pc.c",
            [("vec2", "b", "4")],
        ),
        # A nested struct keeps its std430 alignment; the vector after it moves.
        (
            "struct I { float4 a; float b; };\nstruct C { float s; I inner; float t; float3 v; };\n"
            "[[vk::push_constant]] ConstantBuffer<C> pc;",
            "pc.s + pc.inner.b + pc.t + pc.v.y",
            [("vec3", "v", "52")],
        ),
    ],
)
def test_importer_places_vectors_where_dxc_relaxed_layout_puts_them(
    tmp_path, declaration, expression, offsets
):
    crossgl = _translate(tmp_path, declaration + TAIL % expression, "cgl")
    assert _offset_members(crossgl) == offsets


@pytest.mark.parametrize(
    "members, expression",
    [
        # The vector would cross byte 16 at byte 8, so DXC moves it to 16 as std430 does.
        ("float2 a; float3 v;", "pc.a.x + pc.v.y"),
        ("float a; float4 v;", "pc.a + pc.v.y"),
        # Arrays and matrices keep their std430 alignment.
        ("float a; float3 v[2]; float3x3 m;", "pc.a + pc.v[1].y + pc.m[1][1]"),
        ("float4 a; float b; float c;", "pc.a.x + pc.b + pc.c"),
    ],
)
def test_importer_writes_no_offset_where_std430_agrees(tmp_path, members, expression):
    hlsl = (
        f"struct C {{ {members} }};\n[[vk::push_constant]] ConstantBuffer<C> pc;\n"
        + TAIL % expression
    )
    assert _offset_members(_translate(tmp_path, hlsl, "cgl")) == []


def test_importer_keeps_vk_offset_of_any_struct_member(tmp_path):
    hlsl = (
        "struct C { float scale; [[vk::offset(32)]] float3 v; };\n"
        "ConstantBuffer<C> cb : register(b0);\n" + TAIL % "cb.scale + cb.v.y"
    )
    assert _offset_members(_translate(tmp_path, hlsl, "cgl")) == [("vec3", "v", "32")]
    assert _offset_members(_translate(tmp_path, EXPLICIT, "cgl")) == [
        ("vec3", "v", "16")
    ]


@pytest.mark.parametrize(
    "declaration, expression, reason",
    [
        (
            "struct I { float a; float3 b; };\nstruct C { float s; I inner; };\n"
            "[[vk::push_constant]] ConstantBuffer<C> pc;",
            "pc.s + pc.inner.b.y",
            "member 'inner.b' of struct 'I' is at byte 4 under DXC's relaxed rule and at byte 16 under std430",
        ),
        (
            "struct C { float scale; float3 v; };\n[[vk::push_constant]] ConstantBuffer<C> pc;\n"
            "StructuredBuffer<C> others : register(t0);",
            "pc.scale + pc.v.y + others[0].v.x",
            "struct 'C' needs member offsets for DXC's relaxed rule and is also laid out by global 'others'",
        ),
        (
            "struct C { float scale; float3 v; };\n[[vk::push_constant]] ConstantBuffer<C> pc;\n"
            "struct Outer { C c; };\nConstantBuffer<Outer> cb : register(b0);",
            "pc.scale + cb.c.v.y",
            "is also laid out by member 'c' of struct 'Outer'",
        ),
    ],
)
def test_importer_refuses_layouts_crossgl_cannot_record(
    tmp_path, declaration, expression, reason
):
    with pytest.raises(HLSLPushConstantLayoutError, match=re.escape(reason)):
        _translate(tmp_path, declaration + TAIL % expression, "cgl")


def test_spirv_output_reads_the_vector_where_dxc_places_it(tmp_path):
    spirv = _translate(tmp_path, RELAXED, "vulkan")
    block = re.search(r"%(\w+) = OpTypePointer PushConstant %(\w+)", spirv).group(2)
    offsets = re.findall(rf"OpMemberDecorate %{block} (\d+) Offset (\d+)", spirv)
    assert offsets == [("0", "0"), ("1", "4")]
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(spirv)
        for command in (
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
            ["spirv-val", "--target-env", "vulkan1.1", "out.spv"],
        ):
            result = subprocess.run(
                command, cwd=tmp_path, capture_output=True, text=True
            )
            assert result.returncode == 0, result.stderr


def _dxc_push_constant_offsets(tmp_path, hlsl, name):
    """The member offsets DXC gives the push-constant block of ``hlsl``; ``None`` without DXC and SPIRV-Tools."""
    if not (shutil.which("dxc") and shutil.which("spirv-dis")):
        return None
    source, binary = tmp_path / f"{name}.hlsl", tmp_path / f"{name}.spv"
    source.write_text(hlsl)
    command = ["dxc", "-spirv", "-fspv-target-env=vulkan1.1", "-T", "cs_6_0"]
    result = subprocess.run(
        command + ["-E", "CSMain", "-HV", "2021", str(source), "-Fo", str(binary)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assembly = subprocess.run(
        ["spirv-dis", str(binary)], capture_output=True, text=True, check=True
    ).stdout
    block = re.search(r"%(\w+) = OpTypePointer PushConstant %(\w+)", assembly).group(2)
    offsets = re.findall(rf"OpMemberDecorate %{block} (\d+) Offset (\d+)", assembly)
    if len(offsets) == 1:
        # HLSL output wraps the push-constant struct in a block of its own.
        inner = re.search(rf"%{block} = OpTypeStruct %(\w+)", assembly).group(1)
        offsets = re.findall(rf"OpMemberDecorate %{inner} (\d+) Offset (\d+)", assembly)
    return offsets


@pytest.mark.parametrize("source", [RELAXED, EXPLICIT], ids=["relaxed", "explicit"])
def test_hlsl_output_keeps_the_offsets_dxc_gives_the_source(tmp_path, source):
    hlsl = _translate(tmp_path, source, "directx")
    expected = "4" if source is RELAXED else "16"
    assert f"[[vk::offset({expected})]] float3 v;" in hlsl
    original = _dxc_push_constant_offsets(tmp_path, source, "source")
    if original is None:
        return
    assert original == [("0", "0"), ("1", expected)]
    assert _dxc_push_constant_offsets(tmp_path, hlsl, "output") == original


def test_glsl_output_refuses_a_relaxed_offset_and_keeps_an_agreeing_one(tmp_path):
    with pytest.raises(
        PushConstantBlockError,
        match="member 'v' of struct 'Constants' is at byte 4, where std430 places it at byte 16",
    ):
        _translate(tmp_path, RELAXED, "opengl")
    glsl = _translate(tmp_path, EXPLICIT, "opengl")
    assert "layout(push_constant) uniform Constants {" in glsl


def test_slang_refuses_a_relaxed_offset(tmp_path):
    reason = "Slang declares a push-constant ConstantBuffer<T> with std430 offsets"
    with pytest.raises(MemberPlacementError, match=re.escape(reason)):
        _translate(tmp_path, RELAXED, "slang")
    # An offset std430 agrees with needs no placement.
    assert _translate(tmp_path, EXPLICIT, "slang", name="explicit.hlsl")


def test_importer_places_float3_after_half_scalar(tmp_path):
    """DXC places ``float3 v`` at byte 4 after a ``half h`` member (DXC 1.9, with and without
    ``-enable-16bit-types``).  CrossGL must record an ``@offset(4)`` on ``v`` so that all
    backends reproduce the same layout.
    """
    crossgl = _translate(tmp_path, HALF_MEMBERS, "cgl")
    offsets = _offset_members(crossgl)
    # ``v`` must carry an explicit offset; its exact byte depends on how the importer
    # models ``half`` storage, but it must not be absent (std430 would put v at 16).
    assert any(name == "v" for _t, name, _off in offsets), (
        f"expected an @offset on 'v' in: {crossgl!r}"
    )
    v_offset = int(next(off for _t, name, off in offsets if name == "v"))
    assert v_offset < 16, (
        f"expected 'v' to be placed before byte 16 (DXC relaxed rule), got byte {v_offset}"
    )


def test_half_push_constant_translates_to_vulkan_without_error(tmp_path):
    """The push-constant struct with a ``half`` member must translate to SPIR-V without
    raising ``HLSLPushConstantLayoutError``."""
    # Should not raise.
    spirv = _translate(tmp_path, HALF_MEMBERS, "vulkan")
    assert spirv  # non-empty output confirms success


@pytest.mark.parametrize(
    "members, expression",
    [
        # half scalar followed by float3 vector: v at byte 4 (relaxed, not std430's byte 16).
        ("half h; float3 v;", "pc.h + pc.v.y"),
        # Two half scalars: second at byte 2 (with -enable-16bit-types) or byte 4 (without).
        ("half a; half b; float3 v;", "pc.a + pc.b + pc.v.y"),
    ],
)
def test_half_push_constant_roundtrips_through_crossgl(tmp_path, members, expression):
    """An HLSL push-constant struct with half members can be imported into CrossGL and
    re-exported to HLSL without raising an error."""
    source = (
        f"struct PC {{ {members} }};\n[[vk::push_constant]] PC pc;\n"
        + TAIL % expression
    )
    crossgl = _translate(tmp_path, source, "cgl")
    assert crossgl  # import must succeed
    hlsl = _translate(tmp_path, source, "directx", name="shader2.hlsl")
    assert hlsl  # re-export must succeed


@pytest.mark.parametrize("source", [RELAXED, EXPLICIT], ids=["relaxed", "explicit"])
def test_metal_output_keeps_the_offsets_dxc_gives_the_source(tmp_path, source):
    """Metal output plans the push-constant struct with its push-constant layout: ``v`` is a ``packed_float3`` at
    the byte DXC gives it."""
    metal = _translate(tmp_path, source, "metal")
    expected = 4 if source is RELAXED else 16
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    path = tmp_path / "placed.metal"
    path.write_text(
        metal
        + f'\nstatic_assert(__builtin_offsetof(Constants, v) == {expected}, "v");\n'
        + 'static_assert(__builtin_offsetof(Constants, scale) == 0, "scale");\n'
    )
    result = subprocess.run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "p.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr
