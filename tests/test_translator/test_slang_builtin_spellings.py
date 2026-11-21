"""Slang spellings of CrossGL built-ins and stage qualifiers that Slang names differently.

Each case used to be a wrong row in the Slang output coverage: the generator wrote a GLSL name Slang does not
declare, a select with MSL's argument order, a user varying where a built-in was meant, or dropped a stage
qualifier. Every output is compiled with slangc when it is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused


def translate(tmp_path, source, backend="slang", suffix=".cgl"):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def compile_slang(tmp_path, code, entry="main", stage=None):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source, binary = tmp_path / "out.slang", tmp_path / "out.spv"
    source.write_text(code, encoding="utf-8")
    command = [slangc, str(source), "-target", "spirv", "-o", str(binary)]
    if stage is not None:
        command += ["-entry", entry, "-stage", stage]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    if shutil.which("spirv-val"):
        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", str(binary)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


SUBGROUP = """shader S {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer uint* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            uvec4 m = gl_SubgroupLtMask | gl_SubgroupEqMask | gl_SubgroupGeMask | gl_SubgroupGtMask
                | gl_SubgroupLeMask;
            outp[i] = gl_SubgroupSize + gl_SubgroupInvocationID + gl_NumSubgroups + gl_SubgroupID + m.x;
        }
    }
}
"""


def test_subgroup_builtins_read_slang_wave_intrinsics(tmp_path):
    code = translate(tmp_path, SUBGROUP)
    for intrinsic in (
        "WaveGetLaneLtMask()",
        "WaveGetLaneEqMask()",
        "WaveGetLaneGeMask()",
        "WaveGetLaneGtMask()",
        "WaveGetLaneLeMask()",
        "WaveGetLaneCount()",
        "WaveGetLaneIndex()",
        "WaveGetNumWaves()",
        "WaveGetWaveIndex()",
    ):
        assert intrinsic in code, intrinsic
    assert "gl_Subgroup" not in code and "gl_NumSubgroups" not in code
    compile_slang(tmp_path, code)


def test_workgroup_subgroup_builtins_outside_compute_are_refused(tmp_path):
    # GLSL declares gl_SubgroupID and gl_NumSubgroups only in compute-class stages, as Slang does its intrinsics.
    source = """shader S {
        fragment {
            out vec4 c;
            void main() { c = vec4(float(gl_SubgroupID), 0.0, 0.0, 1.0); }
        }
    }"""
    with pytest.raises(Exception) as caught:
        translate(tmp_path, source)
    assert "gl_SubgroupID" in str(caught.value)
    assert "fragment" in str(caught.value)


SELECT = """shader S {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* outp;
        void main() {
            bvec2 c = bvec2(true, false);
            ivec2 r = select(ivec2(1, 2), ivec2(3, 4), c);
            outp[0] = r.x;
            outp[1] = r.y;
        }
    }
}
"""


def test_select_takes_the_condition_first_in_slang(tmp_path):
    # CrossGL select(f, t, c) is MSL's; Slang's select(c, t, f) is HLSL's.
    code = translate(tmp_path, SELECT)
    assert "select(c, int2(3, 4), int2(1, 2))" in code
    compile_slang(tmp_path, code)


def test_slang_select_imports_in_crossgl_order(tmp_path):
    source = """
RWStructuredBuffer<int2> outp;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    bool2 c = bool2(true, false);
    outp[0] = select(c, int2(3, 4), int2(1, 2));
}
"""
    crossgl = translate(tmp_path, source, backend="cgl", suffix=".slang")
    assert "select(ivec2(1, 2), ivec2(3, 4), c)" in crossgl.replace("  ", " ")
    # Back to Slang, the condition is first again.
    assert "select(c, int2(3, 4), int2(1, 2))" in translate(
        tmp_path, crossgl, backend="slang"
    )


def test_point_size_is_the_slang_system_value(tmp_path):
    source = """shader S {
        vertex {
            void main() {
                gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                gl_PointSize = 4.0;
            }
        }
    }"""
    code = translate(tmp_path, source)
    assert ": SV_PointSize" in code and "PSIZE" not in code
    compile_slang(tmp_path, code)


def test_early_fragment_tests_become_earlydepthstencil(tmp_path):
    source = """shader S {
        fragment {
            layout(early_fragment_tests) in;
            out vec4 c;
            void main() { c = vec4(1.0); }
        }
    }"""
    code = translate(tmp_path, source)
    # clang-format, when installed, joins the attribute lines; the attribute order is what matters.
    assert re.search(r'\[earlydepthstencil\]\s*\[shader\("fragment"\)\]', code)
    compile_slang(tmp_path, code)


@pytest.mark.parametrize(
    "layout, semantic",
    [
        ("depth_greater", "SV_DepthGreaterEqual"),
        ("depth_less", "SV_DepthLessEqual"),
    ],
)
def test_conservative_depth_layouts_select_the_depth_semantic(
    tmp_path, layout, semantic
):
    source = f"""shader S {{
        fragment {{
            layout({layout}) out float gl_FragDepth;
            out vec4 c;
            void main() {{ c = vec4(1.0); gl_FragDepth = 0.5; }}
        }}
    }}"""
    code = translate(tmp_path, source)
    assert f"float gl_FragDepth : {semantic};" in code
    compile_slang(tmp_path, code)


def test_rasterizer_ordered_buffers_declare_the_ordered_view(tmp_path):
    source = """shader S {
        fragment {
            @rasterizer_ordered buffer uint* counts;
            buffer uint* plain;
            out vec4 c;
            void main() { counts[0] = counts[0] + 1u; plain[0] = 1u; c = vec4(1.0); }
        }
    }"""
    code = translate(tmp_path, source)
    assert "RasterizerOrderedStructuredBuffer<uint> counts" in code
    assert "RWStructuredBuffer<uint> plain" in code
    compile_slang(tmp_path, code)


CBUFFER = """shader S {{
    cbuffer Params @buffer_layout(hlsl_cbuffer) {{ {members} }};
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {{ outp[0] = {value}; }}
    }}
}}
"""


def test_hlsl_packed_cbuffer_members_keep_their_hlsl_offsets(tmp_path):
    # HLSL packs b into c0.y and c into c0.w, where std140 would place b at byte 8 and c at byte 16.
    source = CBUFFER.format(
        members="float a; vec2 b; float c; vec3 d;", value="a + b.y + c + d.z"
    )
    code = translate(tmp_path, source)
    for placement in (
        "float a : packoffset(c0);",
        "float2 b : packoffset(c0.y);",
        "float c : packoffset(c0.w);",
        "float3 d : packoffset(c1);",
    ):
        assert placement in code, placement
    compile_slang(tmp_path, code)


def test_a_cbuffer_whose_offsets_agree_needs_no_packoffset(tmp_path):
    code = translate(
        tmp_path, CBUFFER.format(members="vec4 a; float b;", value="a.x + b")
    )
    assert "packoffset" not in code


def test_an_hlsl_member_inside_std140_padding_is_refused(tmp_path):
    # HLSL puts b at byte 20, inside the std140 extent of a[2] (32 bytes), which packoffset cannot express.
    source = CBUFFER.format(members="float a[2]; float b;", value="a[1] + b")
    with pytest.raises(SlangLoweringRefused) as caught:
        translate(tmp_path, source)
    assert "member 'b' sits at byte 20 under HLSL packing" in str(caught.value)
