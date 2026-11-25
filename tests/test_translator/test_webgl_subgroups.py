import re

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.glsl_refusals import WebGLLoweringRefused
from crosstl.translator.codegen.stage_refusals import StageUnsupportedError
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen
from crosstl.translator.semantics.builtin_values import BuiltinValueUnsupportedError


def _fragment(expression, helpers="", *, statements=""):
    return f"""
    shader Probe {{
        {helpers}
        fragment {{
            vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {{
                {statements}
                return vec4({expression}, 0.0, 0.0, 1.0);
            }}
        }}
    }}
    """


def _assert_refuses_without_output(tmp_path, source, *names):
    output = tmp_path / "out.frag"
    with pytest.raises(WebGLLoweringRefused) as caught:
        crosstl.translate(
            str(source), backend="webgl", save_shader=str(output), format_output=False
        )
    for name in names:
        assert f"'{name}'" in str(caught.value)
    assert not output.exists()


def _emitted_float_helper_name(code, parameters, expression):
    declaration = re.search(
        rf"\bfloat (\w+)\({re.escape(parameters)}\) \{{\s*"
        rf"return {re.escape(expression)};\s*\}}",
        code,
    )
    assert declaration is not None, code
    return declaration[1]


@pytest.mark.parametrize(
    "barrier, desktop_spelling",
    [
        ("subgroupBarrier", "subgroupBarrier"),
        ("subgroupExecutionBarrier", "subgroupBarrier"),
        ("subgroupMemoryBarrier", "subgroupMemoryBarrier"),
        ("subgroupMemoryBarrierBuffer", "subgroupMemoryBarrierBuffer"),
        ("subgroupMemoryBarrierImage", "subgroupMemoryBarrierImage"),
    ],
)
def test_webgl_refuses_subgroup_barriers(tmp_path, barrier, desktop_spelling):
    source = tmp_path / "barrier.cgl"
    source.write_text(_fragment("pos.x", statements=f"{barrier}();"), encoding="utf-8")

    desktop = crosstl.translate(str(source), backend="opengl", format_output=False)
    assert f"{desktop_spelling}();" in desktop
    assert "#extension GL_KHR_shader_subgroup_basic : require" in desktop
    _assert_refuses_without_output(tmp_path, source, barrier)


def test_fragment_subgroup_memory_barrier_shared_fails_closed(tmp_path):
    # GL_KHR_shader_subgroup declares subgroupMemoryBarrierShared only for compute
    # shaders (GL_EXT_mesh_shader adds task and mesh), so desktop GLSL refuses a
    # fragment-stage call, and WebGL 2 refuses every subgroup barrier.
    source = tmp_path / "barrier.cgl"
    source.write_text(
        _fragment("pos.x", statements="subgroupMemoryBarrierShared();"),
        encoding="utf-8",
    )

    with pytest.raises(
        BuiltinValueUnsupportedError, match="subgroupMemoryBarrierShared"
    ):
        crosstl.translate(str(source), backend="opengl", format_output=False)
    _assert_refuses_without_output(tmp_path, source, "subgroupMemoryBarrierShared")


@pytest.mark.parametrize(
    "name",
    ["subgroupBarrier", "subgroupExecutionBarrier", "subgroupMemoryBarrierImage"],
)
def test_webgl_keeps_user_function_with_subgroup_barrier_name(name):
    source = _fragment(f"{name}()", f"float {name}() {{ return 0.5; }}")

    code = WebGLCodeGen().generate(parse(source))

    helper_name = _emitted_float_helper_name(code, "", "0.5")
    assert f"vec4({helper_name}(), 0.0, 0.0, 1.0)" in code
    assert "GL_KHR_shader_subgroup" not in code


@pytest.mark.parametrize(
    "name, component",
    [
        ("gl_SubgroupSize", ""),
        ("gl_SubgroupInvocationID", ""),
        ("gl_SubgroupEqMask", ".x"),
        ("gl_SubgroupGeMask", ".x"),
        ("gl_SubgroupGtMask", ".x"),
        ("gl_SubgroupLeMask", ".x"),
        ("gl_SubgroupLtMask", ".x"),
    ],
)
def test_webgl_refuses_subgroup_builtin_reads(tmp_path, name, component):
    source = tmp_path / "builtin.cgl"
    source.write_text(_fragment(f"float({name}{component})"), encoding="utf-8")

    _assert_refuses_without_output(tmp_path, source, name)


@pytest.mark.parametrize("name", ["gl_NumSubgroups", "gl_SubgroupID"])
def test_webgl_refuses_compute_subgroup_builtins_in_fragment(tmp_path, name):
    source_code = _fragment(f"float({name})")
    source = tmp_path / "compute_builtin.cgl"
    source.write_text(source_code, encoding="utf-8")
    output = tmp_path / "out.frag"

    with pytest.raises(BuiltinValueUnsupportedError, match=name):
        crosstl.translate(
            str(source), backend="webgl", save_shader=str(output), format_output=False
        )
    assert not output.exists()

    with pytest.raises(WebGLLoweringRefused, match=f"'{name}'"):
        WebGLCodeGen().generate(parse(source_code))


@pytest.mark.parametrize(
    "members, expression, names",
    [
        (
            "uint size @gl_SubgroupSize; uint lane @gl_SubgroupInvocationID;",
            "fi.size + fi.lane",
            ("gl_SubgroupSize", "gl_SubgroupInvocationID"),
        ),
        (
            "uvec4 mask @gl_SubgroupEqMask;",
            "fi.mask.x",
            ("gl_SubgroupEqMask",),
        ),
    ],
)
def test_webgl_refuses_subgroup_builtin_input_members(
    tmp_path, members, expression, names
):
    source = tmp_path / "inputs.cgl"
    source.write_text(
        f"""
        shader Inputs {{
            struct FI {{ vec4 colour; {members} }};
            fragment {{
                vec4 main(FI fi) @gl_FragColor {{
                    return fi.colour + vec4(float({expression}));
                }}
            }}
        }}
        """,
        encoding="utf-8",
    )

    desktop = crosstl.translate(str(source), backend="opengl", format_output=False)
    assert all(name in desktop for name in names)
    _assert_refuses_without_output(tmp_path, source, *names)


def test_webgl_reports_canonical_operation_for_combined_metal_aliases(tmp_path):
    source = tmp_path / "aliases.metal"
    source.write_text(
        """#include <metal_stdlib>
using namespace metal;
fragment float4 colour(float4 pos [[position]]) {
    return float4(simd_shuffle_xor(pos.y, ushort(1)) + simd_sum(pos.x));
}
""",
        encoding="utf-8",
    )
    desktop = crosstl.translate(str(source), backend="opengl", format_output=False)
    assert "subgroupShuffleXor(" in desktop
    assert "subgroupAdd(" in desktop

    _assert_refuses_without_output(tmp_path, source, "WaveActiveSum")


@pytest.mark.parametrize(
    "intrinsic, expression, glsl_operation",
    [
        ("WaveGetLaneCount", "float(WaveGetLaneCount())", "gl_SubgroupSize"),
        ("WaveIsFirstLane", "WaveIsFirstLane() ? 1.0 : 0.0", "subgroupElect("),
        ("WaveActiveSum", "WaveActiveSum(pos.x)", "subgroupAdd("),
        # A stage whose only subgroup operations are boolean votes uses
        # GL_ARB_shader_group_vote in desktop GLSL.
        (
            "WaveActiveAllTrue",
            "WaveActiveAllTrue(pos.x > 0.0) ? 1.0 : 0.0",
            "allInvocationsARB(",
        ),
        (
            "WaveActiveBallot",
            "float(WaveActiveBallot(pos.x > 0.0).x)",
            "subgroupBallot(",
        ),
        ("WavePrefixSum", "WavePrefixSum(pos.x)", "subgroupExclusiveAdd("),
        ("WaveClusteredSum", "WaveClusteredSum(pos.x, 4u)", "subgroupClusteredAdd("),
        ("WaveReadLaneFirst", "WaveReadLaneFirst(pos.x)", "subgroupBroadcastFirst("),
        ("WaveShuffleXor", "WaveShuffleXor(pos.x, 1u)", "subgroupShuffleXor("),
        ("QuadReadAcrossX", "QuadReadAcrossX(pos.x)", "subgroupQuadSwapHorizontal("),
        ("QuadAll", "QuadAll(pos.x > 0.0) ? 1.0 : 0.0", "subgroupQuadAll("),
        ("QuadAny", "QuadAny(pos.x > 0.0) ? 1.0 : 0.0", "subgroupQuadAny("),
        ("WaveActiveSum", "subgroupAdd(pos.x)", "subgroupAdd("),
        ("WaveActiveSum", "simd_sum(pos.x)", "subgroupAdd("),
    ],
)
def test_webgl_refuses_subgroup_intrinsics(
    tmp_path, intrinsic, expression, glsl_operation
):
    source = tmp_path / "subgroup.cgl"
    source.write_text(_fragment(expression), encoding="utf-8")

    desktop = crosstl.translate(str(source), backend="opengl", format_output=False)
    assert desktop.startswith("#version 450 core\n")
    assert glsl_operation in desktop

    output = tmp_path / "out.frag"
    with pytest.raises(
        WebGLLoweringRefused, match=rf"subgroup intrinsic '{intrinsic}'"
    ):
        crosstl.translate(
            str(source), backend="webgl", save_shader=str(output), format_output=False
        )
    assert not output.exists()


def test_webgl_refuses_subgroup_intrinsics_imported_from_glsl(tmp_path):
    source = tmp_path / "subgroup.frag"
    source.write_text(
        """#version 450 core
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(location = 0) out vec4 color;
void main() {
    color = vec4(subgroupAdd(gl_FragCoord.x));
}
""",
        encoding="utf-8",
    )

    with pytest.raises(
        WebGLLoweringRefused, match="subgroup intrinsic 'WaveActiveSum'"
    ):
        crosstl.translate(str(source), backend="webgl", format_output=False)


@pytest.mark.parametrize("name", ["subgroupAdd", "WaveActiveSum"])
def test_glsl_user_functions_keep_calls_with_wave_names(tmp_path, name):
    source = tmp_path / "user_functions.frag"
    source.write_text(
        f"""#version 450 core
layout(location = 0) out vec4 color;
struct {name}__ {{ float amount; }};
float {name}(float x) {{ return x * {name}__(2.0).amount; }}
float {name}(float x, float y) {{ return x + y; }}
float {name}_(float x) {{ return x - 1.0; }}
void main() {{
    color = vec4({name}(gl_FragCoord.x), {name}(gl_FragCoord.y, 1.0),
                 {name}_(gl_FragCoord.z), 1.0);
}}
""",
        encoding="utf-8",
    )

    crossgl = crosstl.translate(str(source), backend="cgl", format_output=False)
    assert f"float {name}___(float x)" in crossgl
    assert f"{name}___(gl_FragCoord.x)" in crossgl
    assert f"{name}___(gl_FragCoord.y, 1.0)" in crossgl
    assert f"{name}_(gl_FragCoord.z)" in crossgl

    for backend, version in [("webgl", "300 es"), ("opengl", "450 core")]:
        code = crosstl.translate(str(source), backend=backend, format_output=False)
        assert code.startswith(f"#version {version}\n")
        assert "GL_KHR_shader_subgroup" not in code
        struct = re.search(r"\bstruct (\w+) \{\s*float amount;\s*\};", code)
        assert struct is not None, code
        struct_name = struct[1]
        unary_name = _emitted_float_helper_name(
            code, "float x", f"(x * {struct_name}(2.0).amount)"
        )
        binary_name = _emitted_float_helper_name(code, "float x, float y", "(x + y)")
        other_name = _emitted_float_helper_name(code, "float x", "(x - 1.0)")
        assert struct_name not in {unary_name, binary_name, other_name}
        assert other_name not in {unary_name, binary_name}
        assert (
            f"vec4({unary_name}(gl_FragCoord.x), {binary_name}(gl_FragCoord.y, 1.0), "
            f"{other_name}(gl_FragCoord.z), 1.0)" in code
        )


def test_webgl_keeps_user_function_with_simd_alias_name():
    source = _fragment("simd_sum(pos.x)", "float simd_sum(float x) { return x + 1.0; }")

    code = WebGLCodeGen().generate(parse(source))

    assert code.startswith("#version 300 es\n")
    assert "simd_sum(gl_FragCoord.x)" in code
    assert "GL_KHR_shader_subgroup" not in code


@pytest.mark.parametrize(
    "body, name",
    [
        ("return vec4(WaveGetLaneCount());", "WaveGetLaneCount"),
        ("subgroupExecutionBarrier(); return vec4(1.0);", "subgroupExecutionBarrier"),
        ("return vec4(float(gl_SubgroupSize));", "gl_SubgroupSize"),
    ],
)
def test_webgl_checks_only_emitted_stages_for_subgroup_intrinsics(body, name):
    ast = parse(f"""
    shader MixedStages {{
        vertex {{
            vec4 main() @ gl_Position {{ return vec4(1.0); }}
        }}
        fragment {{
            vec4 main() @ gl_FragColor {{ {body} }}
        }}
        compute {{
            void main() {{ uint count = WaveGetLaneCount(); }}
        }}
    }}
    """)
    generator = WebGLCodeGen()

    vertex = generator.generate_program(ast, target_stage="vertex")
    assert "gl_Position = vec4(1.0);" in vertex
    assert "gl_SubgroupSize" not in vertex
    with pytest.raises(StageUnsupportedError):
        generator.generate_program(ast, target_stage="compute")
    with pytest.raises(WebGLLoweringRefused, match=f"'{name}'"):
        generator.generate_program(ast)


@pytest.mark.parametrize("target_stage", ["fragment", None])
def test_webgl_does_not_shadow_barrier_with_other_stage_helper(target_stage):
    ast = parse("""
    shader MixedStages {
        vertex {
            float subgroupBarrier() { return 0.5; }
            vec4 main() @gl_Position { return vec4(subgroupBarrier()); }
        }
        fragment {
            vec4 main() @gl_FragColor {
                subgroupBarrier();
                return vec4(1.0);
            }
        }
    }
    """)
    generator = WebGLCodeGen()

    vertex = generator.generate_program(ast, target_stage="vertex")
    helper_name = _emitted_float_helper_name(vertex, "", "0.5")
    assert f"gl_Position = vec4({helper_name}());" in vertex
    assert "GL_KHR_shader_subgroup" not in vertex
    with pytest.raises(WebGLLoweringRefused, match="'subgroupBarrier'"):
        generator.generate_program(ast, target_stage=target_stage)
