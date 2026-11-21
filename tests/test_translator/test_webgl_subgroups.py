import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.glsl_refusals import WebGLLoweringRefused
from crosstl.translator.codegen.stage_refusals import StageUnsupportedError
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen


def _fragment(expression, helpers=""):
    return f"""
    shader Probe {{
        {helpers}
        fragment {{
            vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {{
                return vec4({expression}, 0.0, 0.0, 1.0);
            }}
        }}
    }}
    """


@pytest.mark.parametrize(
    "intrinsic, expression, glsl_operation",
    [
        ("WaveGetLaneCount", "float(WaveGetLaneCount())", "gl_SubgroupSize"),
        ("WaveIsFirstLane", "WaveIsFirstLane() ? 1.0 : 0.0", "subgroupElect("),
        ("WaveActiveSum", "WaveActiveSum(pos.x)", "subgroupAdd("),
        (
            "WaveActiveAllTrue",
            "WaveActiveAllTrue(pos.x > 0.0) ? 1.0 : 0.0",
            "subgroupAll(",
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
        assert f"struct {name} {{" in code
        assert (
            f"float {name}_2(float x) {{\n    return (x * {name}(2.0).amount);" in code
        )
        assert f"float {name}_2(float x, float y) {{\n    return (x + y);" in code
        assert f"float {name}_(float x) {{\n    return (x - 1.0);" in code
        assert f"{name}_2(gl_FragCoord.x)" in code
        assert f"{name}_2(gl_FragCoord.y, 1.0)" in code
        assert f"{name}_(gl_FragCoord.z)" in code


def test_webgl_keeps_user_function_with_simd_alias_name():
    source = _fragment("simd_sum(pos.x)", "float simd_sum(float x) { return x + 1.0; }")

    code = WebGLCodeGen().generate(parse(source))

    assert code.startswith("#version 300 es\n")
    assert "simd_sum(gl_FragCoord.x)" in code
    assert "GL_KHR_shader_subgroup" not in code


def test_webgl_checks_only_emitted_stages_for_subgroup_intrinsics():
    ast = parse("""
    shader MixedStages {
        vertex {
            vec4 main() @ gl_Position { return vec4(1.0); }
        }
        fragment {
            vec4 main() @ gl_FragColor { return vec4(WaveGetLaneCount()); }
        }
        compute {
            void main() { uint count = WaveGetLaneCount(); }
        }
    }
    """)
    generator = WebGLCodeGen()

    vertex = generator.generate_program(ast, target_stage="vertex")
    assert "gl_Position = vec4(1.0);" in vertex
    assert "gl_SubgroupSize" not in vertex
    with pytest.raises(StageUnsupportedError):
        generator.generate_program(ast, target_stage="compute")
    with pytest.raises(
        WebGLLoweringRefused, match="subgroup intrinsic 'WaveGetLaneCount'"
    ):
        generator.generate_program(ast)
