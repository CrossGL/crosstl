import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen


HELPER_RANGE_SHADER = """
shader RangeForInValidation {
    int helper(int limit) {
        int total = 0;
        for i in 1..=limit {
            total = total + i;
        }
        return total;
    }
}
"""


FRAGMENT_RANGE_SHADER = """
shader FragmentRangeValidation {
    fragment {
        vec4 main() @ gl_FragColor {
            int total = 0;
            for i in 1..=3 {
                total = total + i;
            }
            return vec4(float(total), 0.0, 0.0, 1.0);
        }
    }
}
"""


FRAGMENT_STRUCT_INPUT_SHADER = """
shader FragmentStructInputValidation {
    struct VSOutput {
        vec2 uv @ TEXCOORD0;
        vec3 normal @ NORMAL;
    };

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            return vec4(input.uv, input.normal.x, 1.0);
        }
    }
}
"""


VERTEX_STRUCT_IO_SHADER = """
shader VertexStructIOValidation {
    struct VSInput {
        vec3 position @ POSITION;
        vec2 uv @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
    };

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            output.position = vec4(input.position, 1.0);
            output.uv = input.uv;
            return output;
        }
    }
}
"""


COMBINED_STAGE_IO_SHADER = """
shader CombinedStageIOValidation {
    struct VSInput {
        vec3 position @ POSITION;
        vec2 uv @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
    };

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            output.position = vec4(input.position, 1.0);
            output.uv = input.uv;
            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            return vec4(input.uv, 0.0, 1.0);
        }
    }
}
"""


def run_validator(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"{' '.join(command)} failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_generated_metal_fragment_smoke_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_range.metal"
    output = tmp_path / "fragment_range.air"
    code = MetalCodeGen().generate(crosstl.translator.parse(FRAGMENT_RANGE_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_glsl_fragment_smoke_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_range.frag"
    code = GLSLCodeGen().generate(crosstl.translator.parse(FRAGMENT_RANGE_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_struct_input_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_struct_input.frag"
    code = GLSLCodeGen().generate(
        crosstl.translator.parse(FRAGMENT_STRUCT_INPUT_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_vertex_struct_io_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "vertex_struct_io.vert"
    code = GLSLCodeGen().generate(crosstl.translator.parse(VERTEX_STRUCT_IO_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "vert", str(source)])


def test_generated_glsl_combined_stages_validate_separately_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    ast = crosstl.translator.parse(COMBINED_STAGE_IO_SHADER)
    generator = GLSLCodeGen()

    vertex_source = tmp_path / "combined_stage.vert"
    vertex_source.write_text(generator.generate_stage(ast, "vertex"), encoding="utf-8")
    run_validator([glslang, "-S", "vert", str(vertex_source)])

    fragment_source = tmp_path / "combined_stage.frag"
    fragment_source.write_text(
        generator.generate_stage(ast, "fragment"), encoding="utf-8"
    )
    run_validator([glslang, "-S", "frag", str(fragment_source)])


def test_generated_hlsl_helper_smoke_validates_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "range_for_in.hlsl"
    output = tmp_path / "range_for_in.dxil"
    code = HLSLCodeGen().generate(crosstl.translator.parse(HELPER_RANGE_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator([dxc, "-T", "lib_6_3", str(source), "-Fo", str(output)])
