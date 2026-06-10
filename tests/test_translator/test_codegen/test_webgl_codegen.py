from types import SimpleNamespace

import pytest

import crosstl
import crosstl.translator.codegen as codegen
from crosstl.formatter import format_shader_code
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

WEBGL_SHADER = """
shader WebGLSmoke {
    vertex {
        vec4 main(vec3 position @ POSITION) @ gl_Position {
            return vec4(position, 1.0);
        }
    }
    fragment {
        vec4 main() @ gl_FragColor {
            return vec4(1.0, 0.0, 0.0, 1.0);
        }
    }
}
"""


def parse_shader(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_webgl_backend_is_target_only():
    spec = codegen.get_backend("webgl2")

    assert spec is not None
    assert spec.name == "webgl"
    assert spec.source_registry_name is None
    assert "webgl" not in codegen.source_backend_names()
    assert codegen.normalize_backend_name("target.webgl.glsl") == "webgl"
    assert codegen.get_backend_extension("glsl-es") == ".webgl.glsl"
    assert isinstance(codegen.get_codegen("essl"), WebGLCodeGen)


def test_webgl_codegen_emits_glsl_es_header_and_default_precision():
    generated = WebGLCodeGen().generate(parse_shader(WEBGL_SHADER))
    stripped = generated.lstrip()

    assert stripped.startswith("#version 300 es\n")
    assert stripped.index("#version 300 es") < stripped.index("precision highp float;")
    assert "precision highp float;\n" in generated
    assert "precision highp int;\n" in generated
    assert "#version 450 core" not in generated
    assert "layout(location = 0) out vec4 fragColor;" in generated


def test_webgl_codegen_preserves_explicit_precision_qualifiers():
    shader = """
    precision mediump float;
    precision lowp int;
    shader WebGLPrecision {
        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "precision mediump float;" in generated
    assert "precision lowp int;" in generated
    assert "precision highp float;" not in generated
    assert "precision highp int;" not in generated


def test_webgl_aliases_format_as_glsl():
    assert format_shader_code("void main(){}", "webgl") == format_shader_code(
        "void main(){}", "glsl"
    )
    assert format_shader_code("void main(){}", "shader.webgl.glsl")


def test_translate_crossgl_to_webgl(tmp_path):
    source_path = tmp_path / "shader.cgl"
    source_path.write_text(WEBGL_SHADER, encoding="utf-8")

    generated = crosstl.translate(
        str(source_path), backend="webgl", format_output=False
    )

    assert "#version 300 es" in generated
    assert "precision highp float;" in generated


def test_webgl_codegen_rejects_non_webgl_stages():
    shader = """
    shader WebGLNoCompute {
        compute {
            void main() {
                return;
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="WebGL target does not support shader stage\\(s\\): compute",
    ):
        WebGLCodeGen().generate(parse_shader(shader))


@pytest.mark.parametrize(
    "stage",
    (
        "compute",
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
        "mesh",
        "task",
        "ray_generation",
    ),
)
def test_webgl_codegen_rejects_non_webgl_stage_targets(stage):
    ast = SimpleNamespace(functions=[], stages={})

    with pytest.raises(
        ValueError,
        match=rf"WebGL target does not support shader stage\(s\): {stage}",
    ):
        WebGLCodeGen().generate_program(ast, target_stage=stage)
