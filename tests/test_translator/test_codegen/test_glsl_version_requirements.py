"""The OpenGL emitter emits each program at the lowest ``#version`` its constructs need.

Every generated program here is also compiled with glslangValidator when it is installed, per stage for programs
that carry several stages.
"""

import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_refusals import (
    OpenGLLoweringRefused,
    WebGLLoweringRefused,
)
from crosstl.translator.codegen.glsl_version_requirements import (
    BUILTIN_FUNCTION_REQUIREMENTS,
    generated_program_requirements,
    resolve_version_line,
    with_es_default_precisions,
)
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen


def compile_with_glslang(tmp_path, code, stage):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    source = tmp_path / f"program.{stage}"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [glslang, str(source)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def generate(source, stage=None, codegen=GLSLCodeGen):
    ast = crosstl.translator.parse(source)
    if stage is None:
        return codegen().generate(ast)
    return codegen().generate_stage(ast, stage)


TWO_STAGE_SHADER = """
{version}
shader TwoStage {{
    struct VertexInput {{ vec3 position @ POSITION; vec2 texCoord @ TEXCOORD0; }}
    struct VertexOutput {{ vec2 uv @ TEXCOORD0; vec4 position @ gl_Position; }}
    struct FragmentInput {{ vec2 uv @ TEXCOORD0; }}
    struct FragmentOutput {{ vec4 color @ gl_FragColor; }}
    vertex {{
        VertexOutput main(VertexInput input) {{
            VertexOutput output;
            output.uv = input.texCoord;
            output.position = vec4(input.position, 1.0);
            return output;
        }}
    }}
    fragment {{
        FragmentOutput main(FragmentInput input) {{
            FragmentOutput output;
            output.color = vec4(input.uv, 0.5, 1.0);
            return output;
        }}
    }}
}}
"""


@pytest.mark.parametrize(
    "version,emitted",
    [
        ("#version 330 core", "#version 410 core"),
        ("#version 300 es", "#version 310 es"),
    ],
)
def test_varying_locations_raise_version(tmp_path, version, emitted):
    # Locations on vertex outputs and fragment inputs need GLSL 4.10 or GLSL ES
    # 3.10 (section 4.4.1 and 4.4.2); vertex inputs and fragment outputs have
    # them from 3.30 and ES 3.00.
    code = generate(TWO_STAGE_SHADER.format(version=version))

    assert code.startswith(f"{emitted}\n")
    assert "layout(location = 5) out vec2 uv;" in code
    compile_with_glslang(tmp_path, code, "vert")
    compile_with_glslang(tmp_path, code, "frag")


VERTEX_ONLY_SHADER = """
#version 330 core
shader VertexOnly {
    struct VertexInput { vec3 position @ POSITION; }
    struct VertexOutput { vec4 position @ gl_Position; }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = vec4(input.position, 1.0);
            return output;
        }
    }
}
"""


def test_vertex_input_locations_keep_glsl_330(tmp_path):
    code = generate(VERTEX_ONLY_SHADER, "vertex")

    assert code.startswith("#version 330 core\n")
    assert "layout(location = 0) in vec3 position;" in code
    compile_with_glslang(tmp_path, code, "vert")


SAMPLER_SHADER = """
{version}
shader Sampled {{
    sampler2D tex @binding(3);
    fragment {{
        vec4 main() @gl_FragColor {{
            return {expression};
        }}
    }}
}}
"""


def test_es_300_omits_binding_layouts(tmp_path):
    # layout(binding) needs GLSL ES 3.10 (section 4.4.5).
    code = generate(
        SAMPLER_SHADER.format(
            version="#version 300 es", expression="texture(tex, vec2(0.5))"
        ),
        "fragment",
    )

    assert code.startswith("#version 300 es\n")
    assert "binding" not in code
    assert "uniform sampler2D tex;" in code
    compile_with_glslang(tmp_path, code, "frag")


@pytest.mark.parametrize(
    "version,emitted,binding",
    [
        ("#version 330", "#version 400 core", False),
        ("#version 300 es", "#version 310 es", True),
        ("", "#version 450 core", True),
    ],
)
def test_texture_gather_offset_raises_version(tmp_path, version, emitted, binding):
    # textureGatherOffset arrived in GLSL 4.00 and GLSL ES 3.10; the program is
    # emitted for that version throughout, so ES 3.10 also keeps its bindings.
    code = generate(
        SAMPLER_SHADER.format(
            version=version,
            expression="textureGatherOffset(tex, vec2(0.5), ivec2(1, 0))",
        ),
        "fragment",
    )

    assert code.startswith(f"{emitted}\n")
    assert ("layout(binding = 3) uniform sampler2D tex;" in code) is binding
    compile_with_glslang(tmp_path, code, "frag")


VARIABLE_GATHER_SHADER = """
{version}
shader VariableGather {{
    sampler2D tex;
    struct FSInput {{ vec2 uv @ TEXCOORD0; ivec2 offset @ TEXCOORD1; }}
    fragment {{
        vec4 main(FSInput input) @gl_FragColor {{
            return textureGatherOffset(tex, input.uv, input.offset);
        }}
    }}
}}
"""


@pytest.mark.parametrize(
    "version,emitted",
    [
        # The fragment input locations alone need GLSL 4.10.
        ("#version 330", "#version 410 core"),
        ("#version 310 es", "#version 320 es"),
        (
            "#version 310 es\n#extension GL_EXT_gpu_shader5 : require",
            "#version 310 es",
        ),
    ],
)
def test_variable_gather_offset_raises_version(tmp_path, version, emitted):
    # A non-constant gather offset needs GLSL 4.00, GLSL ES 3.20 or gpu_shader5
    # (section 8.9.4).
    code = generate(VARIABLE_GATHER_SHADER.format(version=version), "fragment")

    assert code.startswith(f"{emitted}\n")
    assert "textureGatherOffset(tex, uv, offset)" in code
    compile_with_glslang(tmp_path, code, "frag")


INTEGER_SAMPLER_SHADER = """
#version 300 es
shader IntegerSampler {
    isampler2D itex;
    sampler2DArray layers;
    fragment {
        vec4 main() @gl_FragColor {
            return vec4(texelFetch(itex, ivec2(0), 0)) + texture(layers, vec3(0.5));
        }
    }
}
"""


def test_es_declares_default_precision_for_opaque_types(tmp_path):
    # GLSL ES has no default precision for float in fragment shaders or for most
    # opaque types (GLSL ES 3.20, section 4.7.4).
    code = generate(INTEGER_SAMPLER_SHADER, "fragment")

    lines = code.splitlines()
    assert lines[0] == "#version 300 es"
    for kind in ("float", "int", "isampler2D", "sampler2DArray"):
        assert f"precision highp {kind};" in lines
    compile_with_glslang(tmp_path, code, "frag")


def test_es_keeps_declared_precisions():
    code = with_es_default_precisions(
        "#version 300 es\nprecision mediump float;\nuniform sampler3D v;\n"
    )

    assert code.splitlines() == [
        "#version 300 es",
        "precision highp int;",
        "precision highp sampler3D;",
        "precision mediump float;",
        "uniform sampler3D v;",
    ]


QUERY_LOD_SHADER = """
#version 300 es
shader QueryLod {
    sampler2D tex;
    fragment {
        vec4 main() @gl_FragColor {
            return vec4(textureQueryLod(tex, vec2(0.5)), 0.0, 1.0);
        }
    }
}
"""


def test_es_refuses_builtins_it_does_not_have():
    with pytest.raises(OpenGLLoweringRefused) as raised:
        generate(QUERY_LOD_SHADER, "fragment")

    assert (
        "GLSL version requirement: built-in function 'textureQueryLod' does not exist "
        "in GLSL ES" in str(raised.value)
    )


ATOMIC_COUNTER_OPS_SHADER = """
{version}
shader AtomicCounterOps {{
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* outp;
        uniform atomic_uint u0;
        void main() {{
            outp[0] = {call};
        }}
    }}
}}
"""


@pytest.mark.parametrize(
    "call",
    [
        "atomicCounterAdd(u0, 2u)",
        "atomicCounterSubtract(u0, 2u)",
        "atomicCounterMin(u0, 2u)",
        "atomicCounterMax(u0, 2u)",
        "atomicCounterAnd(u0, 2u)",
        "atomicCounterOr(u0, 2u)",
        "atomicCounterXor(u0, 2u)",
        "atomicCounterExchange(u0, 2u)",
        "atomicCounterCompSwap(u0, 1u, 2u)",
    ],
)
def test_atomic_counter_operations_raise_version(tmp_path, call):
    # The atomic counter operations beyond increment and decrement arrived in
    # GLSL 4.60 (section 8.10); GLSL ES has none of them.
    code = generate(ATOMIC_COUNTER_OPS_SHADER.format(version="", call=call), "compute")

    assert code.startswith("#version 460 core\n")
    assert f"outp[0] = {call};" in code
    compile_with_glslang(tmp_path, code, "comp")

    with pytest.raises(OpenGLLoweringRefused) as raised:
        generate(
            ATOMIC_COUNTER_OPS_SHADER.format(version="#version 310 es", call=call),
            "compute",
        )
    name = call.split("(")[0]
    assert (
        f"GLSL version requirement: built-in function '{name}' does not exist "
        "in GLSL ES" in str(raised.value)
    )


FIND_MSB_SHADER = """
shader FindMsb {
    isampler2D itex;
    fragment {
        vec4 main() @gl_FragColor {
            return vec4(float(findMSB(texelFetch(itex, ivec2(0), 0).x)));
        }
    }
}
"""


def test_webgl_refuses_constructs_above_glsl_es_300():
    # WebGL 2 compiles GLSL ES 3.00 only, so its version line is never raised.
    with pytest.raises(WebGLLoweringRefused) as raised:
        generate(FIND_MSB_SHADER, "fragment", codegen=WebGLCodeGen)

    assert (
        "GLSL version requirement: built-in function 'findMSB' requires GLSL ES 3.10; "
        "the target is #version 300 es" in str(raised.value)
    )


def test_opengl_es_raises_find_msb(tmp_path):
    code = generate("#version 300 es\n" + FIND_MSB_SHADER, "fragment")

    assert code.startswith("#version 310 es\n")
    compile_with_glslang(tmp_path, code, "frag")


def test_requirements_follow_stage_guards():
    code = "\n".join(
        [
            "#version 330 core",
            "#ifdef GL_VERTEX_SHADER",
            "layout(location = 0) in vec3 p;",
            "#endif",
            "#ifdef GL_FRAGMENT_SHADER",
            "layout(location = 0) out vec4 c;",
            "#endif",
        ]
    )
    guards = {"vertex": "GL_VERTEX_SHADER", "fragment": "GL_FRAGMENT_SHADER"}

    requirements = generated_program_requirements(code, ["fragment", "vertex"], guards)

    assert resolve_version_line("#version 330 core", requirements, set()) == (
        "#version 330 core",
        [],
    )
    swapped = code.replace("GL_VERTEX_SHADER", "GL_TMP").replace(
        "GL_FRAGMENT_SHADER", "GL_VERTEX_SHADER"
    )
    swapped = swapped.replace("GL_TMP", "GL_FRAGMENT_SHADER")
    requirements = generated_program_requirements(
        swapped, ["fragment", "vertex"], guards
    )
    assert resolve_version_line("#version 330 core", requirements, set()) == (
        "#version 410 core",
        [],
    )


def test_extensions_satisfy_requirements_without_raising():
    requirement = BUILTIN_FUNCTION_REQUIREMENTS["textureGatherOffset"]

    assert resolve_version_line(
        "#version 330", [requirement], {"GL_ARB_gpu_shader5"}
    ) == ("#version 330", [])
    assert resolve_version_line("#version 330", [requirement], set()) == (
        "#version 400 core",
        [],
    )
    assert resolve_version_line("#version 330 compatibility", [requirement], set()) == (
        "#version 400 compatibility",
        [],
    )


def test_fixed_version_refuses_instead_of_raising():
    requirement = BUILTIN_FUNCTION_REQUIREMENTS["bitCount"]

    assert resolve_version_line(
        "#version 300 es", [requirement], set(), can_raise=False
    ) == (
        "#version 300 es",
        [
            "built-in function 'bitCount' requires GLSL ES 3.10; "
            "the target is #version 300 es"
        ],
    )
