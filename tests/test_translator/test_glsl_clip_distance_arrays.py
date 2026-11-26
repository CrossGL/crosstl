"""GLSL redeclares a distance built-in with its size when a stage structure names it as a whole array.

GLSL predeclares ``gl_ClipDistance`` and ``gl_CullDistance`` unsized (GLSL 4.60 section 7.1.1). An unsized array
can be neither assigned or read whole nor indexed with a non-constant index, so a CrossGL ``float clip[N]
@gl_ClipDistance`` member, which GLSL names as the whole array, gave ``gl_ClipDistance = float[1](0.0);`` and
``float c[2] = gl_ClipDistance;``, both of which glslang refuses. The vertex output and the fragment input now
redeclare the built-in with the size of the interface's distances, ``out float gl_ClipDistance[2];``. Scalar and
vector distance members index the array with constants and need no redeclaration.
"""

import shutil
import subprocess

import pytest

import crosstl.translator as translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

GLSLANG = shutil.which("glslangValidator")


def generate(source, stage):
    return GLSLCodeGen().generate_stage(translator.parse(source), stage)


def assert_compiles(tmp_path, stages):
    if GLSLANG is None:
        return
    paths = []
    for extension, code in stages.items():
        path = tmp_path / f"shader.{extension}"
        path.write_text(code)
        paths.append(str(path))
    result = subprocess.run(
        [GLSLANG, "-l", *paths], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def program(builtin, member_type, member, vertex_body, fragment_body):
    return f"""
shader Distances {{
    struct VertexOutput {{
        vec4 position @ gl_Position;
        {member_type} {member} @ {builtin};
    }};
    struct FragmentInput {{
        vec4 position @ gl_FragCoord;
        {member_type} {member} @ {builtin};
    }};
    vertex {{
        VertexOutput main(vec4 position @location(0)) {{
{vertex_body}
        }}
    }}
    fragment {{
        vec4 main(FragmentInput input) @ gl_FragColor {{
{fragment_body}
        }}
    }}
}}
"""


ARRAY_VERTEX = """
            VertexOutput output_ = VertexOutput(vec4(0.0), float[2](0.0, 0.0));
            output_.position = position;
            output_.clip = float[2](position.x, position.y);
            return output_;
"""

ARRAY_FRAGMENT = """
            float c[2] = input.clip;
            int k = int(input.position.x) % 2;
            return vec4(c[0], input.clip[k], 0.0, 1.0);
"""


@pytest.mark.parametrize("builtin", ["gl_ClipDistance", "gl_CullDistance"])
def test_whole_array_member_redeclares_the_sized_builtin(tmp_path, builtin):
    source = program(builtin, "float", "clip[2]", ARRAY_VERTEX, ARRAY_FRAGMENT)
    vertex = generate(source, "vertex")
    fragment = generate(source, "fragment")

    assert vertex.count(f"out float {builtin}[2];") == 1
    assert f"{builtin} = float[2](position.x, position.y);" in vertex
    assert fragment.count(f"in float {builtin}[2];") == 1
    assert f"float c[2] = {builtin};" in fragment
    assert_compiles(tmp_path, {"vert": vertex, "frag": fragment})


def test_one_element_array_is_redeclared_with_size_one(tmp_path):
    source = program(
        "gl_ClipDistance",
        "float",
        "clip[1]",
        """
            VertexOutput output_ = VertexOutput(vec4(0.0), float[1](0.0));
            output_.position = position * 2.0;
            output_.clip[0] = position.x;
            return output_;
""",
        "            return vec4(input.clip[0]);",
    )
    vertex = generate(source, "vertex")

    assert "out float gl_ClipDistance[1];" in vertex
    assert "gl_ClipDistance = float[1](0.0);" in vertex
    assert_compiles(tmp_path, {"vert": vertex, "frag": generate(source, "fragment")})


@pytest.mark.parametrize(
    ("member_type", "write", "read"),
    [
        ("float", "output_.clip = position.x;", "return vec4(input.clip);"),
        ("vec2", "output_.clip = position.xy;", "return vec4(input.clip, 0.0, 1.0);"),
    ],
)
def test_scalar_and_vector_members_keep_the_predeclared_builtin(
    tmp_path, member_type, write, read
):
    source = program(
        "gl_ClipDistance",
        member_type,
        "clip",
        f"""
            VertexOutput output_;
            output_.position = position;
            {write}
            return output_;
""",
        f"            {read}",
    )
    vertex = generate(source, "vertex")
    fragment = generate(source, "fragment")

    assert "float gl_ClipDistance[" not in vertex
    assert "float gl_ClipDistance[" not in fragment
    assert_compiles(tmp_path, {"vert": vertex, "frag": fragment})
