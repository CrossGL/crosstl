"""GLSL constants whose initializers read specialization constants.

GLSL output declares the global specialization constants after the other leading constants, so a constant reading
one used to precede it and failed with an undeclared identifier. When the GLSL is compiled to SPIR-V, glslang also
rejects a constant initializer that reads a specialization constant unless it can build it as an
``OpSpecConstantOp`` or an integer or Boolean ``OpSpecConstantComposite`` (SPIR-V 1.6 section 3.52.7): arrays,
structs, matrices and floating-point arithmetic fail with "global const initializers must be constant". Those
constants become global variables in the SPIR-V branch and stay constants for OpenGL source compilation.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.glsl_version_requirements import (
    extension_directives_end,
)

GLSLANG = shutil.which("glslangValidator")
SPIRV_VAL = shutil.which("spirv-val")
SPIRV_GUARD = "#if defined(GL_SPIRV) || defined(VULKAN)\n"

LAYOUT_ARRAY = """
shader SpecArray {
    layout(constant_id = 4) const int S = 5;
    const int arr[3] = {S, 1, 0};
    compute {
        layout(std430, set = 0, binding = 0) buffer Out { int values[]; } outp;
        layout(local_size_x = 1) in;
        void main() {
            uint x = gl_GlobalInvocationID.x;
            outp.values[x] = arr[x % 3u];
        }
    }
}
"""

DEPENDENTS = """
shader SpecDependents {
    const float BIAS = 0.25;
    const int SPEC_I @constant_id(0) = 3;
    const bool SPEC_B @constant_id(1) = true;
    const uint SPEC_U @constant_id(2) = 5u;
    struct Pair { uint a; int b; };
    const Pair pair = Pair(SPEC_I, SPEC_I);
    const bool nonzero = bool(SPEC_I);
    const int fromBool = int(SPEC_B);
    const uvec3 mixed = uvec3(SPEC_B, SPEC_I, SPEC_U);
    const int doubled = int(SPEC_U) * 2;
    const int remainder = SPEC_I % 3;
    const int chosen = SPEC_B ? 1 : 2;
    const uint table[2] = {SPEC_I, 1u};
    const float scale = float(SPEC_I) * 0.5;
    const float scaledTwice = scale * 2.0;
    const int truncated = int(scale);
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = float(pair.a) + float(pair.b) + (nonzero ? 1.0 : 0.0) + float(fromBool)
                + float(mixed.x + mixed.y + mixed.z) + float(doubled + remainder + chosen)
                + float(table[0]) + scale + scaledTwice + float(truncated) + BIAS;
        }
    }
}
"""

ES_COMPUTE = """#version 310 es
layout(local_size_x = 1) in;
layout(constant_id = 0) const int S = 3;
const int arr[2] = int[2](S, 1);
const float scale = float(S) * 0.5;
const int twice = S * 2;
layout(std430, binding = 0) buffer Out { float values[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    values[i] = float(arr[i % 2u] + twice) * scale;
}
"""

ES_FRAGMENT = """#version 300 es
precision highp float;
layout(constant_id = 0) const int S = 3;
const float scale = float(S) * 0.5;
out vec4 color;
void main() { color = vec4(scale, 0.0, 0.0, 1.0); }
"""


def translate_crossgl(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def translate_glsl(tmp_path, source, suffix, backend="opengl"):
    path = tmp_path / f"source.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, source_backend="opengl")


def runtime_declaration(declaration, value):
    return (
        f"{SPIRV_GUARD}{declaration} = {value};\n"
        f"#else\nconst {declaration} = {value};\n#endif\n"
    )


def assert_glslang_accepts(tmp_path, code, stage, spirv=True):
    if GLSLANG is None:
        return
    shader = tmp_path / f"out.{stage}"
    shader.write_text(code, encoding="utf-8")
    compiled = subprocess.run(
        [GLSLANG, "-S", stage, str(shader)], capture_output=True, text=True
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    if not spirv:
        return
    module = tmp_path / f"out.{stage}.spv"
    compiled = subprocess.run(
        [GLSLANG, "-V", "-S", stage, str(shader), "-o", str(module)],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    if SPIRV_VAL is not None:
        validated = subprocess.run(
            [SPIRV_VAL, str(module)], capture_output=True, text=True
        )
        assert validated.returncode == 0, validated.stdout + validated.stderr


def test_constant_reading_a_specialization_constant_follows_it(tmp_path):
    code = translate_crossgl(tmp_path, LAYOUT_ARRAY)

    specialization = code.index("layout(constant_id = 4) const int S = 5;")
    assert code.index("int arr[3] = int[3](S, 1, 0);") > specialization
    assert runtime_declaration("int arr[3]", "int[3](S, 1, 0)") in code
    assert_glslang_accepts(tmp_path, code, "comp")


def test_non_integer_dependents_are_globals_when_compiled_to_spirv(tmp_path):
    code = translate_crossgl(tmp_path, DEPENDENTS)

    assert runtime_declaration("uint table[2]", "uint[2](uint(SPEC_I), 1u)") in code
    assert runtime_declaration("Pair pair", "Pair(uint(SPEC_I), SPEC_I)") in code
    assert runtime_declaration("float scale", "(float(SPEC_I) * 0.5)") in code
    # Constants that read a runtime dependent are runtime too, even integer ones.
    assert runtime_declaration("float scaledTwice", "(scale * 2.0)") in code
    assert runtime_declaration("int truncated", "int(scale)") in code
    assert "GL_EXT_shader_non_constant_global_initializers" not in code
    assert_glslang_accepts(tmp_path, code, "comp")


def test_integer_and_boolean_specialization_operations_stay_constants(tmp_path):
    code = translate_crossgl(tmp_path, DEPENDENTS)

    for declaration in (
        "const float BIAS = 0.25;",
        "const bool nonzero = bool(SPEC_I);",
        "const int fromBool = int(SPEC_B);",
        "const uvec3 mixed = uvec3(SPEC_B, SPEC_I, SPEC_U);",
        "const int doubled = (int(SPEC_U) * 2);",
        # CrossGL % truncates toward zero, so it lowers to integer division.
        "const int remainder = (SPEC_I - (SPEC_I / 3) * 3);",
        "const int chosen = (SPEC_B ? 1 : 2);",
    ):
        assert f"\n{declaration}\n" in code
        assert f"#else\n{declaration}" not in code
    # An independent constant keeps its place ahead of the specialization constants.
    assert code.index("const float BIAS") < code.index("constant_id = 0")
    assert code.index("constant_id = 2") < code.index("const bool nonzero")


def test_es_enables_non_constant_global_initializers_for_spirv(tmp_path):
    code = translate_glsl(tmp_path, ES_COMPUTE, "comp")

    lines = code.splitlines()
    assert lines[:4] == [
        "#version 310 es",
        "#if defined(GL_SPIRV) || defined(VULKAN)",
        "#extension GL_EXT_shader_non_constant_global_initializers : enable",
        "#endif",
    ]
    assert lines[4].startswith("precision highp ")
    assert runtime_declaration("int arr[2]", "int[2](S, 1)") in code
    assert runtime_declaration("float scale", "(float(S) * 0.5)") in code
    assert "\nconst int twice = (S * 2);\n" in code
    assert_glslang_accepts(tmp_path, code, "comp")


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_es_extension_precedes_source_precision_statements(tmp_path, backend):
    code = translate_glsl(tmp_path, ES_FRAGMENT, "frag", backend=backend)

    # GLSL ES 3.20 section 3.4: #extension precedes every non-preprocessor token.
    extension = code.index("#extension GL_EXT_shader_non_constant_global_initializers")
    assert extension < code.index("precision ")
    assert runtime_declaration("float scale", "(float(S) * 0.5)") in code
    # GLSL ES 3.00 has no SPIR-V path, so only the OpenGL ES compile applies.
    assert_glslang_accepts(tmp_path, code, "frag", spirv=False)


def test_shaders_without_dependents_gain_no_guards(tmp_path):
    code = translate_glsl(tmp_path, ES_FRAGMENT.replace("float(S)", "3.0"), "frag")

    assert "GL_EXT_shader_non_constant_global_initializers" not in code
    assert "\nconst float scale = (3.0 * 0.5);\n" in code


@pytest.mark.parametrize(
    "lines, expected",
    [
        (["#version 310 es", "#extension A : enable", "void main() {}"], 2),
        (
            [
                "#version 310 es",
                "#if defined(GL_SPIRV)",
                "#extension A : enable",
                "#else",
                "#extension B : enable",
                "#endif",
                "#extension C : require",
                "precision highp float;",
            ],
            7,
        ),
        (
            [
                "#version 310 es",
                "#if defined(GL_SPIRV)",
                "layout(constant_id = 0) const int S = 3;",
                "#endif",
            ],
            1,
        ),
        (["#version 310 es", "#if defined(GL_SPIRV)", "#extension A : enable"], 1),
    ],
)
def test_extension_directives_end_skips_extension_only_conditionals(lines, expected):
    assert extension_directives_end(lines, 1) == expected
