"""GLSL starts a local's scope after its initializer; C++-scoped targets need the local renamed."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.declaration_scopes import separate_initializer_scopes
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

SHADOWING = """#version 450
struct Instance { mat4 model; vec4 color; };
layout(std430, binding = 0) readonly buffer Instances { Instance instance[]; };
layout(location = 0) in vec4 inPos;
layout(location = 0) out vec4 outColor;
void main() {
    const Instance instance = instance[gl_VertexIndex];
    float scale = 2.0;
    {
        float scale = scale * 3.0;
        outColor = instance.color * scale;
    }
    gl_Position = instance.model * inPos * scale;
}
"""


def translate(tmp_path, backend):
    path = tmp_path / "shader.vert"
    path.write_text(SHADOWING)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_local_shadowing_a_buffer_member_is_read_after_its_declaration(tmp_path):
    # The initializer reads the block member; later uses read the local (they used to read the member too).
    cgl = translate(tmp_path, "cgl")
    assert "const Instance instance = instances.instance[gl_VertexIndex];" in cgl
    assert "(instance.color * scale)" in cgl
    assert (
        "instances.instance.color" not in cgl and "instances.instance.model" not in cgl
    )
    glsl = translate(tmp_path, "opengl")
    if shutil.which("glslangValidator"):
        (tmp_path / "out.vert").write_text(glsl)
        _run(["glslangValidator", "-S", "vert", "out.vert"], tmp_path)


def test_hlsl_and_metal_rename_a_local_that_its_initializer_reads(tmp_path):
    for backend in ("directx", "metal"):
        output = translate(tmp_path, backend)
        assert "float scale_1 = " in output and "scale * 3.0" in output
        assert "float scale = (scale * 3.0)" not in output
        assert "* scale_1" in output
    hlsl = translate(tmp_path, "directx")
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        _run(["dxc", "-T", "vs_6_0", "-E", "VSMain", "out.hlsl"], tmp_path)


def test_glsl_output_keeps_glsl_scoping(tmp_path):
    glsl = translate(tmp_path, "opengl")
    assert "float scale = (scale * 3.0);" in glsl


def _program(source):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(source)


SCOPES = (
    "shader s {\n"
    "    compute {\n"
    "        void main() {\n"
    "            float a = 1.0;\n"
    "            for (int i = 0; i < 2; i++) { float a = a + 1.0; a = a * 2.0; }\n"
    "            if (a > 0.0) { float b = a; float a = b + a; b = a; }\n"
    "            a = a + 3.0;\n"
    "        }\n"
    "    }\n"
    "}\n"
)


def test_renames_follow_the_scopes(tmp_path):
    program = _program(SCOPES)
    assert separate_initializer_scopes(program, "cgl", "directx") == 2
    path = tmp_path / "scopes.cgl"
    path.write_text(SCOPES)
    hlsl = " ".join(crosstl.translate(str(path), backend="directx").split())
    # The loop body declares and writes its own a_1; the if body's b reads the outer a before a_2 hides it.
    assert "float a_1 = (a + 1.0); a_1 = (a_1 * 2.0);" in hlsl
    assert "float b = a; float a_2 = (b + a); b = a_2;" in hlsl
    assert "a = (a + 3.0);" in hlsl


def test_only_glsl_sources_on_cxx_targets_are_renamed():
    assert separate_initializer_scopes(_program(SCOPES), "cgl", "opengl") == 0
    assert separate_initializer_scopes(_program(SCOPES), "directx", "metal") == 0


LATER_NAMES = (
    "shader s {\n"
    "    compute {\n"
    "        void main() {\n"
    "            float a = 1.0;\n"
    "            { float a = a + 1.0; a = a * 2.0; }\n"
    "            float a_1 = a;\n"
    "            a_1 = a_1 + 1.0;\n"
    "        }\n"
    "    }\n"
    "}\n"
)


def test_a_new_name_avoids_names_declared_after_the_rename(tmp_path):
    # The names in use are collected when the first local is renamed; they include the later a_1.
    assert separate_initializer_scopes(_program(LATER_NAMES), "cgl", "directx") == 1
    path = tmp_path / "later.cgl"
    path.write_text(LATER_NAMES)
    hlsl = " ".join(crosstl.translate(str(path), backend="directx").split())
    assert "float a_2 = (a + 1.0); a_2 = (a_2 * 2.0);" in hlsl
    assert "float a_1 = a;" in hlsl


def _nested(statement):
    return (
        "shader s {\n"
        "    compute {\n"
        "        void main() {\n"
        "            float a = 1.0;\n"
        "            int n = 2;\n"
        f"            {statement}\n"
        "        }\n"
        "    }\n"
        "}\n"
    )


NESTED_HAZARDS = [
    "while (n > 0) { n = n - 1; { float a = a + 1.0; } }",
    "if (n > 0) { n = 1; } else { if (a > 0.0) { float a = a * 2.0; } }",
    "for (int i = 0; i < n; i++) { for (int j = 0; j < 2; j++) { float a = a + j; } }",
    "switch (n) { case 0: { float a = a - 1.0; break; } default: break; }",
]


@pytest.mark.parametrize("statement", NESTED_HAZARDS)
def test_a_hazard_under_any_statement_is_renamed(statement):
    # A program without a hazard skips the renamer, so the search must reach every declaration the renamer visits.
    assert separate_initializer_scopes(_program(_nested(statement)), "cgl", "directx") == 1


def test_a_program_without_a_hazard_is_left_unchanged():
    source = _nested("{ float b = a + 1.0; float c = b; a = c; }")
    program = _program(source)
    before = _every_name(program)
    assert separate_initializer_scopes(program, "cgl", "directx") == 0
    assert _every_name(program) == before
    assert {"a", "b", "c", "n"} <= set(before)


def _every_name(node):
    """Every ``name`` string in the tree, in walk order."""
    names, stack, seen = [], [node], set()
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        elif isinstance(item, dict):
            stack.extend(reversed(list(item.values())))
        elif hasattr(item, "__dict__") and id(item) not in seen:
            seen.add(id(item))
            if isinstance(getattr(item, "name", None), str):
                names.append(item.name)
            stack.extend(
                reversed([v for k, v in vars(item).items() if k != "parent"])
            )
    return names
