"""Stage-scope private variables keep one copy per invocation that every function of the stage shares.

``float counter = 1.0;`` at stage scope is SPIR-V ``Private`` storage. The GLSL and HLSL generators used to drop
the declaration and keep its uses, the Slang generator declared it as a uniform, and the Metal generator made it a
local of the entry point that the stage's helpers could not see (issue #3187).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.stage_private_variables import (
    hoist_stage_private_variables,
)

COUNTER = """
shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        float counter = 1.0;

        void bump() {
            counter = counter * 2.0 + 1.0;
        }

        void main() {
            uint i = gl_GlobalInvocationID.x;
            float before = counter;
            counter = fin[i];
            bump();
            outp[i] = counter * 10.0 + before;
        }
    }
}
"""

ENTRY_ONLY = """
shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        float counter = 1.0;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            counter = counter + 2.0;
            outp[i] = counter;
        }
    }
}
"""

MIXED = """
shader P {
    struct Acc {
        float sum;
        int n;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        float base = 3.0;
        float scaled = base * 2.0;
        Acc acc;
        float hist[3];
        int counter = 0;

        void add(float v) {
            acc.sum = acc.sum + v;
            acc.n = acc.n + 1;
            hist[counter] = v;
            counter = counter + 1;
        }

        float shadow(float counter) {
            return counter * 2.0;
        }

        float local_shadow() {
            float base = 100.0;
            return base;
        }

        void main() {
            uint i = gl_GlobalInvocationID.x;
            acc.sum = 0.0;
            acc.n = 0;
            add(fin[i]);
            add(scaled);
            add(shadow(1.0));
            outp[i] = acc.sum * 1000.0 + float(acc.n) * 100.0 + hist[2] + local_shadow() + base;
        }
    }
}
"""

TWO_STAGES = """
shader P {
    struct VIn {
        vec3 position;
    }
    struct VOut {
        vec4 position;
    }
    struct FOut {
        vec4 color;
    }
    vertex {
        float scale = 2.0;
        void grow() {
            scale = scale * 3.0;
        }
        VOut main(VIn input) {
            VOut o;
            grow();
            o.position = vec4(input.position * scale, 1.0);
            return o;
        }
    }
    fragment {
        int scale = 4;
        FOut main() {
            FOut o;
            scale = scale + 1;
            o.color = vec4(float(scale));
            return o;
        }
    }
}
"""

KEPT = """
shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        uniform float u;
        const float K = 2.0;
        shared float tile[4];
        sampler2D tex;
        float value = 1.0;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = value + u + K + tile[0];
        }
    }
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def _compile(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _before(code, first, second):
    return code.index(first) < code.index(second)


def test_glsl_declares_the_variable_at_module_scope(tmp_path):
    code = _translate(tmp_path, COUNTER, "opengl")
    assert "float counter = 1.0;" in code
    assert _before(code, "float counter = 1.0;", "void bump()")
    validator = shutil.which("glslangValidator")
    if validator is not None:
        (tmp_path / "out.comp").write_text(code)
        _compile([validator, "-S", "comp", "out.comp"], tmp_path)


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_hlsl_and_slang_declare_a_static_global(tmp_path, backend):
    code = _translate(tmp_path, COUNTER, backend)
    assert "static float counter = 1.0;" in code
    assert _before(code, "static float counter = 1.0;", "void bump()")


def test_slang_keeps_static_on_module_scope_values(tmp_path):
    source = COUNTER.replace("        float counter = 1.0;\n", "").replace(
        "shader P {\n", "shader P {\n    static float counter = 1.0;\n", 1
    )
    code = _translate(tmp_path, source, "slang")
    assert "static float counter = 1.0;" in code


def test_metal_passes_the_variable_to_the_helpers(tmp_path):
    code = _flat(_translate(tmp_path, COUNTER, "metal"))
    assert "void bump(thread float& counter)" in code
    assert "bump(counter);" in code
    assert re.search(r"kernel void \w+\(.*\{ .*float counter = 1\.0;", code)
    if shutil.which("xcrun") is not None:
        (tmp_path / "out.metal").write_text(_translate(tmp_path, COUNTER, "metal"))
        _compile(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )


def test_metal_keeps_an_entry_only_variable_as_a_local(tmp_path):
    program = _parse(ENTRY_ONLY)
    assert hoist_stage_private_variables(program, "metal") == 0
    code = _flat(_translate(tmp_path, ENTRY_ONLY, "metal"))
    assert "float counter = 1.0;" in code
    assert "thread float" not in code


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_entry_only_variable_is_declared(tmp_path, backend):
    code = _translate(tmp_path, ENTRY_ONLY, backend)
    assert re.search(r"(static )?float counter = 1\.0;", code)


def test_hoisted_declarations_are_static_module_values():
    program = _parse(COUNTER)
    stage = next(iter(program.stages.values()))
    assert hoist_stage_private_variables(program, "opengl") == 1
    assert [v.name for v in stage.local_variables] == ["fin", "outp"]
    hoisted = program.global_variables[-1]
    assert hoisted.name == "counter"
    assert hoisted.qualifiers == ["static"]
    assert hoisted.parent is program


def test_non_constant_initializer_runs_at_the_start_of_the_entry():
    program = _parse(MIXED)
    stage = next(iter(program.stages.values()))
    hoist_stage_private_variables(program, "directx")
    by_name = {v.name: v for v in program.global_variables}
    assert by_name["scaled"].initial_value is None
    first = stage.entry_point.body.statements[0]
    assert first.target.name == "scaled"
    # `base` is redeclared by a helper, so the stage variable takes a fresh name everywhere it is meant.
    assert first.value.left.name == "compute_base"
    assert by_name["compute_base"].initial_value.value == 3.0


def test_shadowing_declarations_keep_their_names(tmp_path):
    code = _flat(_translate(tmp_path, MIXED, "opengl"))
    assert "float compute_base = 3.0;" in code
    assert "int compute_counter = 0;" in code
    assert "float shadow(float counter) { return (counter * 2.0); }" in code
    assert "float base = 100.0; return base;" in code
    assert "hist[compute_counter] = v;" in code
    assert "scaled = (compute_base * 2.0); uint i" in code


def test_variables_of_different_stages_do_not_collide(tmp_path):
    code = _flat(_translate(tmp_path, TWO_STAGES, "directx"))
    assert "static float scale = 2.0;" in code
    assert "static int fragment_scale = 4;" in code
    assert "fragment_scale = (fragment_scale + 1);" in code
    assert "scale = (scale * 3.0);" in code
    glsl = _flat(_translate(tmp_path, TWO_STAGES, "opengl"))
    assert "float scale = 2.0;" in glsl and "int fragment_scale = 4;" in glsl


def test_qualified_and_opaque_declarations_stay_in_the_stage():
    program = _parse(KEPT)
    stage = next(iter(program.stages.values()))
    assert hoist_stage_private_variables(program, "opengl") == 1
    assert [v.name for v in stage.local_variables] == [
        "outp",
        "u",
        "K",
        "tile",
        "tex",
    ]


@pytest.mark.parametrize("target", ["vulkan", "wgsl", "cuda", "cgl"])
def test_targets_with_their_own_private_storage_are_unchanged(target):
    program = _parse(COUNTER)
    assert hoist_stage_private_variables(program, target) == 0
