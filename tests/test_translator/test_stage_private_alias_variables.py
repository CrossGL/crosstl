"""A stage-scope private variable whose type is named through a ``typedef`` is declared like one with the type written.

``real scratch;`` for ``typedef float real;`` at stage scope was dropped by the GLSL and HLSL generators and declared
as a uniform by the Slang generator, while ``float scratch;`` was declared as a module-scope private value (#3818).
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.stage_private_variables import (
    hoist_stage_private_variables,
)

PROGRAM_ALIAS = """shader P {
    typedef float real;
    compute {
        buffer float* outp;
        real scratch;
        void main() {
            scratch = outp[1];
            outp[0] = scratch;
        }
    }
}
"""

STAGE_ALIAS = """shader P {
    compute {
        typedef float real;
        buffer float* outp;
        real scratch = real(2);
        void main() {
            scratch += outp[1];
            outp[0] = scratch;
        }
    }
}
"""

COMPOSITE_ALIASES = """shader P {
    typedef float real;
    typedef real real2;
    struct Pair {
        real x;
        int n;
    };
    compute {
        buffer float* outp;
        real2 values[2];
        vec2<real> v;
        Pair pair;
        void main() {
            values[0] = outp[1];
            v = vec2(values[0]);
            pair.x = v.y;
            outp[0] = pair.x;
        }
    }
}
"""

HELPER = """shader P {
    typedef vec4 colour;
    fragment {
        colour acc;
        void add(vec4 c) {
            acc += c;
        }
        vec4 main(vec4 c @ COLOR) @ gl_FragColor {
            acc = vec4(0.0);
            add(c);
            return acc;
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


def _glslang(tmp_path, code, stage):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    (tmp_path / f"out.{stage}").write_text(code)
    result = subprocess.run(
        [validator, "-S", stage, f"out.{stage}"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source", [PROGRAM_ALIAS, STAGE_ALIAS])
@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_hlsl_and_slang_declare_a_static_global_of_the_aliased_type(
    tmp_path, source, backend
):
    code = _flat(_translate(tmp_path, source, backend))
    assert "static float scratch;" in code
    # The declaration is the first mention of the variable.
    assert code.index("scratch") == code.index("static float scratch;") + len(
        "static float "
    )


@pytest.mark.parametrize("source", [PROGRAM_ALIAS, STAGE_ALIAS])
def test_glsl_declares_the_variable_at_module_scope(tmp_path, source):
    code = _translate(tmp_path, source, "opengl")
    assert "float scratch = 0.0;" in _flat(code)
    _glslang(tmp_path, code, "comp")


def test_stage_alias_initializer_runs_at_the_start_of_the_entry(tmp_path):
    code = _flat(_translate(tmp_path, STAGE_ALIAS, "directx"))
    assert "scratch = float(2);" in code
    assert "real" not in code


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_arrays_vectors_and_structures_through_aliases_are_declared(tmp_path, backend):
    code = _flat(_translate(tmp_path, COMPOSITE_ALIASES, backend))
    assert "static float values[2];" in code
    assert "static float2 v;" in code
    assert "static Pair pair;" in code


def test_glsl_declares_composite_alias_variables(tmp_path):
    code = _translate(tmp_path, COMPOSITE_ALIASES, "opengl")
    flat = _flat(code)
    for declaration in ("float values[2]", "vec2 v", "Pair pair"):
        assert declaration in flat
    _glslang(tmp_path, code, "comp")


def test_metal_passes_an_alias_typed_variable_to_the_helpers(tmp_path):
    code = _flat(_translate(tmp_path, HELPER, "metal"))
    assert "void add(float4 c, thread float4& acc)" in code
    assert "float4 acc = {};" in code


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_alias_typed_variable_used_by_a_helper_is_shared(tmp_path, backend):
    code = _flat(_translate(tmp_path, HELPER, backend))
    assert "static float4 acc;" in code


def test_hoisted_declaration_carries_the_aliased_type():
    program = _parse(COMPOSITE_ALIASES)
    assert hoist_stage_private_variables(program, "directx") == 3
    hoisted = {v.name: v for v in program.global_variables if not v.is_type_alias}
    values = hoisted["values"].var_type
    assert isinstance(values, A.ArrayType)
    assert isinstance(values.element_type, A.PrimitiveType)
    assert values.element_type.name == "float"
    assert hoisted["v"].var_type.element_type.name == "float"
    assert hoisted["v"].vtype is hoisted["v"].var_type
    assert hoisted["pair"].var_type.name == "Pair"
    assert hoisted["values"].qualifiers == ["static"]


def test_alias_carrying_a_qualifier_is_left_to_the_generators():
    program = _parse(
        PROGRAM_ALIAS.replace("typedef float real;", "typedef const float real;")
    )
    assert hoist_stage_private_variables(program, "directx") == 0
    stage = next(iter(program.stages.values()))
    assert "scratch" in [v.name for v in stage.local_variables]


def test_alias_of_an_opaque_type_is_left_to_the_generators():
    program = _parse(
        PROGRAM_ALIAS.replace("typedef float real;", "typedef sampler2D real;")
    )
    assert hoist_stage_private_variables(program, "opengl") == 0
