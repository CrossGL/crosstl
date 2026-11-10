"""A one-value brace initializer of a plain enumeration is its value (#4190).

C++ initializes ``Mode m = {Mode::B};`` to ``Mode::B``. WGSL output wrote ``array(Mode_B)``, SPIR-V output a composite
of ``int``, Rust output ``[Mode::B]`` and Mojo output an ``InlineArray``, and GLSL output refused the program as an
aggregate initializer. A list with more than one value is refused, as it is for a scalar.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.semantics.scalar_brace_values import (
    ScalarBraceInitializerError,
    zero_empty_scalar_initializers,
)

SHADER = textwrap.dedent("""
    shader P {
        enum Mode { A, B };
        compute {
            buffer float* outp;
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                Mode m = {Mode::B};
                const Mode k = {Mode::A};
                outp[i] = float(int(m)) + float(int(k));
            }
        }
    }
""")

KERNEL = textwrap.dedent("""
    enum class Mode { A, B };
    extern "C" __global__ void first(float* out) {
        Mode m = {Mode::B};
        out[0] = (float)m;
    }
""")

TARGETS = [
    "directx",
    "opengl",
    "wgsl",
    "spirv",
    "metal",
    "cuda",
    "hip",
    "slang",
    "rust",
    "mojo",
]


def translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def without_braces(source):
    return source.replace("{Mode::B}", "Mode::B").replace("{Mode::A}", "Mode::A")


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_one_value_enum_initializer_is_its_value(tmp_path, backend, nested):
    source = SHADER
    if nested:
        source = source.replace("{Mode::B}", "{{Mode::B}}")
    expected = translate(tmp_path, without_braces(SHADER), backend)
    assert translate(tmp_path, source, backend) == expected


@pytest.mark.parametrize("backend", TARGETS)
def test_one_value_initializer_of_an_enum_with_an_underlying_type(tmp_path, backend):
    source = SHADER.replace("enum Mode {", "enum Mode : uint {")
    expected = translate(tmp_path, without_braces(source), backend)
    assert translate(tmp_path, source, backend) == expected


@pytest.mark.parametrize("backend", TARGETS)
def test_one_value_enum_initializer_from_cuda_is_its_value(tmp_path, backend):
    expected = translate(tmp_path, without_braces(KERNEL), backend, ".cu")
    assert translate(tmp_path, KERNEL, backend, ".cu") == expected


@pytest.mark.parametrize("backend", TARGETS)
def test_nested_empty_enum_initializer_is_the_empty_initializer(tmp_path, backend):
    expected = translate(tmp_path, SHADER.replace("{Mode::B}", "{}"), backend)
    source = SHADER.replace("{Mode::B}", "{{}}")
    assert translate(tmp_path, source, backend) == expected


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize("values", ["Mode::A, Mode::B", "{Mode::A, Mode::B}"])
def test_overfull_enum_initializer_is_a_typed_refusal(tmp_path, backend, values):
    source = SHADER.replace("{Mode::B}", "{" + values + "}")
    with pytest.raises(TranslationDiagnostic) as caught:
        translate(tmp_path, source, backend)
    assert type(caught.value).__name__ == "ScalarBraceInitializerError"
    assert caught.value.target == {"spirv": "vulkan"}.get(backend, backend)
    assert "'m'" in str(caught.value)
    assert "got 2" in str(caught.value)


PAYLOAD_ENUM = textwrap.dedent("""
    shader P {
        enum Mode { First(float), Second }
        compute {
            buffer float* outp;
            void main() {
                Mode m = {Mode::Second};
                outp[0] = 1.0;
            }
        }
    }
""")


@pytest.mark.parametrize("backend", ["metal", "hip", "slang"])
def test_one_value_initializer_of_an_enum_with_payloads_keeps_its_braces(
    tmp_path, backend
):
    # Such an enumeration is a structure on these targets; its brace list is an aggregate initializer.
    output = translate(tmp_path, PAYLOAD_ENUM, backend)
    assert "Mode m = {" in output


@pytest.mark.parametrize("backend", TARGETS)
def test_brace_list_of_an_enum_array_is_not_a_scalar_list(tmp_path, backend):
    # An array of the enumeration takes as many values as it has elements.
    source = without_braces(SHADER).replace(
        "Mode m = Mode::B;", "Mode arr[2] = {Mode::A, Mode::B}; Mode m = arr[1];"
    )
    translate(tmp_path, source, backend)


def _alias_program(values):
    declaration = A.VariableNode(
        "v",
        A.NamedType("ModeAlias"),
        A.ArrayLiteralNode([A.IdentifierNode(value) for value in values]),
    )
    program = A.ASTNode()
    program.declarations = [
        A.EnumNode("Mode", []),
        A.VariableNode("ModeAlias", A.NamedType("Mode"), is_type_alias=True),
        declaration,
    ]
    return program, declaration


@pytest.mark.parametrize("backend", TARGETS)
def test_one_value_initializer_through_an_enum_alias_is_its_value(backend):
    program, declaration = _alias_program(["Mode_B"])
    assert zero_empty_scalar_initializers(program, backend) == 1
    assert isinstance(declaration.initial_value, A.IdentifierNode)
    assert declaration.initial_value.name == "Mode_B"


@pytest.mark.parametrize("backend", TARGETS)
def test_overfull_initializer_through_an_enum_alias_is_refused(backend):
    program, _ = _alias_program(["Mode_A", "Mode_B"])
    with pytest.raises(ScalarBraceInitializerError):
        zero_empty_scalar_initializers(program, backend)
