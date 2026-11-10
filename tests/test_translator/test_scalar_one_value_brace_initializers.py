"""Scalar brace lists carry values, not aggregate constructors (#3760)."""

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.semantics.scalar_brace_values import (
    zero_empty_scalar_initializers,
)

SHADER = """
shader Probe {
    buffer float* outp;
    float g = {1.0};
    const float c = {2};
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        float s = {3.0};
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float a = {outp[i]};
            float b = {int(i)};
            outp[i] = g + c + s + a + b;
        }
    }
}
"""
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
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", TARGETS)
def test_singleton_matches_explicit_destination_conversion(tmp_path, backend):
    control = (
        SHADER.replace("{1.0}", "1.0")
        .replace("{2}", "float(2)")
        .replace("{3.0}", "3.0")
        .replace("{outp[i]}", "float(outp[i])")
        .replace("{int(i)}", "float(int(i))")
    )
    assert translate(tmp_path, SHADER, backend) == translate(tmp_path, control, backend)


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize("values", ["1, 2", "{1, 2}"])
def test_overfull_scalar_is_a_typed_refusal(tmp_path, backend, values):
    source = SHADER.replace("{1.0}", "{" + values + "}")
    with pytest.raises(TranslationDiagnostic) as caught:
        translate(tmp_path, source, backend)
    assert type(caught.value).__name__ == "ScalarBraceInitializerError"
    assert caught.value.target == {"spirv": "vulkan"}.get(backend, backend)
    assert "scalar-brace-initializer" in caught.value.code
    assert "2" in str(caught.value)


@pytest.mark.parametrize("backend", ["directx", "opengl", "wgsl", "spirv", "metal"])
@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_imported_scalar_aliases_and_constants(tmp_path, backend, suffix):
    source = """
    typedef float real_t;
    extern "C" __global__ void first(float* out) {
        real_t a = {1};
        const int n = {2};
        bool b = {n};
        out[0] = a + (float)n + (float)b;
    }
    """
    control = (
        source.replace("{1}", "(float)1").replace("{2}", "2").replace("{n}", "(bool)n")
    )
    assert translate(tmp_path, source, backend, suffix) == translate(
        tmp_path, control, backend, suffix
    )


@pytest.mark.parametrize("backend", ["directx", "opengl", "wgsl", "metal", "spirv"])
def test_nested_singletons_and_empty_value(tmp_path, backend):
    source = SHADER.replace("{1.0}", "{{{1.0}}}").replace("{2}", "{{}}")
    control = SHADER.replace("{2}", "{}")
    assert translate(tmp_path, source, backend) == translate(tmp_path, control, backend)


@pytest.mark.parametrize("backend", ["cgl", "crossgl"])
def test_crossgl_carries_singleton_and_overfull_lists(tmp_path, backend):
    output = translate(tmp_path, SHADER.replace("{1.0}", "{1.0, 2.0}"), backend)
    assert "{1.0, 2.0}" in output
    assert "{outp[i]}" in output


@pytest.mark.parametrize("backend", ["directx", "opengl", "wgsl", "metal", "spirv"])
@pytest.mark.parametrize(
    "scalar",
    ["bool", "int", "uint", "i8", "u8", "i16", "u16", "i64", "u64", "half", "double"],
)
def test_scalar_conversion_matches_brace_free_control(tmp_path, backend, scalar):
    source = """
    shader Probe {
        compute {
            buffer float* outp;
            void main() {
                int n = 3;
                SCALAR value = INITIALIZER;
                outp[0] = float(value);
            }
        }
    }
    """.replace("SCALAR", scalar)
    control = source.replace("INITIALIZER", f"{scalar}(n)")
    braced = source.replace("INITIALIZER", "{n}")
    try:
        expected = translate(tmp_path, control, backend)
    except TranslationDiagnostic as error:
        # A target without this width must refuse just as it does without braces.
        with pytest.raises(type(error)) as caught:
            translate(tmp_path, braced, backend)
        assert caught.value.code == error.code
    else:
        assert translate(tmp_path, braced, backend) == expected


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize("kind", ["array", "vector", "matrix", "struct"])
def test_pass_leaves_non_scalar_brace_values_alone(backend, kind):
    f32 = A.PrimitiveType("float")
    types = {
        "array": A.ArrayType(f32, 2),
        "vector": A.VectorType(f32, 2),
        "matrix": A.MatrixType(f32, 2, 2),
        "struct": A.NamedType("Pair"),
    }
    value = A.ArrayLiteralNode(
        [
            A.LiteralNode(1, A.PrimitiveType("int")),
            A.LiteralNode(2, A.PrimitiveType("int")),
        ]
    )
    declaration = A.VariableNode("v", types[kind], value)
    program = A.ASTNode()
    program.declarations = [declaration]
    assert zero_empty_scalar_initializers(program, backend) == 0
    assert declaration.initial_value is value


def test_scalar_alias_chain_is_converted_once():
    value = A.ArrayLiteralNode([A.IdentifierNode("n")])
    declaration = A.ConstantNode("c", A.NamedType("Real"), value)
    program = A.ASTNode()
    program.declarations = [
        A.VariableNode("Real", A.NamedType("Small"), is_type_alias=True),
        A.VariableNode("Small", A.PrimitiveType("i16"), is_type_alias=True),
        declaration,
    ]
    assert zero_empty_scalar_initializers(program, "wgsl") == 1
    assert declaration.value.function.name == "i16"
    assert declaration.value.arguments == value.elements
    assert zero_empty_scalar_initializers(program, "wgsl") == 0
