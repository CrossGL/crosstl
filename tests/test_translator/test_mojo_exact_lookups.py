"""Mojo lookups that used to pass an unknown key through as its own Mojo spelling.

Each lookup now either names the Mojo spelling exactly or refuses the key:

- operators resolve through the shared C-family table, with Mojo's ``and``/``or``/``not`` keywords;
- ray-tracing calls with an argument count the runtime helper does not take are refused;
- a built-in type template of another language (``vector<float, 3>``, ``matrix<float, 2, 3>``) is refused, while
  Mojo's own parameterised types keep their spelling;
- a resource is classified by its exact Mojo runtime spelling.
"""

from __future__ import annotations

import pytest

import crosstl.translator.codegen.mojo_codegen as mojo_codegen
from crosstl.translator import parse
from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.codegen.mojo_exact_lookups import (
    MOJO_RESOURCE_KINDS,
    mojo_operator,
)
from crosstl.translator.codegen.mojo_refusals import MojoLoweringRefused
from tests.lowering_refusal_helpers import refused


def _mojo(source):
    return MojoCodeGen().generate(parse(source))


def _compute(body):
    return (
        "shader S {\n"
        "  compute {\n"
        "    void main() {\n"
        "      int a = 4;\n"
        f"      {body}\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


@pytest.mark.parametrize(
    ("token", "spelling"),
    [
        ("PLUS", "+"),
        ("&&", "and"),
        ("LOGICAL_OR", "or"),
        ("NOT", "not"),
        ("!", "not"),
        ("BITWISE_NOT", "~"),
        ("ASSIGN_SHIFT_RIGHT", ">>="),
        ("POST_INCREMENT", "++"),
    ],
)
def test_operators_resolve_through_the_shared_table(token, spelling):
    assert mojo_operator(token) == spelling


def test_an_operator_crossgl_does_not_define_is_refused():
    message = refused(MojoLoweringRefused, mojo_operator, "<=>")
    assert "'<=>' has no Mojo spelling" in message


def test_logical_operators_are_written_as_keywords():
    code = _mojo(_compute("bool t = (a > 1) && (a < 9) || !(a == 2);"))
    assert "(((a > 1) and (a < 9)) or (not (a == 2)))" in code


_RAY_PROGRAM = """
shader S {
    struct RayPayload { vec3 color; };
    accelerationStructureEXT scene @binding(3) @set(2);
    ray_generation {
        void main() {
            RayDesc ray;
            RayPayload payload;
            %s
        }
    }
}
"""


def test_trace_ray_with_the_helper_argument_count_is_called():
    code = _mojo(_RAY_PROGRAM % "TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);")
    assert "trace_ray(" in code


def test_trace_ray_with_another_argument_count_is_refused():
    message = refused(
        MojoLoweringRefused,
        _mojo,
        _RAY_PROGRAM % "TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload, payload);",
    )
    assert (
        "TraceRay with 9 arguments: the Mojo runtime helper trace_ray takes 8 or 11"
        in message
    )


@pytest.mark.parametrize(
    "declaration",
    ["vector<float, 3> v;", "matrix<float, 2, 3> m;", "vector<int, 2> iv;"],
)
def test_builtin_type_templates_of_other_languages_are_refused(declaration):
    template = declaration.split("<", 1)[0]
    message = refused(MojoLoweringRefused, _mojo, _compute(declaration))
    assert f"type {template}<...>: built-in type template with no Mojo spelling" in (
        message
    )


def test_mojo_parameterised_types_keep_their_spelling():
    generator = MojoCodeGen()
    assert generator.mojo_native_generic_mapped_type("SIMD<DType.float32, 4>") == (
        "SIMD[DType.float32, 4]"
    )
    assert generator.mojo_native_generic_mapped_type(
        "UnsafePointer<Int64, MutAnyOrigin>"
    ) == ("UnsafePointer[Int64, MutAnyOrigin]")


def test_every_resource_spelling_the_generator_writes_has_a_kind():
    generator = MojoCodeGen()
    spellings = {
        str(spelling)
        for spelling in list(generator.type_mapping.values()) + list(
            mojo_codegen.MOJO_RESOURCE_TYPE_MAPPING.values()
        )
    }
    for aliases in mojo_codegen.MOJO_HLSL_WRITABLE_TEXTURE_TYPE_MAPPING.values():
        spellings |= set(aliases)
    resources = {
        spelling
        for spelling in spellings
        if spelling.startswith(
            ("Image", "IImage", "UImage", "Texture", "Sampler", "RayTracing")
        )
    }
    assert resources == set(MOJO_RESOURCE_KINDS)


@pytest.mark.parametrize(
    ("type_name", "kind"),
    [
        ("sampler2DShadow", "texture"),
        ("sampler2D", "texture"),
        ("image2D", "image"),
        ("uimage3D", "image"),
        ("Texture2DInt", "texture"),
        ("accelerationStructureEXT", "acceleration_structure"),
        ("ImageParams", None),
        ("TextureSettings", None),
    ],
)
def test_resource_kinds_follow_exact_spellings(type_name, kind):
    assert MojoCodeGen().mojo_resource_kind(type_name) == kind


def test_integer_types_are_recognised_by_source_or_mojo_spelling():
    generator = MojoCodeGen()
    assert generator.is_scalar_integer_type("int")
    assert generator.is_scalar_integer_type("Int32")
    assert not generator.is_scalar_integer_type("float")
    assert not generator.is_scalar_integer_type("MyInt")
