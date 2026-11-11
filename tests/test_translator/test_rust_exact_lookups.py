"""Rust lookups that used to pass an unknown key through as its own Rust spelling.

Each lookup now either names the Rust spelling exactly or refuses the key:

- operators resolve through the shared C-family table, and unary ``+``, which Rust lacks, is written as its operand;
- a call to a built-in function of another language that the Rust runtime does not define is refused instead of
  being written as a call nothing defines;
- ray-tracing calls with an argument count the runtime function does not take are refused;
- a reserved ``gl_``/``SV_`` semantic the generator does not know is refused, and a user semantic is kept;
- a built-in type template (``vector<float, 3>``) is never written verbatim;
- a user type is never classified as a resource because of a prefix of its name.
"""

from __future__ import annotations

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.rust_codegen import RustCodeGen
from crosstl.translator.codegen.rust_exact_lookups import (
    RUST_RAY_STAGE_METADATA_NAMES,
    RUST_RESOURCE_KINDS,
    call_names_resolving_to,
    rust_operator,
)
from crosstl.translator.codegen.rust_refusals import RustLoweringRefused
from tests.lowering_refusal_helpers import refused


def _rust(source):
    return RustCodeGen().generate(parse(source))


def _compute(body, prelude=""):
    return (
        "shader S {\n"
        f"{prelude}"
        "  compute {\n"
        "    void main() {\n"
        "      int a = 4;\n"
        "      float f = 0.5;\n"
        "      uint u = 3u;\n"
        f"      {body}\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def test_unary_plus_is_written_as_its_operand():
    code = _rust(_compute("int r = +a; int n = ~a; int m = -a;"))
    assert "let r: i32 = (a);" in code
    assert "(+a)" not in code
    assert "let n: i32 = (!a);" in code
    assert "let m: i32 = (-a);" in code


@pytest.mark.parametrize(
    ("token", "spelling"),
    [("PLUS", "+"), ("BITWISE_NOT", "~"), ("ASSIGN_SHIFT_LEFT", "<<="), ("%", "%")],
)
def test_operators_resolve_through_the_shared_table(token, spelling):
    assert rust_operator(token) == spelling


def test_an_operator_crossgl_does_not_define_is_refused():
    message = refused(RustLoweringRefused, rust_operator, "<=>")
    assert "'<=>' has no Rust spelling" in message


@pytest.mark.parametrize(
    "call",
    [
        "float r = log10(f);",
        "bool r = CheckAccessFullyMapped(u);",
        "uint r = GetRenderTargetSampleCount();",
    ],
)
def test_foreign_builtin_calls_without_a_runtime_function_are_refused(call):
    message = refused(RustLoweringRefused, _rust, _compute(call))
    name = call.split("= ", 1)[1].split("(", 1)[0]
    assert (
        f"call to {name}: {name} is a built-in function of another language" in message
    )
    assert "the Rust runtime defines no function of that name" in message


@pytest.mark.parametrize(
    ("call", "written"),
    [
        ("float r = sqrt(f);", "sqrt(f)"),
        ("float r = clamp(f, 0.0, 1.0);", "clamp(f, 0.0, 1.0)"),
        ("float r = smoothstep(0.0, 1.0, f);", "smoothstep(0.0, 1.0, f)"),
        ("float r = lerp(f, 1.0, 0.5);", "lerp(f, 1.0, 0.5)"),
    ],
)
def test_runtime_functions_are_still_called(call, written):
    assert written in _rust(_compute(call))


def test_a_declared_function_may_reuse_a_builtin_name():
    source = (
        "shader S {\n"
        "  float log10(float x) { return x * 2.0; }\n"
        "  compute { void main() { float r = log10(0.5); } }\n"
        "}\n"
    )
    assert "log10(0.5)" in _rust(source)


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


def test_trace_ray_with_the_runtime_argument_count_is_called():
    code = _rust(_RAY_PROGRAM % "TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);")
    assert "trace_ray(*SCENE, 0, 255, 0, 1, 0, ray, payload);" in code


def test_trace_ray_with_another_argument_count_is_refused():
    message = refused(
        RustLoweringRefused,
        _rust,
        _RAY_PROGRAM % "TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload, payload);",
    )
    assert "TraceRay with 9 arguments: the Rust runtime's trace_ray takes 8 or 11" in (
        message
    )


def test_every_ray_stage_has_a_metadata_name():
    generator = RustCodeGen()
    assert set(RUST_RAY_STAGE_METADATA_NAMES) == generator.rust_ray_stage_names()
    generator.rust_ray_stage_metadata_name("ray_shadow")
    assert generator.refusals.reasons == [
        "ray-tracing stage ray_shadow: not a CrossGL ray-tracing stage"
    ]


def test_unmapped_semantics_are_kept_only_outside_reserved_namespaces():
    generator = RustCodeGen()
    assert generator.map_semantic("TEXCOORD0") == "texcoord(0)"
    assert generator.map_semantic("MY_VARYING") == "MY_VARYING"
    assert generator.refusals.reasons == []
    generator.map_semantic("SV_ShadingRate")
    generator.map_semantic("gl_ShadingRateEXT")
    assert generator.refusals.reasons == [
        "semantic SV_ShadingRate: names a built-in variable with no Rust equivalent",
        "semantic gl_ShadingRateEXT: names a built-in variable with no Rust equivalent",
    ]


def test_builtin_type_templates_are_not_written_verbatim():
    message = refused(
        RustLoweringRefused,
        _rust,
        _compute("vector<float, 3> v = vector<float, 3>(1.0, 2.0, 3.0);"),
    )
    assert "type vector<...>: built-in type template with no Rust spelling" in message


def test_user_generic_templates_keep_their_name():
    source = (
        "shader S {\n"
        "  struct Pair<T> { T first; T second; };\n"
        "  compute { void main() { Pair<float> p; p.first = 1.0; } }\n"
        "}\n"
    )
    assert "Pair<f32>" in _rust(source)


def test_a_structure_named_like_a_texture_is_not_a_resource():
    generator = RustCodeGen()
    assert generator.rust_resource_kind("TextureParams") is None
    assert generator.rust_resource_kind("ImageInfo") is None
    assert generator.rust_resource_kind("sampler2D") == "texture"
    assert generator.rust_resource_kind("image2D") == "image"
    assert generator.rust_resource_kind("Image2D") == "image"
    assert generator.rust_resource_kind("accelerationStructureEXT") == (
        "acceleration_structure"
    )


def test_every_mapped_resource_type_has_a_kind():
    generator = RustCodeGen()
    spellings = {
        str(rust).split("<", 1)[0].rsplit("::", 1)[-1]
        for rust in generator.type_mapping.values()
    }
    resource_spellings = {
        spelling
        for spelling in spellings
        if spelling.startswith(("Image", "Texture", "DepthTexture", "Sampler"))
        or spelling == "RayTracingAccelerationStructure"
    }
    assert resource_spellings <= set(RUST_RESOURCE_KINDS)


def test_constant_buffer_member_types_use_the_one_type_mapping():
    generator = RustCodeGen()
    for name in ("float4", "int2", "mat3", "float2x2", "int8", "float16"):
        assert generator.map_type_to_rust(name) == generator.map_type(name)
    assert "Vec8" not in generator.map_type_to_rust("int8")


def test_sqrt_constraints_follow_the_function_map():
    generator = RustCodeGen()
    assert "sqrt" in call_names_resolving_to(generator.function_map, {"sqrt"})
    assert call_names_resolving_to({"rsqrt": "inverse_sqrt"}, {"sqrt"}) == {"sqrt"}
    assert call_names_resolving_to({"sqrt": "root"}, {"sqrt"}) == frozenset()


def test_unknown_wave_operations_have_no_result_type():
    assert RustCodeGen().wave_op_result_type("WaveNotAnOperation", []) is None


def test_an_uncollected_static_is_refused():
    generator = RustCodeGen()
    generator.static_symbol_names = {"counter": "COUNTER"}
    assert generator.static_symbol_name("counter") == "COUNTER"
    generator.static_symbol_name("other")
    assert generator.refusals.reasons == [
        "static other: not collected before generation"
    ]
